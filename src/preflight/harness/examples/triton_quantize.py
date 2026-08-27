"""Per-row symmetric int8 quantise-dequantise in Triton.

Quantization is the one category KernelBenchX finds completely unsolved: 0 of 30
across every method they evaluated. It is not a hard formula, it is that nothing can
be transcribed. The kernel has to derive its own scale from a reduction, pick a
rounding mode, clamp to the integer range, and scale back.

Three details carry the whole thing:

* **Round to nearest *even*, not merely to nearest.** `libdevice.rint` matches
  `torch.round`; `tl.math` has only `floor` and `ceil` in Triton 3.7, and there is no
  `tl.math.rint`. Writing `floor(v + 0.5)` instead would be round-half-up, which
  differs from the reference on exact ties -- and with a power-of-two scale ties are
  reachable rather than hypothetical, because `x / scale` is an exponent shift that
  leaves the mantissa intact, so any value whose last set bit lands on 2^-1 is exactly
  half way. Truncating is worse again: it biases every value toward zero by up to half
  a step, and since this op's tolerance is storage quantisation rather than an
  accumulation allowance, that bias is a correctness failure rather than noise.
* **The absmax is a max reduction, so it is exact.** There is no accumulation error
  to hide behind: a correct kernel lands within an ulp of the reference.
* **A zero row has no scale.** absmax is then zero, the division is an infinity, and
  the clamp turns it into a NaN. Guarding costs one `tl.where`.

One pass to find the scale, one to apply it. The row is read twice, which is what the
operation costs unless the row fits in registers.
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _quant_kernel(X, Y, stride, N, QMAX, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    absmax = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        v = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        absmax = tl.maximum(absmax, tl.abs(v))
    peak = tl.max(absmax)
    # The exponent read straight out of the float, not derived through log2.
    #
    # The reference takes frexp(absmax).exponent, so peak sits in [2^(e-1), 2^e) and
    # the scale is 2^(e-7). Reproducing that as ceil(log2(peak)) is wrong at every
    # exact power of two -- log2(2^m) is m, while frexp reports m+1 -- and fragile just
    # below one, where an inexact fp32 log2 can round up and shift the whole row's scale
    # by a factor of two. Extracting the IEEE-754 exponent field is exact and has
    # neither problem: bits 23..30 hold m + 127 with peak in [2^m, 2^(m+1)), so
    # e = m + 1 and the scale is 2^(m-6).
    bits = peak.to(tl.int32, bitcast=True)
    unbiased = ((bits >> 23) & 0xFF) - 127
    scale = libdevice.exp2((unbiased - 6).to(tl.float32))
    # An all-zero row has no scale; without this the division is inf and the clamp NaN.
    scale = tl.where(peak > 0.0, scale, 1.0)

    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        v = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        q = libdevice.rint(v / scale)
        q = tl.minimum(tl.maximum(q, -QMAX), QMAX)
        tl.store(Y + cols, q * scale, mask=mask)


def launch_candidate(inputs, out, meta):
    x = inputs["x"]
    rows, cols = x.shape
    block = min(triton.next_power_of_2(cols), 1024)
    _quant_kernel[(rows,)](
        x, out, x.stride(0), cols, float(meta["qmax"]), BLOCK=block, num_warps=8,
    )
