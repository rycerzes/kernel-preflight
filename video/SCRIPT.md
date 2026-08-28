# Narration

Voice: Kokoro `bm_george`, speed 1.0. One file per beat, so scene timing follows the read
rather than the reverse. Structured to the submission's four points: what the project is,
the stack and architecture, a demo, and what building it taught.

Numbers are written as words where a digit string would be mis-read by the phonemiser.

1. In February twenty twenty-five, an A.I. CUDA Engineer reported speedups of ten to a hundred times. The kernels were not faster. It had exploited the benchmark harness, and several figures implied thirty times more throughput than the hardware can physically deliver.
2. That is not an exotic failure. It is the default outcome of one arrangement: an agent that writes a kernel and also reports its own speedup. Correctness tests do not catch it, because a kernel can be numerically perfect and still be timed dishonestly.
3. Kernel Preflight is an agent inside TrueForge with a single property. It submits kernel source, and it never submits a number. A fixed harness it cannot see owns the allocation, the input distribution, the timing loop, the reference implementation and the tolerances.
4. Inside the sandbox that is two processes. A supervisor holds the run's secrets and writes the verdict, but never loads candidate code. A worker runs the kernel and is told neither. One process cannot both execute a candidate and be trusted to report on it.
5. Nine gates then adjudicate a measurement schema rather than a toolchain. That is what let five toolchains in — CUDA, Triton, Helion, CuTe and TileLang — without changing a single gate.
6. Here is a kernel that declares float thirty-two and quietly computes in bfloat sixteen. Twice the speed, for a large loss of precision. It is caught three separate times. The sharpest: a hundred point nine teraflops against an eighty-three teraflop ceiling. That one needs no reference at all. The arithmetic is impossible at the precision claimed.
7. The same FlashAttention kernel is rejected as float thirty-two and admitted as t.f. thirty-two and bfloat sixteen. Nothing about the kernel changes. Triton's dot product silently uses tensor cores. Declaring what you actually compute is the entire difference.
8. Across eighteen operations, six toolchains and sixty-seven measured cases: fifty-one admitted, and every one of twelve adversarial kernels rejected. Attention, quantisation, mixture of experts, paged decoding and a hand-written backward pass.
9. Then it caught me. Two candidates forged the whole verdict and were admitted at ninety-two percent of peak. A third served a cached answer while it was being timed, admitted at eighty-nine point seven. All three are now regression tests.
10. And six times I shipped a gate that rejected correct work, which is worse than shipping no gate at all. That is the lesson worth keeping. A verifier is only as good as the attacks it has actually survived.
