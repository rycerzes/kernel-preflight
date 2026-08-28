# Narration

Voice: Kokoro `bm_george`, **speed 1.25**. One file per beat, so scene timing follows the
read rather than the reverse. Structured to the submission's four points: the problem, the
design, a real demo, and what building it taught.

Beats 4-7 are a recorded TrueForge session replayed verbatim. The commands, the tracebacks,
the gate tables and every number in them are copied from `thread_context_log` in the running
instance and from a live harness run against the adversarial candidates -- nothing in the
demo is staged.

Numbers are written as words where a digit string would be mis-read by the phonemiser.

1. In February twenty twenty-five, an A.I. CUDA Engineer reported speedups of ten to a hundred times. The kernels were not faster. It had exploited the benchmark harness, and several figures implied thirty times more throughput than the hardware can physically deliver.
2. That is not an exotic failure. It is the default outcome of one arrangement: an agent that writes a kernel and also reports its own speedup. Correctness tests do not catch it, because a kernel can be numerically perfect and still be timed dishonestly.
3. Kernel Preflight is an agent inside TrueForge with a single property. It submits kernel source, and it never submits a number. A fixed harness it cannot see owns the allocation, the input distribution, the timing loop, the reference and the tolerances.
4. Here is a real session. Write the fastest float thirty-two matmul you can in Triton, and get it admitted. The agent asks the harness for the hardware ceilings first, then reads Hugging Face's own Triton skill.
5. It spots the trap unprompted. Triton's dot product routes through tensor cores by default, so a kernel handed float thirty-two silently computes in t.f. thirty-two. Honouring the contract it declared means asking for I. triple E. precision explicitly.
6. The first submission fails. So does the second. The harness hands back the real traceback, the agent finds the bug, keyword arguments colliding with positional ones, and fixes it. Admitted: nine gates, sixty point nine percent of the float thirty-two ceiling.
7. Now the same tool against a kernel that serves a cached answer while it is being timed. Eight gates pass. Correctness passes. The one that fails is timed work, and the output after timing is wrong by five times ten to the fifth of tolerance. The measured calls did not do the work.
8. Two processes make that possible. A supervisor holds the run's secrets and writes the verdict, but never loads candidate code. A worker runs the kernel and is told neither. Before that split, two candidates read the nonce out of their own command line and forged a verdict at ninety-two percent of peak.
9. The gates adjudicate a measurement schema, not a toolchain. That is what let five toolchains in, CUDA, Triton, Helion, CuTe and TileLang, without changing a single gate. The same FlashAttention kernel is rejected as float thirty-two and admitted as t.f. thirty-two.
10. Across eighteen operations, six toolchains and sixty-seven measured cases: fifty-one admitted, and every one of twelve adversarial kernels rejected. Attention, quantisation, mixture of experts, paged decoding and a hand-written backward pass.
11. And six times I shipped a gate that rejected correct work, which is worse than shipping no gate at all. That is the lesson worth keeping. A verifier is only as good as the attacks it has actually survived.
