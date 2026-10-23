"""Step 4c: does this image's CUDA stack run on a GB10? One minute, a few hundred MB of GPU memory.

Checks, each reported on its own line: torch sees the GPU; a bf16 matmul is right; a Triton kernel
compiles (the image's ptxas) and runs; a compiled vLLM CUDA op (built for SM 12.0) runs on SM 12.1;
FlashInfer imports. Run by smoke-gpu.sh in a throwaway container with no network.
"""
import time
import traceback

results = []


def check(name):
    def wrap(fn):
        t = time.time()
        try:
            detail = fn()
            results.append((name, "OK", detail, time.time() - t))
        except Exception as exc:  # noqa: BLE001
            results.append((name, "FAIL", f"{type(exc).__name__}: {exc}", time.time() - t))
            traceback.print_exc()
        return fn
    return wrap


@check("cuda device")
def _device():
    import torch
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    return (f"{p.name} sm_{p.major}{p.minor}, driver CUDA {torch.cuda.driver_version() if hasattr(torch.cuda, 'driver_version') else '?'}"
            f", runtime {torch.version.cuda}, free {free / 2**30:.1f} of {total / 2**30:.1f} GiB")


@check("bf16 matmul")
def _matmul():
    import torch
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    err = (a @ b).float().sub(a.float() @ b.float()).abs().max().item()
    assert err < 1.0, f"max abs error {err}"
    return f"max abs error vs fp32 {err:.3f}"


@check("triton kernel")
def _triton():
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def add(x, y, out, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tl.store(out + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)

    n = 1 << 20
    x, y = torch.rand(n, device="cuda"), torch.rand(n, device="cuda")
    out = torch.empty_like(x)
    add[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    assert torch.allclose(out, x + y)
    return f"triton {triton.__version__}"


@check("vllm compiled op")
def _vllm_op():
    import torch
    from vllm import _custom_ops as ops
    x = torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(4096, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    ops.rms_norm(out, x, w, 1e-6)
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    err = (out.float() - ref).abs().max().item()
    assert err < 0.1, f"max abs error {err}"
    return f"rms_norm max abs error {err:.4f}"


@check("flashinfer import")
def _flashinfer():
    import flashinfer
    return f"flashinfer {flashinfer.__version__}"


if __name__ == "__main__":
    for name, status, detail, secs in results:
        print(f"{status:4} {name:18} {secs:6.1f}s  {detail}")
    print("SMOKE " + ("PASS" if all(r[1] == "OK" for r in results) else "FAIL"))
