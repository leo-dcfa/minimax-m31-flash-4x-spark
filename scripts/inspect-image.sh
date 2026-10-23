#!/usr/bin/env bash
# Step 4b: read the pulled image without a GPU and without network: what it runs, what it ships,
# where its libnccl lives (NCCL_PATH in site.env), and whether its vLLM has the recipe's flags.
#   bash ~/src/minimax-m31-flash-4x-spark/scripts/inspect-image.sh
set -euo pipefail
TAG=vllm/vllm-openai:minimax-m31-flash-86f582da

echo "== image config"
docker image inspect "$TAG" --format 'entrypoint={{json .Config.Entrypoint}} cmd={{json .Config.Cmd}} user={{json .Config.User}} workdir={{.Config.WorkingDir}} ports={{json .Config.ExposedPorts}}'

docker run --rm --network none --entrypoint bash "$TAG" -c '
set -u
echo "== versions"
python3 - <<PY
import importlib.metadata as m
for p in ("vllm", "torch", "triton", "flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache",
          "nvidia-nccl-cu13", "transformers", "b12x", "deep-gemm", "nvidia-cutlass-dsl"):
    try:
        print(f"{p:24} {m.version(p)}")
    except m.PackageNotFoundError:
        print(f"{p:24} -")
import torch
print("torch cuda", torch.version.cuda, "| nccl built", torch.cuda.nccl.version(), "| arch list", torch.cuda.get_arch_list())
PY
echo "== libnccl files"
find / -xdev -name "libnccl.so*" 2>/dev/null
echo "== what torch and vllm link for NCCL"
T=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
ldd "$T/lib/libtorch_cuda.so" | grep -i nccl || echo "libtorch_cuda: no direct libnccl link"
python3 -c "from vllm.distributed.device_communicators.pynccl_wrapper import find_nccl_library as f; print(\"pynccl loads\", f())" 2>/dev/null || true
echo "== vllm serve flags the recipe uses"
vllm serve --help=all 2>/dev/null | grep -oE -- "--(moe-backend|attention-config|kv-cache-memory-bytes|nnodes|node-rank|master-addr|headless|mm-encoder-tp-mode|limit-mm-per-prompt|disable-hybrid-kv-cache-manager|speculative-config|reasoning-parser|tool-call-parser)\b" | sort -u
echo "== reasoning and tool parsers named minimax"
python3 -c "
from vllm.reasoning import ReasoningParserManager as R
from vllm.tool_parsers import ToolParserManager as T
print(\"reasoning:\", [n for n in R.list_registered() if \"minimax\" in n])
print(\"tools:\", [n for n in T.list_registered() if \"minimax\" in n])" 2>&1 | tail -2
'
