# MiniMax-M3.1-Flash on four DGX Sparks (TP4)

Serving MiniMax's own release of MiniMax-M3.1-Flash with vLLM across four GB10 systems (DGX Spark
or compatible) cabled in a switchless ring, tensor parallel 4. Built one step at a time: each step
is explained here before it runs, and the log at the bottom records what happened.

**Status: candidate.** Nothing serves yet. Steps 1-3 are done: every node holds a verified copy of
the weights. Step 4 is done: every node has the image, and it runs on a GB10. Step 5: the four
nodes are free. Step 6 (first boot, `scripts/launch.sh`) is written, not run.

Site-specific values (node names, addresses, paths, the API key file) live in `site.env`, which
is git-ignored: copy `site.env.example` and fill it in. The scripts read it when it exists.

Running a script on a node from another machine: put the remote command in single quotes, or
your local shell expands `~` before ssh sends it:

```
ssh <node> 'bash ~/src/minimax-m31-flash-4x-spark/scripts/download-status.sh'
```

## The model

Released 2026-10-23 by MiniMax.

| | |
|---|---|
| Size | ~428B parameters, 23B active per token (128 experts, 4 chosen per token, plus 1 shared expert) |
| Layers | 60, every one with MiniMax Sparse Attention (MSA): each token attends to the top 16 blocks of 128 tokens, picked by a small "indexer" |
| Attention | 64 query heads, 4 KV heads, head dim 128 |
| Context | up to 1,048,576 tokens |
| Inputs | text, images, video; text out |
| Draft head | DSpark (`dspark/`, 5 small layers) proposes several tokens a step, the big model checks them in one pass |
| Precision | trained with quantisation (QAT): routed experts NVFP4, attention and dense layers MXFP8 |
| On disk | 252.4 GB, 49 safetensors shards |
| Licence | MiniMax Community License: free for non-commercial use; commercial use needs "Built with MiniMax M3.1-Flash" shown and a one-time notice to api@minimax.io (written authorisation above $20M yearly revenue) |

These are MiniMax's own quantised weights: no third-party quant involved.

## Why four GB10s

Tensor parallel splits every layer's weights across the GPUs, so each node holds a quarter:
252 GB / 4 ≈ 63 GB a node. A GB10 has ~121 GiB of memory shared by the CPU and the GPU, so a
quarter of the model leaves room for the KV cache, activations, CUDA graphs and the OS. On two
nodes it would be ~126 GB each: it does not fit.

The cost of TP is talk: every layer ends with an all-reduce across the four nodes, so decode speed
depends on the CX7 links as much as on the GPUs.

## The ring

```
                 cable A
  rank 0 (head) ───────── rank 1
       │ cable D              │ cable B
  rank 3 ───────────────── rank 2 (across from the head)
                 cable C
```

No switch: each node has a CX7 cable to each of its two neighbours, and each cable is its own
subnet. Every node also has `ring0`, a dummy interface holding one address that is routed through
the neighbours, so any node can reach any other (the head reaches rank 2 through rank 1). vLLM and
NCCL bootstrap over `ring0`; the data moves over RoCE on the CX7 ports. Stock NCCL assumes a
switch, so the ring runs NCCL 2.30.7 patched for two PCI domains per node (the dual-PCI-domain
patch from FujitsuPolycom/sparkring) with `NCCL_SWITCHLESS_RING_ONLY=1` and `NCCL_ALGO=Ring`.
Rank order must follow the ring order.

## The engine

On release day no engine had been run on a GB10 with this model:

- **vLLM**: support is an open pull request (vllm-project/vllm#63232) plus a dedicated image,
  `vllm/vllm-openai:minimax-m31-flash` (arm64 included). Its fastest kernels (NVFP4 KV cache,
  CUTLASS sparse attention) are SM100 only (B200/GB200), but on other GPUs it falls back:
  Triton for the sparse attention with an fp8 KV cache, b12x or Marlin kernels for the MXFP8
  layers, b12x / CUTLASS / Marlin for the NVFP4 experts. The GB10 is SM 12.1. vLLM also ships
  its own config and processor classes for this model, so MiniMax's `.py` files are not needed.
- **SGLang**: MSA kernels are SM100 only. Not usable here.
- **NVIDIA Flare**: added the model the same day, "SM121 execution not run".

So: vLLM's image, pinned by digest. The open question is whether that image runs on a GB10 at
all (built with CUDA 13.4, compiled up to SM 12.0, while DGX OS's driver is CUDA 13.0); step 4
answers it in a minute.

Prior art: tdwyer ran the predecessor (MiniMax-M3, NVIDIA's NVFP4) on four Sparks with vLLM TP4,
Triton attention, fp8 KV and DSpark: ~42 tok/s on short prompts, ~29 tok/s at 60K context.

## Memory on a GB10

The CPU and the GPU share one pool of memory. Two habits keep a node from freezing:

- Cap long jobs that read or write a lot of data (downloads, checksums, copies). The scripts here
  run under `systemd-run --user --scope -p MemoryHigh=... -p MemoryMax=...`, which charges the
  job's page cache to the job and kills only the job if it runs away.
- Drop the page cache (`sync; echo 3 > /proc/sys/vm/drop_caches`) on every node before loading
  the model, and keep it low during the load.

## Steps

| # | Step | Where | Status |
|---|---|---|---|
| 1 | Download the weights at a pinned revision (`scripts/download.sh`) | rank 3 | done, 55 min |
| 2 | Verify every file against the hash Hugging Face publishes (`scripts/verify.sh`) | rank 3 | done, 2 min, 66/66 OK |
| 3 | Copy to the other three nodes over the CX7 cables, verify on each (`scripts/copy.sh`) | rank 3 → rank 0, then rank 0 → ranks 1 and 2 | done, 28 min, 66/66 OK on each |
| 4 | Pull the vLLM image by digest and read what it runs; find its libnccl; one-minute GPU smoke test | all four | done: smoke PASS, image on all four |
| 5 | Free the four nodes, drop the page caches | all four | freed; caches dropped by launch.sh's flushers |
| 6 | First boot, target model alone (no DSpark): `scripts/launch.sh` on the head | all four | written, not run |
| 7 | Validate: answers, tool call, image, effort levels, speed | | |
| 8 | Turn DSpark on, measure again | | |

The download runs on a worker node rather than the head, which also runs the API server.

Step 3 is driven from the head, so it only needs the ssh keys the head already has to the other
nodes: the head pulls from rank 3 over cable D, checks its copy, then pushes to ranks 1 and 2 at
once (rank 2's bytes cross rank 1, which forwards them), and each of them checks its copy. A bad
copy stops the chain before it is passed on. Each rsync is memory-capped at both ends.

## Files

- `recipe.yaml`: what runs, where it comes from, the memory plan, the vLLM flags (draft).
- `nccl.env`: the NCCL settings for the switchless ring, one comment per group.
- `site.env.example`: the site-specific values; copy to `site.env` (git-ignored).
- `scripts/download.sh`, `scripts/download-status.sh`: step 1.
- `manifest/76faee1b.json`: the expected size and hash of each of the 66 files, from the Hub's
  API for the pinned revision (sha256 for the 50 LFS files, blob sha1 for 16 small text files).
- `scripts/verify.sh`, `scripts/verify.py`: step 2, and the check after each copy in step 3.
- `scripts/copy.sh`: step 3, `--from <host>` pulls to this node, `--to <host> ...` pushes in parallel.
- `scripts/pull-image.sh`: step 4, pull the image by digest on this node, tag it, drop the page cache.
- `scripts/inspect-image.sh`: step 4, read the image with no GPU and no network (versions, libnccl).
- `scripts/smoke-gpu.sh`, `scripts/smoke_gpu.py`: step 4, the one-minute GPU test in a throwaway
  container (no network, 8 GB cap).
- `scripts/run-rank.sh`: the recipe's `docker run` for one rank on one node (`DRY_RUN=1` prints it).
- `scripts/launch.sh`: step 6 from the head: preflight all four nodes, start the cache flushers and
  the ranks (3, 2, 1 headless, then 0), stop everything if a node falls below 3 GiB free or a
  rank exits, finish when `/health` answers. `DRY_RUN=1` runs the preflight and prints each command.
- `scripts/cache-flusher.sh`: drops a node's page cache whenever it holds more than 40 GiB, during
  the load.
- `scripts/stop.sh`: remove the four containers and stop the flushers; weights, image and caches stay.

## Log

- **2026-10-23** Revision pinned to `76faee1b` (main at 02:26 UTC; the 02:23 commit differs only
  in README.md).
- **2026-10-23** Step 1: 66 files, 49 shards, 252.4 GB in 55 min (~76 MB/s), exit 0. The
  memory-capped scope peaked at 3.0 GiB. tmux ends its session when the script exits, so "no
  sessions" afterwards means it finished: read `~/models/MiniMax-M3.1-Flash.download.exit`.
- **2026-10-23** Recipe review, before any boot: the first boot runs the target alone (DSpark at
  step 8) on Marlin experts (faster FP4 kernels as a later, separate step); a named cache volume
  keeps compiled kernels between boots; vLLM measures image memory at startup; the NCCL settings
  moved to `nccl.env`, copied whole from a deployment proven on this kind of ring; the image's
  libnccl path is found in step 4 instead of guessed. `--gpu-memory-utilization 0.80` stays: with
  the KV size pinned it no longer sizes the cache, but vLLM still requires that much free memory
  at startup.
- **2026-10-23** Step 2: all 66 files match the Hub's hashes (50 sha256, 16 blob sha1), nothing
  extra on disk; 2 min at ~2.1 GB/s with four workers, under a 4 GiB memory cap.
- **2026-10-23** Step 3, 28 min end to end: rank 3 → head 12 min at 345 MB/s; head check 2 min;
  head → ranks 1 and 2 at once, 12 min at 360 MB/s each (720 MB/s out of the head, both over the
  head-rank 1 cable); ranks 1 and 2 checked in parallel, 2 min. 66/66 OK on all three. Each
  stream is limited by rsync over ssh (one encrypted stream ~350 MB/s), not by the cables.
  Another model kept serving on the ring throughout, each node at 13-15 GiB free.
- **2026-10-23** Step 4. The image is vLLM `0.1.dev23341+g8744dd2fb` with torch 2.14 built for CUDA
  13.0 (kernels sm_80..sm_120), Triton 3.8, FlashInfer 0.7.2 and NCCL 2.30.7 at one path, which
  torch and vLLM both use; no b12x package. Every flag the recipe uses is accepted, and the
  `minimax_m3` reasoning and tool parsers are there. vLLM cannot print its own `--help` without a
  GPU ("Failed to infer device type"). GPU smoke test on a GB10 (sm_121, driver CUDA 13.0): PASS
  (bf16 matmul, a Triton kernel, a vLLM op compiled for sm_120, FlashInfer). Pulls took 345-487 s
  a node over Wi-Fi; `docker image inspect` reports 11.5 GB on a node using the containerd image
  store (compressed) and 27.6 GB on the others (unpacked), the same digest everywhere.
