"""Step 2: check every downloaded file against the hash Hugging Face publishes for it.

    python3 -I verify.py <manifest.json> <model dir> [workers]

The manifest (manifest/<revision>.json in this repo) was built from the Hub's API for the pinned
revision. Weight shards and other LFS files carry a sha256; small text files only a blob id,
the sha1 of "blob <size>\\0" + content. A file passes when its size and hash both match.
Also reports files in the model dir that the manifest does not list (hf's .cache/ excluded).
Exit code 0 only when every file passes and nothing is missing.
"""
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

CHUNK = 16 * 1024 * 1024


def check(args):
    root, entry = args
    path = os.path.join(root, entry["path"])
    try:
        size = os.path.getsize(path)
    except OSError:
        return entry["path"], "MISSING", ""
    if size != entry["size"]:
        return entry["path"], "BAD SIZE", f"{size} != {entry['size']}"
    if "sha256" in entry:
        h, want = hashlib.sha256(), entry["sha256"]
    else:
        h, want = hashlib.sha1(), entry["blob_sha1"]
        h.update(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
        # Drop this file from the page cache: on a GB10 that memory is the GPU's too (Linux only).
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    got = h.hexdigest()
    return entry["path"], ("OK" if got == want else "BAD HASH"), "" if got == want else f"{got} != {want}"


def main():
    manifest_path, root = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    manifest = json.load(open(manifest_path))
    entries = manifest["files"]
    total = sum(e["size"] for e in entries)
    print(f"{manifest['repo']}@{manifest['revision'][:8]}: {len(entries)} files, {total / 1e9:.1f} GB, {workers} workers")
    start, done_bytes, failures = time.time(), 0, []
    sizes = {e["path"]: e["size"] for e in entries}
    # Largest first, so the long shards start early and the pool finishes together.
    jobs = [(root, e) for e in sorted(entries, key=lambda e: -e["size"])]
    with Pool(workers) as pool:
        for i, (path, status, detail) in enumerate(pool.imap_unordered(check, jobs), 1):
            done_bytes += sizes[path]
            rate = done_bytes / max(time.time() - start, 1e-6) / 1e9
            if status != "OK":
                failures.append((path, status, detail))
            print(f"[{i:2}/{len(jobs)}] {status:8} {path} {detail}  ({done_bytes / 1e9:.0f} GB, {rate:.2f} GB/s)", flush=True)
    listed = set(sizes)
    extra = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".cache"]
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            if rel not in listed:
                extra.append(rel)
    print(f"done in {time.time() - start:.0f} s")
    for path in sorted(extra):
        print(f"EXTRA (not in manifest): {path}")
    if failures:
        for path, status, detail in failures:
            print(f"FAIL {status}: {path} {detail}")
        sys.exit(1)
    print(f"ALL {len(entries)} FILES OK")


if __name__ == "__main__":
    main()
