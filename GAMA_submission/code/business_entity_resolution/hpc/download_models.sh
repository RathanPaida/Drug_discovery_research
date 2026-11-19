#!/bin/bash
# Stage the two pre-trained models (both MIT / Apache-2.0, 568M params each)
# into $BER_MODEL_DIR. ~2.3 GB each; needs internet. run_amazon_ml.sh calls it
# by itself when the models are missing (fine where jobs are online, like the
# our development node); where compute nodes are offline, run it on the login node:
#   tmux new -s ber_models ; bash hpc/download_models.sh
# Resumable: re-running only fetches what is missing.
#
#   BAAI/bge-m3               -> $BER_MODEL_DIR/bge-m3               (MIT)        bi-encoder
#   BAAI/bge-reranker-v2-m3   -> $BER_MODEL_DIR/bge-reranker-v2-m3   (Apache-2.0) cross-encoder
#
# Faster alternative if the login node is slow: download on a laptop with
#   pip install huggingface_hub && hf download BAAI/bge-m3 --local-dir bge-m3 \
#       --include "*.json" "*.model" "model.safetensors"
# (same for bge-reranker-v2-m3), then upload both folders into $BER_MODEL_DIR
# with scp/rsync/WinSCP.

set -euo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_here/_config.sh"
if ! ber_online https://huggingface.co; then
    echo "ERROR: $(hostname) cannot reach huggingface.co, and the download needs internet." >&2
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        echo "  This cluster's job nodes are offline: run it on the login node:  bash $CODE_DIR/hpc/download_models.sh" >&2
    fi
    echo "  Or download both models elsewhere and upload them into $BER_MODEL_DIR (see the top of this script)." >&2
    exit 1
fi
activate_env
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
# Both repos are public: download anonymously. A stored/expired token (HF_TOKEN
# or ~/.cache/huggingface/token) would otherwise be sent and rejected with a
# misleading "401 ... Repository Not Found".
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
# writing to a log file (a job), not a terminal: skip the progress bars, whose
# thousands of \r-updates would bury the log; each model still gets a line
[ -t 1 ] || export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$BER_MODEL_DIR"

python - <<'PY'
import os
from huggingface_hub import HfApi, snapshot_download

dest = os.environ["BER_MODEL_DIR"]
ANON = False                    # token=False: never send a token
for repo, name in (("BAAI/bge-m3", "bge-m3"), ("BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3")):
    local = os.path.join(dest, name)
    if all(os.path.exists(os.path.join(local, f)) for f in ("config.json", "tokenizer.json", "model.safetensors")):
        print(f">> {repo}: already staged at {local}", flush=True)
        continue
    files = HfApi().list_repo_files(repo, token=ANON)
    weights = ["model.safetensors"] if "model.safetensors" in files else ["pytorch_model.bin"]
    allow = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "sentencepiece.bpe.model"] + weights
    print(f">> {repo}: {', '.join(f for f in files if f in allow)}", flush=True)
    path = snapshot_download(repo_id=repo, local_dir=os.path.join(dest, name), allow_patterns=allow, token=ANON)
    print(f"   -> {path}", flush=True)
    # newer transformers prefer safetensors: convert a legacy .bin once, here on the login node
    binp, stp = os.path.join(path, "pytorch_model.bin"), os.path.join(path, "model.safetensors")
    if os.path.exists(binp) and not os.path.exists(stp):
        import torch
        from safetensors.torch import save_file
        sd = torch.load(binp, map_location="cpu", weights_only=True)
        save_file({k: v.contiguous() for k, v in sd.items()}, stp, metadata={"format": "pt"})
        os.remove(binp)
        print(f"   converted pytorch_model.bin -> model.safetensors", flush=True)
PY

echo
echo "Staged models:"
du -sh "$BER_MODEL_DIR"/* || true
