"""Push the private ACE-Step worker kernel to Kaggle.

The job payload is embedded in the private Kaggle notebook source, matching the
deployment pattern used by cpu-and-gpu-ai-runs. Never print JOB or GH_TOKEN.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
KERNEL_SLUG = "music-studio-worker"


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = (proc.stdout or "") + (proc.stderr or "")
    if output.strip():
        print(output.strip(), flush=True)
    return proc.returncode, output


def main() -> None:
    user = os.environ["KAGGLE_USERNAME"]
    token = os.environ["KAGGLE_API_TOKEN"]
    gh_token = os.environ["GH_TOKEN"]
    repo = os.environ["GH_REPO"]
    if not token or not gh_token or "/" not in repo:
        raise SystemExit("Missing Kaggle credentials or GitHub repository.")

    job = {
        "gh_token": gh_token,
        "gh_repo": repo,
        "owner": repo.split("/", 1)[0],
        "idle_minutes": max(5, min(120, int(os.getenv("IN_IDLE_MINUTES", "25")))),
        "model": os.getenv("IN_MODEL", "acestep-v15-turbo"),
        "lm_model": os.getenv("IN_LM_MODEL", "acestep-5Hz-lm-0.6B"),
        "lm_backend": os.getenv("IN_LM_BACKEND", "pt"),
        "max_hours": 5.2,
        "run_id": os.getenv("GITHUB_RUN_ID", str(int(time.time()))),
    }

    push_dir = pathlib.Path("/tmp/kaggle-music-worker")
    shutil.rmtree(push_dir, ignore_errors=True)
    push_dir.mkdir(parents=True)

    source = (REPO_ROOT / "kaggle" / "music_worker.py").read_text(encoding="utf-8")
    (push_dir / "music_worker.py").write_text(
        "JOB = " + repr(job) + "\n\n" + source, encoding="utf-8"
    )

    metadata = {
        "id": f"{user}/{KERNEL_SLUG}",
        "title": "Kaggle Music Studio Worker",
        "code_file": "music_worker.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (push_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    accelerator = os.getenv("IN_ACCELERATOR", "NvidiaTeslaT4")
    print(f"Pushing private Kaggle kernel {user}/{KERNEL_SLUG} ({accelerator})", flush=True)
    command = ["kaggle", "kernels", "push", "-p", str(push_dir), "--accelerator", accelerator]
    code, output = run(command)

    if code != 0 and "409" in output:
        print("Kernel is busy; waiting 45 seconds before retrying.", flush=True)
        time.sleep(45)
        code, output = run(command)

    if code != 0 and "409" in output:
        alternate = f"{KERNEL_SLUG}-{job['run_id']}"
        metadata["id"] = f"{user}/{alternate}"
        (push_dir / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print("Existing worker still busy; using a unique kernel for this run.", flush=True)
        code, output = run(command)

    if code != 0:
        raise SystemExit("Kaggle kernel push failed. Check Kaggle credentials, quota and kernel status.")

    match = re.search(r"kaggle\.com/code/([^/\s]+)/([^/\s?#]+)", output)
    ref = f"{match.group(1)}/{match.group(2)}" if match else metadata["id"]
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
        handle.write(f"ref={ref}\n")
    print(f"Kernel deployed: {ref}", flush=True)


if __name__ == "__main__":
    main()
