"""Push the private ACE-Step worker kernel to Kaggle.

The job payload is embedded in the private Kaggle notebook source, matching the
deployment pattern used in cpu-and-gpu-ai-runs. Never print JOB or GH_TOKEN.
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

# This slug matches the actual Kaggle URL previously returned by the API:
# https://www.kaggle.com/code/<user>/kaggle-music-studio-worker
KERNEL_SLUG = "kaggle-music-studio-worker"
KERNEL_TITLE = "Kaggle Music Studio Worker"


def inject_job(source: str, job: dict[str, object]) -> str:
    """Insert runtime configuration without moving a future import below code."""
    assignment = "JOB = " + repr(job)
    future_import = "from __future__ import annotations"
    if future_import in source:
        return source.replace(
            future_import,
            future_import + "\n\n" + assignment,
            1,
        )
    return assignment + "\n\n" + source


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = (proc.stdout or "") + (proc.stderr or "")
    if output.strip():
        print(output.strip(), flush=True)
    return proc.returncode, output


def set_outputs(ref: str, deployed: bool = False) -> None:
    """Publish the canonical ref even on push failure so diagnostics can run."""
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(f"ref={ref}\n")
        handle.write(f"deployed={'true' if deployed else 'false'}\n")


def kernel_state(ref: str) -> tuple[str, str]:
    """Return ACTIVE, FINISHED, or UNKNOWN from Kaggle's current kernel status."""
    code, output = run(["kaggle", "kernels", "status", ref])
    upper = output.upper()

    if any(marker in upper for marker in (
        "CANCEL_ACKNOWLEDGED", "CANCELLED", "CANCELED", "ERROR", "COMPLETE"
    )):
        return "FINISHED", output
    if "QUEUED" in upper or "RUNNING" in upper:
        return "ACTIVE", output
    if code != 0:
        return "UNKNOWN", output
    return "UNKNOWN", output


def wait_until_not_active(ref: str, wait_seconds: int = 12 * 60) -> str:
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        state, _ = kernel_state(ref)
        if state != "ACTIVE":
            return state
        print("The existing Kaggle kernel is still queued/running; checking again in 30 seconds.", flush=True)
        time.sleep(30)
    return "ACTIVE"


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
        "idle_minutes": max(5, min(120, int(os.getenv("IN_IDLE_MINUTES", "30")))),
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
    worker_source = inject_job(source, job)
    # Compile before upload so Kaggle never receives an invalid generated script.
    compile(worker_source, "music_worker.py", "exec")
    (push_dir / "music_worker.py").write_text(worker_source, encoding="utf-8")

    metadata = {
        "id": f"{user}/{KERNEL_SLUG}",
        "title": KERNEL_TITLE,
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
    ref = f"{user}/{KERNEL_SLUG}"
    set_outputs(ref, deployed=False)
    print(f"Pushing private Kaggle kernel {ref} ({accelerator})", flush=True)
    command = ["kaggle", "kernels", "push", "-p", str(push_dir), "--accelerator", accelerator]

    last_output = ""
    for attempt in range(1, 4):
        code, output = run(command)
        last_output = output
        if code == 0:
            break
        if "409" not in output or attempt == 3:
            raise SystemExit(
                "Kaggle kernel push failed. Check the Kaggle status above, account limits and credentials."
            )

        print(
            f"Kaggle returned HTTP 409 (attempt {attempt}/3). "
            "Checking whether the canonical kernel is still busy before retrying.",
            flush=True,
        )
        state = wait_until_not_active(ref)
        if state == "ACTIVE":
            raise SystemExit(
                f"Kaggle kernel {ref} remained queued/running for 12 minutes; "
                "not creating duplicate kernels. Re-run after that job finishes."
            )
        # Give Kaggle's save endpoint a moment to release its update lock.
        time.sleep(20)

    match = re.search(r"kaggle\.com/code/([^/\s]+)/([^/\s?#]+)", last_output)
    if match:
        ref = f"{match.group(1)}/{match.group(2)}"
    set_outputs(ref, deployed=True)
    print(f"Kernel deployed: {ref}", flush=True)


if __name__ == "__main__":
    main()
