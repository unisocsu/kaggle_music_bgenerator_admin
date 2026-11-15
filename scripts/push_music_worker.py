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
import urllib.error
import urllib.request

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


def set_outputs(ref: str, deployed: bool = False, issue_url: str = "") -> None:
    """Publish identifiers even on push failure so diagnostics and summaries can run."""
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(f"ref={ref}\n")
        handle.write(f"deployed={'true' if deployed else 'false'}\n")
        handle.write(f"issue_url={issue_url}\n")


def github_api(method: str, repo: str, path: str, token: str,
               payload: dict[str, object] | None = None) -> dict[str, object]:
    url = f"https://api.github.com/repos/{repo}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "Kaggle-Music-Studio",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"GitHub API {method} {path}: HTTP {exc.code}: {detail}") from exc
    return json.loads(raw.decode("utf-8")) if raw else {}


def create_generation_issue(repo: str, token: str) -> str:
    prompt = os.getenv("IN_PROMPT", "").strip()
    if not prompt:
        raise SystemExit("The dispatch prompt is required.")
    if len(prompt) > 4000:
        raise SystemExit("Prompt is too long; maximum is 4,000 characters.")
    lyrics = os.getenv("IN_LYRICS", "").strip()
    style = os.getenv("IN_STYLE", "").strip()
    instructions = os.getenv("IN_INSTRUCTIONS", "").strip()
    if len(lyrics) > 12000:
        raise SystemExit("Lyrics are too long; maximum is 12,000 characters.")
    if len(style) > 500:
        raise SystemExit("Style is too long; maximum is 500 characters.")
    if len(instructions) > 2000:
        raise SystemExit("Instructions are too long; maximum is 2,000 characters.")

    try:
        duration = int(os.getenv("IN_DURATION", "180"))
    except ValueError as exc:
        raise SystemExit("Duration must be one of 120, 180, 240 or 300 seconds.") from exc
    if duration not in {120, 180, 240, 300}:
        raise SystemExit("Duration must be one of 120, 180, 240 or 300 seconds.")

    language = os.getenv("IN_LANGUAGE", "he")
    if language not in {"he", "en", "unknown", "es", "fr", "ar", "yi"}:
        raise SystemExit("Unsupported vocal language.")

    # Ensure the queue label exists; it may already have been created by the web UI.
    try:
        github_api("GET", repo, "/labels/music-request", token)
    except RuntimeError as exc:
        if "HTTP 404" not in str(exc):
            raise
        try:
            github_api("POST", repo, "/labels", token, {
                "name": "music-request",
                "color": "8b9cff",
                "description": "Music Studio generation requests",
            })
        except RuntimeError as create_exc:
            if "HTTP 422" not in str(create_exc):
                raise

    title = os.getenv("IN_SONG_TITLE", "").strip()[:120] or prompt[:65] or "Music Studio request"
    request_data = {
        "version": 1,
        "prompt": prompt,
        "lyrics": lyrics,
        "duration": duration,
        "language": language,
        "style": style,
        "instructions": instructions,
    }
    issue = github_api("POST", repo, "/issues", token, {
        "title": title,
        "body": (
            "Music Studio dispatch request.\\n\\n"
            "<!--music-request-->\\n"
            + json.dumps(request_data, ensure_ascii=False, separators=(",", ":"))
            + "\\n<!--/music-request-->"
        ),
        "labels": ["music-request"],
    })
    issue_url = str(issue.get("html_url") or "")
    if not issue_url:
        raise RuntimeError("GitHub created the request but did not return its issue URL.")
    print(f"Queued music request: {issue_url}", flush=True)
    return issue_url


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

    issue_url = ""
    if os.getenv("IN_CREATE_REQUEST", "").lower() == "true":
        issue_url = create_generation_issue(repo, gh_token)

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
    set_outputs(ref, deployed=False, issue_url=issue_url)
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
    set_outputs(ref, deployed=True, issue_url=issue_url)
    print(f"Kernel deployed: {ref}", flush=True)


if __name__ == "__main__":
    main()
