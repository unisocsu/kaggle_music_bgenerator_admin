# -*- coding: utf-8 -*-
"""Private Kaggle worker for ACE-Step 1.5."""
from __future__ import annotations

import csv
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import zipfile
from typing import Any

import requests

JOB = globals().get("JOB", {})
OUT = pathlib.Path("/kaggle/working/music-studio")
OUT.mkdir(parents=True, exist_ok=True)

GH_API = "https://api.github.com"
GH_REPO = JOB.get("gh_repo", "")
GH_OWNER = JOB.get("owner", "")
GH_TOKEN = JOB.get("gh_token", "")
GH_HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GH_TOKEN}",
    "X-GitHub-Api-Version": "2022-11-28",
}
ACE_URL = "http://127.0.0.1:8001"
BOT_MARKER = "<!--music-studio-bot-->"
PROGRESS_MARKER = "<!--music-progress-->"
STARTED_MARKER = "<!--music-started-->"
ASSISTANT_MARKER = "<!--music-assistant-->"
FEEDBACK_MARKER = "<!--music-feedback-->"
REQUEST_MARKER = "<!--music-request-->"

HTTP = requests.Session()
HTTP.headers.update(GH_HEADERS)
PROGRESS_IDS: dict[int, int] = {}
LAST_SONG_CREATED = time.time()
SERVER_PROCESS: subprocess.Popen | None = None


def log(message: str) -> None:
    print(f"[music-studio] {message}", flush=True)


def gh(method: str, path: str, **kwargs: Any) -> requests.Response:
    url = path if path.startswith("https://") else GH_API + path
    response = HTTP.request(method, url, timeout=(15, 90), **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(f"GitHub API {method} {path}: HTTP {response.status_code}: {response.text[:800]}")
    return response


def gh_json(method: str, path: str, **kwargs: Any) -> Any:
    response = gh(method, path, **kwargs)
    return response.json() if response.content else {}


def issue_comments(issue_number: int) -> list[dict[str, Any]]:
    return gh_json("GET", f"/repos/{GH_REPO}/issues/{issue_number}/comments?per_page=100")


def post_comment(issue_number: int, body: str) -> dict[str, Any]:
    return gh_json("POST", f"/repos/{GH_REPO}/issues/{issue_number}/comments", json={"body": body})


def update_comment(comment_id: int, body: str) -> dict[str, Any]:
    return gh_json("PATCH", f"/repos/{GH_REPO}/issues/comments/{comment_id}", json={"body": body})


def set_progress(issue_number: int, title: str, details: str = "") -> None:
    body = f"{PROGRESS_MARKER}\n**{title}**"
    if details:
        body += f"\n\n{details}"
    body += f"\n\n{BOT_MARKER}"
    comment_id = PROGRESS_IDS.get(issue_number)
    if comment_id is None:
        for comment in issue_comments(issue_number):
            if PROGRESS_MARKER in (comment.get("body") or "") and comment.get("user", {}).get("login") == "github-actions[bot]":
                comment_id = int(comment["id"])
                PROGRESS_IDS[issue_number] = comment_id
                break
    if comment_id:
        update_comment(comment_id, body)
    else:
        created = post_comment(issue_number, body)
        PROGRESS_IDS[issue_number] = int(created["id"])


def parse_request(issue: dict[str, Any]) -> dict[str, Any] | None:
    body = issue.get("body") or ""
    index = body.find(REQUEST_MARKER)
    if index < 0:
        return None
    raw = body[index + len(REQUEST_MARKER):].strip().split("<!--/music-request-->")[0].strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except (ValueError, TypeError):
        return None


def latest_assistant_meta(comments: list[dict[str, Any]]) -> dict[str, Any]:
    for comment in reversed(comments):
        body = comment.get("body") or ""
        match = re.search(r"<!--music-meta:(.*?)-->", body, re.S)
        if ASSISTANT_MARKER in body and match:
            try:
                result = json.loads(match.group(1))
                if isinstance(result, dict):
                    return result
            except ValueError:
                pass
    return {}


def done_for_feedback(comments: list[dict[str, Any]], comment_id: int) -> bool:
    marker = f"<!--music-feedback-result:{comment_id}-->"
    return any(marker in (item.get("body") or "") for item in comments)


def list_music_issues() -> list[dict[str, Any]]:
    issues = gh_json(
        "GET",
        f"/repos/{GH_REPO}/issues?state=open&labels=music-request&per_page=100&sort=created&direction=asc",
    )
    return [
        item for item in issues
        if "pull_request" not in item
        and (
            item.get("user", {}).get("login") == GH_OWNER
            or (
                item.get("user", {}).get("login") == "github-actions[bot]"
                and REQUEST_MARKER in (item.get("body") or "")
            )
        )
    ]


def gpu_memory_preflight() -> None:
    """Estimate peak VRAM before downloading/loading checkpoints; never pool GPUs."""
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=15,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not inspect GPU memory before model loading: {exc}") from exc

    rows = list(csv.reader(io.StringIO(raw)))
    if not rows:
        raise RuntimeError("No CUDA GPU was reported by nvidia-smi.")

    # The worker uses cuda:0. Two T4 cards are separate 15GB devices, not one 30GB pool.
    first = [cell.strip() for cell in rows[0]]
    if len(first) < 4:
        raise RuntimeError(f"Unexpected nvidia-smi memory output: {raw[:500]}")
    gpu_index, gpu_name = first[0], first[1]
    total_gb = float(first[2]) / 1024
    free_gb = float(first[3]) / 1024

    lm_name = str(JOB.get("lm_model", "acestep-5Hz-lm-0.6B"))
    # ACE-Step upstream estimates: 2B DiT ~4.46GB and LM 0.6B ~1.2GB
    # in bf16. The worker forces float32 for T4 stability, so weight estimates
    # are doubled; 2GB is reserved for CUDA/runtime/activations.
    lm_bf16_gb = {"acestep-5Hz-lm-0.6B": 1.2, "acestep-5Hz-lm-1.7B": 3.4}
    if lm_name not in lm_bf16_gb:
        raise RuntimeError(f"Unsupported LM for the conservative T4 profile: {lm_name}")
    estimated_dit_gb = 4.46 * 2
    estimated_lm_gb = lm_bf16_gb[lm_name] * 2
    runtime_reserve_gb = 2.0
    required_gb = estimated_dit_gb + estimated_lm_gb + runtime_reserve_gb

    log(
        f"GPU preflight: GPU {gpu_index} {gpu_name}; "
        f"total={total_gb:.1f}GB free={free_gb:.1f}GB; "
        f"estimated weights={estimated_dit_gb + estimated_lm_gb:.1f}GB "
        f"+ runtime reserve={runtime_reserve_gb:.1f}GB; "
        f"required free VRAM={required_gb:.1f}GB."
    )
    if free_gb < required_gb:
        raise RuntimeError(
            f"GPU memory preflight refused model loading: need about {required_gb:.1f}GB "
            f"free on cuda:{gpu_index}, but only {free_gb:.1f}GB is free. "
            "Choose the 0.6B LM / Tesla T4 profile or free GPU memory first. "
            "Multiple GPUs are not combined."
        )


def patch_ace_step_cuda_dtype(repo_dir: pathlib.Path) -> None:
    """Make ACE-Step honor ACESTEP_DTYPE on CUDA, including pre-Ampere Tesla T4."""
    target = repo_dir / "acestep" / "core" / "generation" / "handler" / "init_service_orchestrator.py"
    if not target.is_file():
        raise RuntimeError(f"ACE-Step dtype source file not found: {target}")

    source = target.read_text(encoding="utf-8")
    patch_marker = "# MUSIC_STUDIO_CUDA_DTYPE_OVERRIDE"
    if patch_marker in source:
        log("ACE-Step CUDA dtype override is already installed.")
        return

    old_block = '''            elif resolved_device == "cuda":
                if gpu_config.cuda_supports_bfloat16():
                    self.dtype = torch.bfloat16
                else:
                    self.dtype = torch.float16
                    logger.info(
                        "[initialize_service] Pre-Ampere CUDA detected: "
                        "using float16 instead of bfloat16."
                    )
'''
    new_block = '''            elif resolved_device == "cuda":
                # MUSIC_STUDIO_CUDA_DTYPE_OVERRIDE
                requested_dtype = os.environ.get("ACESTEP_DTYPE", "").strip().lower()
                cuda_dtype_map = {
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                }
                if requested_dtype:
                    selected_dtype = cuda_dtype_map.get(requested_dtype)
                    if selected_dtype is None:
                        raise ValueError(
                            f"Unsupported ACESTEP_DTYPE={requested_dtype!r}; "
                            "choose float32, float16 or bfloat16."
                        )
                    if selected_dtype == torch.bfloat16 and not gpu_config.cuda_supports_bfloat16():
                        raise ValueError(
                            "ACESTEP_DTYPE=bfloat16 was requested, but this CUDA GPU "
                            "does not support native bfloat16."
                        )
                    self.dtype = selected_dtype
                    logger.info(
                        f"[initialize_service] CUDA dtype explicitly selected by "
                        f"ACESTEP_DTYPE={requested_dtype}: using {self.dtype}."
                    )
                elif gpu_config.cuda_supports_bfloat16():
                    self.dtype = torch.bfloat16
                else:
                    self.dtype = torch.float16
                    logger.info(
                        "[initialize_service] Pre-Ampere CUDA detected: "
                        "using float16 instead of bfloat16."
                    )
'''
    if old_block not in source:
        raise RuntimeError(
            "ACE-Step dtype selection block did not match the reviewed upstream source; "
            "refusing to apply a potentially unsafe patch. Check the upstream version."
        )
    patched = source.replace(old_block, new_block, 1)
    compile(patched, str(target), "exec")
    target.write_text(patched, encoding="utf-8")
    log("Patched ACE-Step CUDA dtype selection; ACESTEP_DTYPE=float32 will now be honored.")
    log("ACE-Step dtype patch syntax check passed.")


def install_and_start_ace() -> None:
    global SERVER_PROCESS
    if SERVER_PROCESS and SERVER_PROCESS.poll() is None:
        return

    gpu_memory_preflight()
    repo_dir = pathlib.Path("/kaggle/temp/ACE-Step-1.5")
    if not repo_dir.exists():
        log("Cloning ACE-Step 1.5")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ACE-Step/ACE-Step-1.5.git", str(repo_dir)],
            check=True,
        )

    # Patch the upstream dtype selector before uv sync installs the local ACE-Step source.
    patch_ace_step_cuda_dtype(repo_dir)

    log("Installing ACE-Step dependencies; first start downloads the model weights")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "uv"], check=True)
    subprocess.run(["uv", "sync", "--no-dev"], cwd=repo_dir, check=True)

    venv_python = repo_dir / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise RuntimeError("ACE-Step virtual environment was not created by uv sync.")

    env = dict(os.environ)
    env.update({
        "ACESTEP_CONFIG_PATH": JOB.get("model", "acestep-v15-turbo"),
        "ACESTEP_LM_MODEL_PATH": JOB.get("lm_model", "acestep-5Hz-lm-0.6B"),
        "ACESTEP_LM_BACKEND": JOB.get("lm_backend", "pt"),
        "ACESTEP_INIT_LLM": "true",
        # Force DiT, VAE and LM checkpoints to download/load before the server is
        # considered ready. Without this, /health can return 200 while models
        # are still lazy-loaded on the first request.
        "ACESTEP_NO_INIT": "false",
        "ACESTEP_DEVICE": "cuda",
        # Tesla T4 is pre-Ampere; ACE-Step reports fp16 latent overflow/NaNs
        # on this GPU. Use fp32 for numerically stable diffusion latents.
        "ACESTEP_DTYPE": "float32",
        "ACESTEP_API_HOST": "127.0.0.1",
        "ACESTEP_API_PORT": "8001",
        "ACESTEP_USE_FLASH_ATTENTION": "false",
        "HF_HOME": "/kaggle/temp/huggingface",
        "TOKENIZERS_PARALLELISM": "false",
    })
    pathlib.Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    log("Visible GPU inventory: " + subprocess.getoutput("nvidia-smi -L"))
    log(f"ACE-Step precision configuration: ACESTEP_DTYPE={env.get('ACESTEP_DTYPE', 'default')} (Tesla T4 stability mode)")

    server_log = open(OUT / "acestep-server.log", "a", encoding="utf-8")
    SERVER_PROCESS = subprocess.Popen(
        [str(venv_python), "acestep/api_server.py"],
        cwd=repo_dir,
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )

    # First-time checkpoint downloads and eager GPU initialization can take
    # several minutes. Do not treat HTTP 200 alone as readiness: ACE-Step's
    # health route is also available while lazy loading is enabled.
    deadline = time.time() + 30 * 60
    last_status_log = 0.0
    while time.time() < deadline:
        if SERVER_PROCESS.poll() is not None:
            tail = subprocess.getoutput(f"tail -n 100 {OUT / 'acestep-server.log'}")
            raise RuntimeError(
                "ACE-Step API exited during model initialization. Last server output:\n"
                + tail[-7000:]
            )
        try:
            response = requests.get(ACE_URL + "/health", timeout=4)
            if response.ok:
                payload = response.json()
                data = payload.get("data", payload) if isinstance(payload, dict) else {}
                if not isinstance(data, dict):
                    data = {}
                dit_ready = bool(data.get("models_initialized")) and bool(data.get("loaded_model"))
                lm_ready = bool(data.get("llm_initialized")) and bool(data.get("loaded_lm_model"))
                if dit_ready and lm_ready:
                    log(
                        "ACE-Step models are loaded: "
                        f"DiT={data.get('loaded_model')} · LM={data.get('loaded_lm_model')}"
                    )
                    return
                now = time.time()
                if now - last_status_log >= 30:
                    log(
                        "API responds, waiting for model initialization: "
                        f"DiT loaded={dit_ready} · LM loaded={lm_ready}"
                    )
                    last_status_log = now
        except (requests.RequestException, ValueError):
            pass
        time.sleep(6)
    tail = subprocess.getoutput(f"tail -n 120 {OUT / 'acestep-server.log'}")
    raise RuntimeError(
        "ACE-Step models did not finish loading within 30 minutes. "
        "Last server output:\n" + tail[-7000:]
    )


def ace_post(path: str, payload: dict[str, Any], timeout: int = 3600) -> Any:
    response = requests.post(ACE_URL + path, json=payload, timeout=(15, timeout))
    response.raise_for_status()
    result = response.json()
    if result.get("error"):
        raise RuntimeError(str(result["error"]))
    if int(result.get("code", 200)) >= 400:
        raise RuntimeError(str(result.get("error") or result))
    return result.get("data", result)


def format_prompt(prompt: str, lyrics: str, duration: int, language: str) -> tuple[str, str, dict[str, Any]]:
    try:
        data = ace_post("/format_input", {
            "prompt": prompt[:4000],
            "lyrics": lyrics[:12000],
            "param_obj": json.dumps({"duration": duration, "language": language}, ensure_ascii=False),
        }, timeout=600)
        if not isinstance(data, dict):
            data = {}
        caption = str(data.get("caption") or data.get("prompt") or prompt).strip() or prompt
        formatted_lyrics = str(data.get("lyrics") or lyrics).strip()
        return caption, formatted_lyrics, data
    except Exception as exc:
        log(f"format_input fallback: {type(exc).__name__}: {exc}")
        return prompt, lyrics, {"format_warning": str(exc)[:400]}


def parse_result(value: Any) -> dict[str, Any]:
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, str):
        try:
            value = json.loads(value)
            if isinstance(value, list) and value:
                value = value[0]
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def task_failure_details(task_result: dict[str, Any], task_id: str) -> str:
    """Recover the ACE-Step exception when /query_result omits its error."""
    for key in ("error", "exception", "traceback", "error_message", "message"):
        value = task_result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:5000]

    nested = parse_result(task_result.get("result"))
    for key in ("error", "exception", "traceback", "error_message", "message"):
        value = nested.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:5000]

    # ACE-Step's local-cache response omits JobRecord.error on failed jobs.
    # The server log retains the task-specific exception and traceback.
    log_path = OUT / "acestep-server.log"
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        marker = f"Job {task_id} FAILED:"
        for index in range(len(lines) - 1, -1, -1):
            if marker in lines[index]:
                return "\n".join(lines[index:index + 70])[-5000:]
    except OSError as exc:
        return f"Could not read ACE-Step server log: {type(exc).__name__}: {exc}"

    safe_keys = sorted(
        key for key in task_result.keys()
        if key not in {"result", "prompt", "lyrics", "file", "wave"}
    )
    return (
        "ACE-Step returned failed status without an error field. "
        f"task_id={task_id}; response_keys={safe_keys}"
    )


def generate_audio(issue_number: int, prompt: str, lyrics: str, duration: int,
                   language: str, version: int, feedback: str = "") -> tuple[pathlib.Path, str, str, dict[str, Any]]:
    set_progress(issue_number, "👀 ראיתי את הפרומפט",
                 "מודל השפה קורא עכשיו את הבקשה ומכין את תיאור המוזיקה.")
    source_prompt = prompt
    if feedback:
        source_prompt = (
            f"{prompt}\n\nUser revision request: {feedback}\n"
            "Apply the requested musical changes while preserving the parts the user did not ask to change. "
            "Be specific about arrangement, instrumentation, energy, vocal phrasing and song structure."
        )
    caption, new_lyrics, format_meta = format_prompt(source_prompt, lyrics, duration, language)

    set_progress(issue_number, "מכין משימת יצירת אודיו",
                 f"גרסה {version} · אורך {duration} שניות · שפה: {language}")
    payload = {
        "prompt": caption,
        "lyrics": new_lyrics,
        "thinking": True,
        "use_format": False,
        "vocal_language": language,
        "audio_duration": duration,
        "audio_format": "mp3",
        "model": JOB.get("model", "acestep-v15-turbo"),
        "lm_model_path": JOB.get("lm_model", "acestep-5Hz-lm-0.6B"),
        "lm_backend": JOB.get("lm_backend", "pt"),
        "inference_steps": 8,
        "batch_size": 1,
        "use_cot_caption": True,
        "use_cot_language": True,
    }
    if not new_lyrics.strip():
        payload["sample_mode"] = True
        payload["sample_query"] = (
            caption + f". Generate a complete song with vocals and lyrics in {language}. "
            "Use a clear verse/chorus structure."
        )
    task = ace_post("/release_task", payload, timeout=600)
    task_id = task.get("task_id") or task.get("id") if isinstance(task, dict) else None
    if not task_id:
        raise RuntimeError(f"ACE-Step returned no task_id: {str(task)[:800]}")

    set_progress(issue_number, "יוצר את השיר",
                 "המשימה התקבלה ב־ACE-Step. הסטטוס נבדק מול תור היצירה של המודל.")
    task_result: dict[str, Any] = {}
    last_update = 0.0
    deadline = time.time() + max(45 * 60, duration * 90)
    while time.time() < deadline:
        query = ace_post("/query_result", {"task_id_list": [task_id]}, timeout=60)
        entries = query if isinstance(query, list) else query.get("data", []) if isinstance(query, dict) else []
        if isinstance(entries, dict):
            entries = [entries]
        if entries:
            task_result = entries[0] if isinstance(entries[0], dict) else {}
            nested = parse_result(task_result.get("result"))
            if nested:
                task_result.update(nested)
            status = int(task_result.get("status", 0))
            if status == 2:
                details = task_failure_details(task_result, str(task_id))
                log(f"ACE-Step task {task_id} failed; diagnostic details:\n{details[-3000:]}")
                raise RuntimeError(f"ACE-Step generation failed: {details[:1000]}")
            if status == 1:
                break
        if time.time() - last_update > 35:
            set_progress(issue_number, "יוצר את השיר",
                         "המודל עדיין עובד. עדכון זה מבוסס על סטטוס המשימה, לא על אחוז משוער.")
            last_update = time.time()
        time.sleep(8)
    else:
        raise TimeoutError("ACE-Step generation exceeded the per-song timeout.")

    audio_path = str(task_result.get("file") or task_result.get("audio_path") or "")
    if not audio_path:
        raise RuntimeError("ACE-Step completed but returned no audio file path.")
    audio_url = audio_path if audio_path.startswith("http") else ACE_URL + (audio_path if audio_path.startswith("/") else "/" + audio_path)

    set_progress(issue_number, "מוריד את קובץ האודיו", "השיר נוצר; מוריד את ה־MP3 לאריזה.")
    response = requests.get(audio_url, timeout=(30, 600))
    response.raise_for_status()
    audio_file = OUT / f"song-v{version}.mp3"
    audio_file.write_bytes(response.content)
    if not audio_file.stat().st_size:
        raise RuntimeError("Downloaded audio is empty.")

    actual_lyrics = str(task_result.get("lyrics") or new_lyrics or "").strip()
    meta = {
        "task_id": task_id,
        "version": version,
        "duration_seconds": duration,
        "vocal_language": language,
        "model": JOB.get("model", "acestep-v15-turbo"),
        "lm_model": JOB.get("lm_model", "acestep-5Hz-lm-0.6B"),
        "caption": caption,
        "lyrics": actual_lyrics,
        "format_metadata": format_meta,
        "seed_value": task_result.get("seed_value"),
        "generation_info": task_result.get("generation_info"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return audio_file, caption, actual_lyrics, meta


def get_release(issue_number: int) -> dict[str, Any]:
    tag = f"music-issue-{issue_number}"
    response = requests.get(f"{GH_API}/repos/{GH_REPO}/releases/tags/{tag}", headers=GH_HEADERS, timeout=30)
    if response.status_code == 200:
        return response.json()
    if response.status_code != 404:
        response.raise_for_status()
    return gh_json("POST", f"/repos/{GH_REPO}/releases", json={
        "tag_name": tag,
        "name": f"Music project #{issue_number}",
        "body": f"Generated music versions for https://github.com/{GH_REPO}/issues/{issue_number}.",
        "draft": False,
        "prerelease": False,
        "target_commitish": "main",
    })


def upload_asset(release: dict[str, Any], path: pathlib.Path, name: str, content_type: str) -> dict[str, Any]:
    existing = next((a for a in release.get("assets", []) if a.get("name") == name), None)
    if existing:
        gh("DELETE", f"/repos/{GH_REPO}/releases/assets/{existing['id']}")
    upload_url = release["upload_url"].split("{", 1)[0] + "?name=" + requests.utils.quote(name)
    headers = dict(GH_HEADERS)
    headers["Content-Type"] = content_type
    with path.open("rb") as source:
        response = requests.post(upload_url, headers=headers, data=source, timeout=(30, 1800))
    if response.status_code >= 400:
        raise RuntimeError(f"Release asset upload failed: HTTP {response.status_code}: {response.text[:800]}")
    return response.json()


def publish_version(issue_number: int, version: int, audio_file: pathlib.Path,
                    caption: str, lyrics: str, meta: dict[str, Any], feedback: str) -> dict[str, str]:
    set_progress(issue_number, "אורז את התוצאה ומעלה ל־Release",
                 "קובץ ה־ZIP נבנה עכשיו. נכסי ה־Release זמינים אחרי שההעלאה מסתיימת.")
    bundle = OUT / f"bundle-v{version}"
    shutil.rmtree(bundle, ignore_errors=True)
    bundle.mkdir(parents=True)
    shutil.copy2(audio_file, bundle / "song.mp3")
    (bundle / "prompt.txt").write_text(caption + "\n", encoding="utf-8")
    (bundle / "lyrics.txt").write_text(lyrics + "\n", encoding="utf-8")
    (bundle / "improvement-notes.txt").write_text(feedback + "\n", encoding="utf-8")
    (bundle / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (bundle / "README.txt").write_text(
        "Kaggle Music Studio output\n"
        f"Version: {version}\n"
        "Audio: song.mp3\nPrompt: prompt.txt\nLyrics: lyrics.txt\nMetadata: metadata.json\n",
        encoding="utf-8",
    )

    zip_path = OUT / f"music-v{version}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file_path in sorted(bundle.iterdir()):
            archive.write(file_path, arcname=file_path.name)

    release = get_release(issue_number)
    zip_asset = upload_asset(release, zip_path, f"music-v{version}.zip", "application/zip")
    mp3_asset = upload_asset(release, audio_file, f"song-v{version}.mp3", "audio/mpeg")
    release_body = (
        "Generated with ACE-Step 1.5 on Kaggle.\n\n"
        f"Project: https://github.com/{GH_REPO}/issues/{issue_number}\n\n"
        "Each version ZIP contains the MP3, prompt, lyrics, improvement notes and metadata.\n\n"
        f"Latest published version: v{version}."
    )
    gh("PATCH", f"/repos/{GH_REPO}/releases/{release['id']}", json={"body": release_body})
    return {
        "release_url": release.get("html_url", ""),
        "zip_url": zip_asset.get("browser_download_url", ""),
        "mp3_url": mp3_asset.get("browser_download_url", ""),
    }


def publish_result(issue_number: int, version: int, caption: str, lyrics: str,
                   meta: dict[str, Any], links: dict[str, str], feedback_id: int | None) -> None:
    summary = (
        f"## גרסה {version} מוכנה 🎵\n\n"
        "**תקציר תכנון:** תיאור הסגנון והמילים עובדו לפי הבקשה. "
        "בגרסאות שיפור הוחל המשוב על הפרומפט והמילים הקודמים.\n\n"
        f"**הפרומפט המעודכן:**\n\n{caption}\n\n"
        f"**מילים:**\n\n{lyrics[:5000] or 'לא הוזנו מילים.'}\n\n"
        f"[הורדת ZIP]({links['zip_url']}) · [השמעת MP3 / הורדה]({links['mp3_url']}) · "
        f"[פתיחת Release]({links['release_url']})\n\n"
        "כתוב מה תרצה לשנות כדי ליצור גרסה נוספת."
    )
    public_meta = {
        "version": version, "prompt": caption, "lyrics": lyrics,
        "release_url": links["release_url"], "zip_url": links["zip_url"],
        "mp3_url": links["mp3_url"], "task_id": meta.get("task_id"),
    }
    meta_marker = "<!--music-meta:" + json.dumps(public_meta, ensure_ascii=False, separators=(",", ":")) + "-->"
    feedback_marker = f"<!--music-feedback-result:{feedback_id}-->" if feedback_id else STARTED_MARKER
    post_comment(issue_number, f"{ASSISTANT_MARKER}\n{summary}\n\n{meta_marker}\n{feedback_marker}\n{BOT_MARKER}")


def handle_generation(issue: dict[str, Any], request_data: dict[str, Any], feedback: str = "",
                      feedback_id: int | None = None, previous: dict[str, Any] | None = None) -> None:
    global LAST_SONG_CREATED
    issue_number = int(issue["number"])
    previous = previous or {}
    version = int(previous.get("version", 0)) + 1
    duration = max(30, min(300, int(request_data.get("duration", 180))))
    language = str(request_data.get("language", "he"))[:8]
    base_prompt = str(previous.get("prompt") or request_data.get("prompt") or "")[:4000]
    lyrics = str(previous.get("lyrics") or request_data.get("lyrics") or "")[:12000]
    style = str(request_data.get("style") or "")[:500]
    instructions = str(request_data.get("instructions") or "")[:2000]
    prompt_parts = [base_prompt] if base_prompt else []
    if style:
        prompt_parts.append("Requested musical style: " + style)
    if instructions:
        prompt_parts.append("Additional generation instructions: " + instructions)
    base_prompt = "\n\n".join(prompt_parts)[:4000]

    try:
        audio_file, caption, new_lyrics, meta = generate_audio(
            issue_number, base_prompt, lyrics, duration, language, version, feedback
        )
        links = publish_version(issue_number, version, audio_file, caption, new_lyrics, meta, feedback)
        LAST_SONG_CREATED = time.time()
        publish_result(issue_number, version, caption, new_lyrics, meta, links, feedback_id)
        set_progress(issue_number, f"גרסה {version} פורסמה בהצלחה",
                     f"[ZIP]({links['zip_url']}) · [MP3]({links['mp3_url']}) · [Release]({links['release_url']})")
    except Exception as exc:
        log(f"Generation failed for issue #{issue_number}: {type(exc).__name__}: {exc}")
        marker = f"<!--music-feedback-result:{feedback_id}-->" if feedback_id else STARTED_MARKER
        failure_meta = {
            "version": int((previous or {}).get("version", 0)),
            "prompt": str((previous or {}).get("prompt") or request_data.get("prompt") or ""),
            "lyrics": str((previous or {}).get("lyrics") or request_data.get("lyrics") or ""),
        }
        meta_marker = "<!--music-meta:" + json.dumps(failure_meta, ensure_ascii=False, separators=(",", ":")) + "-->"
        post_comment(issue_number,
                     f"{ASSISTANT_MARKER}\n## יצירת הגרסה נכשלה\n\n"
                     f"{type(exc).__name__}: {str(exc)[:1000]}\n\n"
                     "בדוק את Kaggle diagnostics ב־Actions לפני ניסיון נוסף. אפשר לשלוח משוב חדש אחרי תיקון התקלה.\n"
                     f"{meta_marker}\n{marker}\n{BOT_MARKER}")


def process_issue(issue: dict[str, Any]) -> bool:
    issue_number = int(issue["number"])
    request_data = parse_request(issue)
    if not request_data:
        return False
    comments = issue_comments(issue_number)

    if not any(STARTED_MARKER in (c.get("body") or "") for c in comments):
        set_progress(issue_number, "בקשת מוזיקה התקבלה", "מכין את סביבת היצירה.")
        handle_generation(issue, request_data)
        return True

    latest = latest_assistant_meta(comments)
    if not latest:
        return False
    for comment in comments:
        author = comment.get("user", {}).get("login")
        body = comment.get("body") or ""
        comment_id = int(comment.get("id", 0))
        if author != GH_OWNER or FEEDBACK_MARKER not in body or done_for_feedback(comments, comment_id):
            continue
        feedback = body.split(FEEDBACK_MARKER, 1)[1].strip()
        if not feedback:
            post_comment(issue_number, f"{ASSISTANT_MARKER}\nכתוב בבקשה מה תרצה לשפר.\n<!--music-feedback-result:{comment_id}-->")
            return True
        set_progress(issue_number, "התקבל משוב לגרסה חדשה",
                     "מודל השפה מתאים את תיאור המוזיקה והמילים למשוב שלך.")
        handle_generation(issue, request_data, feedback, comment_id, latest)
        return True
    return False


def main() -> None:
    global LAST_SONG_CREATED
    if not GH_TOKEN or not GH_REPO or not GH_OWNER:
        raise SystemExit("Missing GitHub worker credentials.")
    idle_minutes = max(5, min(120, int(JOB.get("idle_minutes", 30))))
    max_runtime = float(JOB.get("max_hours", 5.2)) * 3600
    start = time.time()
    LAST_SONG_CREATED = start
    log(f"Worker started for {GH_REPO}; it stays warm until {idle_minutes} minutes pass without a successfully published song.")
    try:
        while time.time() - start < max_runtime:
            issues = list_music_issues()
            if issues and (SERVER_PROCESS is None or SERVER_PROCESS.poll() is not None):
                for issue in issues:
                    set_progress(int(issue["number"]), "מכין סביבת יצירה", "מתקין ACE-Step וטוען את המודלים ל־GPU.")
                install_and_start_ace()

            did_work = False
            for issue in issues:
                if process_issue(issue):
                    did_work = True
                    break
            if did_work:
                continue
            idle_seconds = time.time() - LAST_SONG_CREATED
            if idle_seconds >= idle_minutes * 60:
                log(f"No successfully published song for {idle_minutes} minutes; stopping the worker.")
                break
            time.sleep(12)
    finally:
        if SERVER_PROCESS and SERVER_PROCESS.poll() is None:
            SERVER_PROCESS.terminate()
            try:
                SERVER_PROCESS.wait(timeout=20)
            except subprocess.TimeoutExpired:
                SERVER_PROCESS.kill()
        log("Worker stopped.")


if __name__ == "__main__":
    main()
