# Kaggle Music Studio

Hebrew-first music generation studio powered by **ACE-Step 1.5** on Kaggle GPU.

This repository follows the GitHub-Issues control plane used in `unisocsu/cpu-and-gpu-ai-runs`, but runs one persistent Kaggle music worker instead of a text LLM worker.

## What it does

- Browser chat creates one GitHub Issue per song project.
- A GitHub Actions workflow pushes a private Kaggle GPU kernel.
- Kaggle runs ACE-Step 1.5 (Turbo DiT + configurable 5Hz LM).
- Gemini 3.8 Flash plans the arrangement, refines/generates lyrics when none are supplied, and writes a separate English album-art prompt.
- Stable Diffusion Turbo generates a 512×512 cover on GPU 1 while ACE-Step generates audio on GPU 0. The image step is skipped if a second GPU or sufficient free VRAM is unavailable.
- FFmpeg embeds the cover as the MP3 front-cover image without re-encoding audio; the ZIP also contains `album-cover.jpg`.
- The worker updates the Issue with live stage/status messages.
- A generated version is packaged as a ZIP and uploaded to a GitHub Release.
- The page reads the release, downloads the ZIP, extracts the MP3 into a browser Blob, and plays it.
- Send feedback in the same conversation to generate a revised version; every version is kept as a separate Release asset.

The page exposes **music-plan summaries, prompt revisions, lyrics, model settings and progress events**. It does not expose private model chain-of-thought.

## First-time setup

### 1. GitHub repository settings

This repository is public by default. Issues, prompts, lyrics, progress comments and Release assets are therefore public. Do not submit private material. Keep the Kaggle notebook private.

Create a fine-grained GitHub Personal Access Token for this repository only:

- Issues: Read and write
- Actions: Read and write
- Metadata: Read

Enter the token in the Studio settings. The page keeps it in the current browser session only.

### 2. Kaggle secrets

In GitHub → Settings → Secrets and variables → Actions, add:

- `KAGGLE_USERNAME`: your Kaggle username
- `KAGGLE_API_TOKEN`: your Kaggle API token
- `GEMINI_API_KEY`: Google AI Studio API key used by the Gemini music planner (optional; without it, the worker falls back to ACE-Step prompt formatting and a prompt-derived cover).

The Gemini key is passed from GitHub Actions to the private Kaggle kernel at launch; it is not written to the public repository. Never paste the key into an Issue or prompt.

The GitHub workflow uses its temporary `GITHUB_TOKEN` with Issues and Contents write permissions to update Issues and publish Release assets.

### 3. Deploy the page

In GitHub → Settings → Pages, set **Source** to **GitHub Actions**. The `Deploy Music Studio Pages` workflow publishes `docs/` automatically after page changes. Open the Pages URL, set the repository and GitHub token in Settings, then click **Start music worker**.

### 4. Generate music

Create a new project and enter a music description and optional lyrics. The default target is a 3-minute song, Hebrew vocal language, ACE-Step 1.5 Turbo, and the 0.6B LM for a stable first run on T4.

The first Kaggle start installs ACE-Step, downloads the model checkpoints, and eagerly loads both the Turbo DiT and 0.6B language model before the worker reports ready. The readiness check verifies the loaded model names through ACE-Step's `/health` response; HTTP 200 alone is not treated as proof that the weights are loaded. Later requests in the same worker session reuse the already-loaded models, so feedback versions start immediately.

The worker has two limits:

- **Hard limit: 90 minutes** from startup, model loading included (`max_runtime_minutes`, 5–180). A new song is not started when fewer than 6 minutes remain; the request stays queued for the next worker.
- **Idle limit: 30 minutes** without activity (`idle_minutes`), counted from model readiness or the last processed request.

When the worker stops it posts a short notice in the project, and sending feedback from the page starts it again automatically. The page shows whether a worker run is active, and it never starts a second one while one is running.

### How the page and worker talk through Issues

- The worker polls open `music-request` Issues with an ETag (unchanged polls cost no rate limit) and only reads the comments of issues that changed.
- Requests and feedback are only accepted from the repository owner (or the dispatch workflow), because the repository is public.
- The models are loaded only when there is real work. Each request gets an immediate acknowledgement, then live progress, then the result with ZIP/MP3 links.
- If model start-up fails, the request stays in the queue and the next worker run retries it.

### Improve a previous version

Open the same project and send specific feedback such as:

- Make the chorus more energetic.
- Bring the piano forward and reduce the synths.
- The vocal phrasing is rushed; leave more space between lines.
- Keep the melody and structure, but make the drums less busy.

The worker passes the original prompt, current lyrics and your feedback through ACE-Step's prompt/lyrics formatting LM, records the revised prompt, and generates the next version. This text-feedback loop does not claim to perform objective audio-quality analysis.

## Generate directly from GitHub Actions

Open **Actions → Generate music (Kaggle dispatch) → Run workflow**. The dispatch form accepts a project title, music prompt, optional lyrics, duration (2–5 minutes), vocal language, style direction and extra instructions. It creates a tagged music request and starts the private Kaggle worker automatically.

The dispatch workflow deliberately pins the generation profile to **Turbo DiT + 0.6B LM + PyTorch on one Tesla T4**. This is the conservative profile for the current T4 setup. Before downloading or loading model weights, the worker checks free VRAM on `cuda:0`, estimates the float32 DiT/LM weight footprint plus a runtime reserve, and stops before loading if the free memory is insufficient. Two T4 cards are treated as separate devices; their VRAM is never added together. The estimate is a preflight guard, not a guarantee against every runtime allocation spike.

## Direct Kaggle GPU generator (separate pipeline)

A second, isolated generator is available at the GitHub Pages path `/direct.html`.

- Open `https://unisocsu.github.io/kaggle_music_bgenerator_admin/direct.html` (after Pages deploys).
- Enter a GitHub fine-grained token for this repository with Issues read/write and Metadata read.
- Submit only `caption`, `lyrics`, and `duration`. The page creates an Issue containing the exact values.
- `.github/workflows/direct-kaggle-song.yml` listens for those marked Issues, creates a **private Kaggle notebook with GPU enabled**, pushes it with the Kaggle CLI, waits for completion, downloads the MP3, publishes a separate Release, and comments the result back on the same Issue.
- The page polls that Issue's comments and displays the progress/result updates.

This path is intentionally separate from the existing Music Studio worker and does not call Gemini or the existing prompt planner. It uses ACE-Step 1.5 Turbo with its 0.6B language model. The existing `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN` Actions secrets are reused; no new secrets are required. The GitHub token is entered in the page and is not stored in the repository.

The direct path currently creates a fresh Kaggle kernel for each Issue, so model setup/checkpoint downloads may repeat. Kaggle GPU availability and execution quotas remain subject to Kaggle.

## Runtime notes

- Kaggle GPU availability and session limits are controlled by Kaggle and can change.
- Two T4 devices are not treated as one 32GB GPU. The initial configuration is conservative and uses the supported CUDA path; the second GPU may remain unused depending on the runtime.
- The ZIP is uploaded after generation completes. Live progress is sent to the Issue while the model is working.
- The page tries to fetch and extract the MP3 from the ZIP to a Blob. It also has a direct MP3/ZIP fallback for browsers or networks that block cross-origin ZIP reads.
- A public GitHub Release is not private storage. Audio and metadata are visible to anyone who can access this public repository.

## References

- [ACE-Step 1.5](https://github.com/ACE-Step/ACE-Step-1.5)
- [ACE-Step REST API](https://github.com/ACE-Step/ACE-Step-1.5/blob/main/docs/en/API.md)
- [ACE-Step inference API](https://github.com/ACE-Step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md)
