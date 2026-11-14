# Kaggle Music Studio

Hebrew-first music generation studio powered by **ACE-Step 1.5** on Kaggle GPU.

This repository follows the GitHub-Issues control plane used in `unisocsu/cpu-and-gpu-ai-runs`, but runs one persistent Kaggle music worker instead of a text LLM worker.

## What it does

- Browser chat creates one GitHub Issue per song project.
- A GitHub Actions workflow pushes a private Kaggle GPU kernel.
- Kaggle runs ACE-Step 1.5 (Turbo DiT + configurable 5Hz LM).
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

The GitHub workflow uses its temporary `GITHUB_TOKEN` with Issues and Contents write permissions to update Issues and publish Release assets.

### 3. Deploy the page

Enable GitHub Pages for branch `main`, folder `/docs`. Open the generated Pages URL, set the repository and GitHub token in Settings, then click **Start music worker**.

### 4. Generate music

Create a new project and enter a music description and optional lyrics. The default target is a 3-minute song, Hebrew vocal language, ACE-Step 1.5 Turbo, and the 0.6B LM for a stable first run on T4.

The first Kaggle start must install ACE-Step and download its model weights. Later requests in the same worker session reuse the already-loaded models. The worker automatically shuts down after the configured idle period.

### Improve a previous version

Open the same project and send specific feedback such as:

- Make the chorus more energetic.
- Bring the piano forward and reduce the synths.
- The vocal phrasing is rushed; leave more space between lines.
- Keep the melody and structure, but make the drums less busy.

The worker passes the original prompt, current lyrics and your feedback through ACE-Step's prompt/lyrics formatting LM, records the revised prompt, and generates the next version. This text-feedback loop does not claim to perform objective audio-quality analysis.

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
