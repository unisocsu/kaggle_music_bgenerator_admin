"""Wait for a Kaggle kernel run to finish, without exposing credentials."""
from __future__ import annotations

import os
import subprocess
import time

REF = os.environ["KERNEL_REF"]


def status() -> str:
    proc = subprocess.run(
        ["kaggle", "kernels", "status", REF],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    if output.strip():
        print(output.strip(), flush=True)
    if proc.returncode != 0:
        return "UNKNOWN"

    upper = output.upper()
    # Check terminal failure states before the generic RUNNING/QUEUED states.
    if any(marker in upper for marker in (
        "CANCEL_ACKNOWLEDGED", "CANCELLED", "CANCELED", "ERROR"
    )):
        return "FAILED"
    if "COMPLETE" in upper:
        return "COMPLETE"
    if "RUNNING" in upper or "QUEUED" in upper:
        return "RUNNING"
    return "UNKNOWN"


def main() -> None:
    deadline = time.time() + 5.5 * 60 * 60
    while time.time() < deadline:
        current = status()
        if current == "COMPLETE":
            print("Kaggle worker completed.", flush=True)
            return
        if current == "FAILED":
            raise SystemExit("Kaggle worker failed or was cancelled; see the diagnostics artifact.")
        if current == "UNKNOWN":
            print("Kaggle returned an unknown worker state; retrying status in 60 seconds.", flush=True)
        time.sleep(60)
    raise SystemExit("Timed out waiting for Kaggle worker.")


if __name__ == "__main__":
    main()
