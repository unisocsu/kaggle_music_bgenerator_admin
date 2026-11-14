"""Wait for a Kaggle kernel run to finish, without exposing credentials."""
from __future__ import annotations

import os
import subprocess
import sys
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
    print(output.strip(), flush=True)
    if proc.returncode != 0:
        return "UNKNOWN"
    upper = output.upper()
    if "COMPLETE" in upper:
        return "COMPLETE"
    if "ERROR" in upper or "CANCEL" in upper:
        return "FAILED"
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
            raise SystemExit("Kaggle worker failed or was cancelled.")
        time.sleep(60)
    raise SystemExit("Timed out waiting for Kaggle worker.")


if __name__ == "__main__":
    main()
