"""Fetch exactly the MarkLLM revision used by this project; run after pip install."""

from pathlib import Path
import subprocess
import sys

from codeguard import MARKLLM_REVISION

URL = "https://github.com/THU-BPM/MarkLLM.git"


def main():
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / ".deps" / "MarkLLM"
    destination = destination.resolve()
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", URL, str(destination)], check=True)
    origin = subprocess.check_output(["git", "-C", str(destination), "remote", "get-url", "origin"], text=True).strip()
    if origin != URL:
        raise SystemExit(f"Refusing to modify a checkout with a different origin: {origin}")
    head = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if head != MARKLLM_REVISION or not (destination / "watermark/kgw/kgw.py").exists():
        subprocess.run(["git", "-C", str(destination), "diff", "--exit-code"], check=True)
        subprocess.run(["git", "-C", str(destination), "diff", "--cached", "--exit-code"], check=True)
        subprocess.run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", MARKLLM_REVISION], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", MARKLLM_REVISION], check=True)
    print(f"MarkLLM {MARKLLM_REVISION} at {destination}")


if __name__ == "__main__":
    main()
