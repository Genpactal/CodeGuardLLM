"""Container entry point. Never run this worker directly on the host."""

import json
import os
import subprocess
import sys
import time

# Parent retains control of reporting; generated stdout cannot forge worker JSON.
# Catch SystemExit as well as regular exceptions, so sys.exit(0) is not a pass.
CHILD = r'''
import json, os, sys
payload = json.load(sys.stdin)
report_fd = int(sys.argv[1])
namespace = {"__name__": "__main__"}
try:
    exec(compile(payload["code"], "<solution>", "exec"), namespace)
    exec(compile(payload["test"], "<tests>", "exec"), namespace)
    namespace["check"](namespace[payload["entry_point"]])
except BaseException:
    sys.exit(1)
os.write(report_fd, b"P")
'''


def main():
    payload = json.load(sys.stdin)
    started = time.monotonic()
    read_fd, write_fd = os.pipe()
    try:
        child = subprocess.run(
            [sys.executable, "-I", "-c", CHILD, str(write_fd)], input=json.dumps(payload).encode(),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=payload["timeout_seconds"], check=False, pass_fds=(write_fd,),
        )
        os.set_blocking(read_fd, False)
        try:
            completed_tests = os.read(read_fd, 1) == b"P"
        except BlockingIOError:
            completed_tests = False
        status = "passed" if child.returncode == 0 and completed_tests else "failed"
    except subprocess.TimeoutExpired:
        status = "timeout"
    finally:
        os.close(read_fd)
        os.close(write_fd)
    print(json.dumps({"status": status, "seconds": time.monotonic() - started}))


if __name__ == "__main__":
    main()
