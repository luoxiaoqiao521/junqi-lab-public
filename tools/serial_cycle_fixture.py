"""Exercise the real worker with two isolated rounds and a resumable checkpoint."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

from serial_worker import ROOT, read, write_receipt

CONTROL = ROOT / "build/serial-cycle-fixture-control.json"


def issue_first(folder, control):
    if control["scenario"] == "reject":
        print("JUNQI_VERDICT rejected")
        return
    if control["scenario"] != "resume":
        raise ValueError("unknown bounded fixture scenario")
    with (folder / "issued.txt").open("x", encoding="ascii") as issued:
        issued.write("reviewer accepted exactly one independently scoped successor\n")
    print("JUNQI_VERDICT accepted")


def run_second(folder):
    first = hashlib.sha256((folder / "issued.txt").read_bytes()).hexdigest()
    checkpoint = folder / "checkpoint.json"
    if not checkpoint.exists():
        child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(120)"])
        try:
            write_receipt(checkpoint, dict(first=first, stage_pid=os.getpid(), child_pid=child.pid))
            print("CHECKPOINT waiting for planned interruption", flush=True)
            time.sleep(50)
            raise TimeoutError("fixture interruption did not arrive in time")
        finally:
            child.terminate()
            child.wait(timeout=5)
    if read(checkpoint)["first"] != first:
        raise ValueError("first-round artifact changed since the checkpoint")
    with (folder / "accepted.txt").open("x", encoding="ascii") as accepted:
        accepted.write(first + "\n")
    print("JUNQI_VERDICT accepted")


if __name__ == "__main__":
    key = os.environ["JUNQI_SERIAL_KEY"]
    folder = Path(ROOT / "build/serial-cycle-fixture" / key)
    folder.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == []:
        issue_first(folder, read(CONTROL))
    elif sys.argv[1:] == ["audit"]:
        run_second(folder)
    else:
        sys.exit("invalid bounded fixture command")
