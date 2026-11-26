"""Run and reproduce the native rank-stock checks."""
import os
import sys

from serial_worker import ROOT
from verify_hypothesis_archive import build, verify


def main(audit):
    directory = ROOT/"build/capture-inventory"/os.environ["JUNQI_SERIAL_KEY"]
    directory.mkdir(parents=True,exist_ok=True)
    exe = directory/"hypothesis_archive.exe"
    if not audit:
        build(directory)
    if verify(exe,[]):
        raise ValueError("unexpected output from native fixture checks")
    print("PASS exact live and collision-loss stock against 144 enumerated rank cases")
    if audit:
        print("JUNQI_RESEARCH supported")
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise ValueError("expected audit or no argument")
    main(bool(sys.argv[1:]))
