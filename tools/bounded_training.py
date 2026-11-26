"""Serial, resumable training with bounded runtime and compact checkpoints."""
import argparse
import ctypes
from ctypes import wintypes
import json
import msvcrt
import os
from pathlib import Path
import shutil
import subprocess
import time

from evidence_model import ENCODING, load, save, sha
from evaluate_transfer import compare, totals

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/REPORTS/evidence-model-contract.json"
MAX_BYTES = 32 * 1024 * 1024
GENERATIONS = 8
STARTS = (90000, 100000)
EVAL_START = 110000
EVAL_COUNT = 32


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def size(directory):
    return sum(p.stat().st_size for p in directory.iterdir() if p.is_file())


def write(path, value):
    data = (json.dumps(value, separators=(",", ":"), allow_nan=False) + chr(10)).encode()
    if size(path.parent) + len(data) > MAX_BYTES:
        raise RuntimeError("artifact budget exceeded")
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


class IO(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in
                ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]


class Memory(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in
        ("peak_working_set", "working_set", "paged_peak", "paged", "nonpaged_peak",
         "nonpaged", "pagefile", "pagefile_peak")]


def counters(process):
    io, memory = IO(), Memory()
    memory.cb = ctypes.sizeof(memory)
    kernel, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    kernel.GetProcessIoCounters.argtypes = [wintypes.HANDLE, ctypes.POINTER(IO)]
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
    if not kernel.GetProcessIoCounters(int(process._handle), ctypes.byref(io)):
        raise ctypes.WinError()
    if not psapi.GetProcessMemoryInfo(int(process._handle), ctypes.byref(memory), memory.cb):
        raise ctypes.WinError()
    return dict(read_bytes=io.read_bytes, write_bytes=io.write_bytes,
                peak_working_set=memory.peak_working_set)


def command(directory, deadline, metrics, *arguments, input_text=None):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("run deadline reached")
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    process = subprocess.Popen([str(directory / "junqi.exe"), *map(str, arguments)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               stdin=subprocess.PIPE if input_text is not None else None,
                               env=environment, creationflags=0x40 | 0x08000000)
    try:
        output, errors = process.communicate(input_text, timeout=min(180, remaining))
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise TimeoutError("engine deadline reached")
    usage = counters(process)
    metrics["read_bytes"] += usage["read_bytes"]
    metrics["write_bytes"] += usage["write_bytes"]
    metrics["peak_working_set"] = max(metrics["peak_working_set"], usage["peak_working_set"])
    if process.returncode:
        raise RuntimeError(errors[-2000:])
    return [json.loads(line) for line in output.splitlines()]


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest["generations"] != GENERATIONS or manifest["starts"] != list(STARTS):
            raise ValueError("run configuration changed")
        return manifest
    source = read(SOURCE)
    shutil.copyfile(source["executable_path"], directory / "junqi.exe")
    shutil.copyfile(source["rules_path"], directory / "ruleset.json")
    for index, stream in enumerate(source["streams"]):
        model = load(stream["child_path"], directory / "junqi.exe", directory / "ruleset.json")
        initial_path = directory / f"initial-{index}.json"
        if initial_path.exists():
            existing = load(initial_path, directory / "junqi.exe", directory / "ruleset.json")
            if existing["weights"] != model["weights"]:
                raise ValueError("initial checkpoint changed")
        else:
            save(initial_path, model["weights"], sha(stream["child_path"]),
                 directory / "junqi.exe", directory / "ruleset.json")
    manifest = dict(source_sha256=sha(SOURCE), engine=sha(directory / "junqi.exe"),
                    rules=sha(directory / "ruleset.json"), generations=GENERATIONS,
                    starts=list(STARTS), eval_start=EVAL_START, eval_count=EVAL_COUNT)
    write(manifest_path, manifest)
    return manifest


def evaluate(directory, deadline, metrics, model, policy):
    weights = model["weights"] if policy == "evidence" else model["weights"][:5]
    return command(directory, deadline, metrics, "transfer-probe", directory / "ruleset.json",
                   *weights, EVAL_START, EVAL_COUNT, policy, 0)


def run_round(directory, index, generation, deadline):
    report_path = directory / f"round-{index}-{generation}.json"
    if report_path.exists():
        return
    executable, rules = directory / "junqi.exe", directory / "ruleset.json"
    initial_path = directory / f"initial-{index}.json"
    parent_path = initial_path if generation == 1 else directory / f"model-{index}-{generation-1}.json"
    parent = load(parent_path, executable, rules)
    metrics = dict(read_bytes=0, write_bytes=0, peak_working_set=0)
    start = time.monotonic()
    seed = STARTS[index] + (generation - 1) * 512
    training, = command(directory, deadline, metrics, "resume-evidence", rules, *parent["weights"], seed)
    path = directory / f"model-{index}-{generation}.json"
    if not path.exists():
        checkpoint = dict(schema=1, encoding=ENCODING, rules_sha256=sha(rules),
                          engine_sha256=sha(executable), parent_sha256=sha(parent_path),
                          weights=training["weights"])
        write(path, checkpoint)
    child = load(path, executable, rules)
    if child["weights"] != training["weights"] or child["parent_sha256"] != sha(parent_path):
        raise ValueError("interrupted checkpoint differs from deterministic retraining")
    rows = evaluate(directory, deadline, metrics, child, "evidence")
    masked = evaluate(directory, deadline, metrics, child, "short")
    previous_path = directory / f"round-{index}-{generation-1}.json"
    if generation == 1:
        parent_rows = evaluate(directory, deadline, metrics, parent, "evidence")
        initial_rows = parent_rows
    else:
        parent_rows = read(previous_path)["rows"]
        initial_rows = read(directory / f"round-{index}-1.json")["initial_rows"]
    report = dict(stream=index, generation=generation, training_seed=seed, training=training,
                  model_sha256=sha(path), parent_sha256=sha(parent_path), rows=rows, masked=masked,
                  totals=totals(rows), versus_parent=compare(parent_rows, rows),
                  versus_initial=compare(initial_rows, rows), versus_masked=compare(masked, rows),
                  engine_io_including_pipes=metrics, seconds=time.monotonic()-start)
    if generation == 1:
        report["initial_rows"] = initial_rows
    write(report_path, report)
    print(json.dumps(dict(stream=index, generation=generation, totals=report["totals"],
                          versus_initial=report["versus_initial"],
                          artifact_bytes=size(directory))), flush=True)


def audit(directory):
    manifest = read(directory / "manifest.json")
    if sha(directory / "junqi.exe") != manifest["engine"] or sha(directory / "ruleset.json") != manifest["rules"]:
        raise ValueError("frozen engine or rules changed")
    complete = 0
    for index in range(2):
        initial_rows = None
        parent_path = directory / f"initial-{index}.json"
        for generation in range(1, GENERATIONS + 1):
            path = directory / f"round-{index}-{generation}.json"
            if not path.exists():
                break
            report = read(path)
            model_path = directory / f"model-{index}-{generation}.json"
            model = load(model_path, directory / "junqi.exe", directory / "ruleset.json")
            expected = [(seed, seat) for seed in range(EVAL_START, EVAL_START + EVAL_COUNT)
                        for seat in range(4)]
            if [(row["seed"], row["seat"]) for row in report["rows"]] != expected:
                raise ValueError("evaluation layout blocks changed")
            if model["weights"] != report["training"]["weights"]:
                raise ValueError("saved weights differ from training")
            if model["parent_sha256"] != sha(parent_path) or sha(model_path) != report["model_sha256"]:
                raise ValueError("model lineage mismatch")
            if generation == 1:
                parent_rows = initial_rows = report["initial_rows"]
            if (compare(parent_rows, report["rows"]) != report["versus_parent"] or
                    compare(initial_rows, report["rows"]) != report["versus_initial"] or
                    compare(report["masked"], report["rows"]) != report["versus_masked"] or
                    totals(report["rows"]) != report["totals"]):
                raise ValueError("raw rows do not support reported comparison")
            parent_rows, parent_path = report["rows"], model_path
            complete += 1
    print(f"PASS bounded training audit; completed_rounds={complete}; artifact_bytes={size(directory)}")
    return complete


def run(directory, hours):
    manifest = prepare(directory)
    audit(directory)
    deadline = time.monotonic() + hours * 3600
    started = time.monotonic()
    status, error = "complete", None
    write(directory / "summary.json",
          dict(status="running", started_utc=time.time(), hours_limit=hours, error=None))
    try:
        for generation in range(1, GENERATIONS + 1):
            for index in range(2):
                if (directory / "STOP").exists() or time.monotonic() >= deadline:
                    status = "stopped"
                    break
                if shutil.disk_usage(directory).free < 1024**3 or size(directory) > MAX_BYTES - 1024**2:
                    status = "storage_limit"
                    break
                run_round(directory, index, generation, deadline)
            if status != "complete":
                break
    except Exception as failure:
        status, error = "failed", str(failure)
    summary = dict(status=status, error=error, completed_rounds=audit(directory),
                   seconds_this_invocation=time.monotonic()-started, hours_limit=hours,
                   artifact_bytes_before_summary=size(directory), max_artifact_bytes=MAX_BYTES,
                   engine_sha256=manifest["engine"], rules_sha256=manifest["rules"])
    write(directory / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if error:
        raise RuntimeError(error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--hours", type=int, default=6)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--profile", choices=("historical-evidence", "archive-pressure", "archive-pressure-practice"),
                        default="historical-evidence")
    parser.add_argument("--reference-run", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    directory.relative_to(ROOT / "build")
    if args.hours < 1:
        raise ValueError("runtime must be a positive integer number of hours")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a+b") as lock:
        if lock.seek(0, 2) == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        if args.profile == "archive-pressure":
            from archive_pressure_training import run
            run(directory, min(args.hours, 6), args.audit, command, write)
        elif args.profile == "archive-pressure-practice":
            from archive_pressure_practice import run, REFERENCE
            reference = (args.reference_run or REFERENCE).resolve()
            reference.relative_to(ROOT / "build")
            if reference == directory:
                raise ValueError("practice requires an output directory separate from the reference run")
            run(directory, reference, min(args.hours, 6), args.audit, command, write)
        elif args.audit:
            audit(directory)
        elif (directory / "summary.json").exists() and read(directory / "summary.json")["status"] == "complete":
            audit(directory)
            print("PASS already complete; no training or artifact rewrite")
        else:
            run(directory, args.hours)


if __name__ == "__main__":
    main()
