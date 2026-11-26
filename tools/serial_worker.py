"""Run one declared Junqi stage and emit exactly one validated successor."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
EVENT = "junqi.serial.advance"
EXECUTION_PATHS = ("core/", "observe/", "strategy/", "sim/", "tools/", "tests/", "config/")
RESEARCH_BUDGET = "ops/research-budget.json"
RESEARCH_OUTCOMES = ("supported", "refuted", "inconclusive", "blocked")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_receipt(path, receipt):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    temporary.replace(path)


def load_plan():
    plan_file = ROOT / "ops/serial.json"
    plan = read(plan_file)
    ticket = read(ROOT / "ops/task.json")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if plan["ticket"] != ticket["id"] or plan["baseline"] != ticket["baseline"] or head != plan["baseline"]:
        raise ValueError("plan, active ticket, and checked-out baseline disagree")
    if not re.fullmatch(r"[a-z0-9-]+", plan["ticket"]):
        raise ValueError("invalid ticket ID")
    validate_steps(plan["steps"])
    validate_research(plan, head)
    for relative, expected in plan.get("inputs", {}).items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"missing declared local input: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"fingerprint mismatch: {relative}")
    digest = plan_digest(plan_file, ticket["allowed"])
    return plan, f'{plan["ticket"]}-{digest}'


def validate_research(plan, baseline):
    if "research" not in plan:
        return
    contract = plan["research"]
    for field in ("question", "competing_explanation", "single_change", "refuter"):
        if not isinstance(contract.get(field), str) or not contract[field].strip():
            raise ValueError(f"missing research question field: {field}")
    if (set(contract["routes"]) != set(RESEARCH_OUTCOMES) or
            not all(isinstance(x, str) and x.strip() for x in contract["routes"].values())):
        raise ValueError("research routing needs four explicit dispositions")
    study = contract["study"]
    if not re.fullmatch(r"[a-z0-9-]+", study) or RESEARCH_BUDGET not in plan.get("inputs", {}):
        raise ValueError("research budget must be bound to the current plan")
    ledger = read(ROOT / RESEARCH_BUDGET)
    group = ledger[study]
    allocations = group["allocated_seconds"]
    cap = group["hard_seconds"]
    if (type(cap) is not int or cap<1 or any(type(s) is not int or s<1 for s in allocations.values()) or
            sum(allocations.values())>cap or allocations.get(plan["ticket"])!=sum(s["seconds"] for s in plan["steps"])):
        raise ValueError("research stages do not fit their cumulative budget")
    if any(step.get("resumable") for step in plan["steps"]):
        raise ValueError("a research retry needs its own budget reservation")
    old = subprocess.run(["git", "show", f"{baseline}:{RESEARCH_BUDGET}"], cwd=ROOT, capture_output=True, timeout=10)
    if old.returncode==0:
        for key, previous in json.loads(old.stdout).items():
            current=ledger[key]
            if current["hard_seconds"]!=previous["hard_seconds"] or any(
                current["allocated_seconds"].get(ticket)!=amount for ticket,amount in previous["allocated_seconds"].items()
            ):
                raise ValueError("earlier research allocations must carry forward")


def bind_research_reservation(plan, key, serial):
    if "research" not in plan:
        return
    folder=serial/"research"/plan["research"]["study"]
    folder.mkdir(parents=True,exist_ok=True)
    slot=folder/(plan["ticket"]+".json")
    if slot.exists():
        if read(slot)["plan"]!=key:
            raise ValueError("research reservation already bound to a different plan")
    else:
        write_receipt(slot,{"plan":key,"seconds":sum(step["seconds"] for step in plan["steps"])})


def plan_digest(plan_file, paths, root=ROOT):
    digest = hashlib.sha256(plan_file.read_bytes())
    for relative in sorted(set(paths)):
        if not relative.startswith(EXECUTION_PATHS):
            continue
        path = root / relative
        digest.update(relative.encode() + b"\0")
        if path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:20]


def event_id(key, index):
    return f"{key}-{index}"


def requested_stage(context, key, count):
    events = [event for event in context["events"] if event["type"] == EVENT]
    if not events:
        return 0
    matches = [index for index in range(1, count)
               if any(event["id"] == event_id(key, index) for event in events)]
    return matches[0] if len(matches) == 1 else None


def resume_index(context, key, count):
    markers = [event.get("detail", "") for event in context["events"]
               if event["type"] == "run.manual" and str(event.get("detail")).startswith("resume:")]
    if not markers:
        return False, None
    targets = [i for i in range(count) if markers == [f"resume:{key}:{i}"]]
    return True, targets[0] if targets else None


@contextmanager
def worker_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "worker.lock").open("a+b") as lock:
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class CandidateRejected(ValueError):
    pass


def command(step):
    seconds = step["seconds"]
    if type(seconds) is not int or not 1 <= seconds <= 330:
        raise ValueError("invalid phase timeout")
    if step["kind"] == "check":
        return [os.environ.get("JUNQI_PWSH", "pwsh"), "-NoProfile", "-File", "tools/check.ps1"]
    script = step["script"]
    if step["kind"] != "python" or not re.fullmatch(r"tools/[a-z0-9_]+\.py", script):
        raise ValueError("invalid stage command")
    arguments = step.get("args", [])
    if arguments not in ([], ["audit"]):
        raise ValueError("invalid script arguments")
    return [sys.executable, "-X", "utf8", script, *arguments]


def validate_steps(steps):
    if not isinstance(steps, list) or not 1 <= len(steps) <= 5:
        raise ValueError("expected one to five stages")
    names = set()
    for step in steps:
        name = step.get("name") if isinstance(step, dict) else None
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("expected a distinct name for every stage")
        names.add(name)
        try:
            for field in ("resumable", "verdict"):
                if field in step and type(step[field]) is not bool:
                    raise ValueError(f"invalid {field}")
            if "round" in step and not re.fullmatch(r"[a-z0-9-]+", step["round"]):
                raise ValueError("invalid review round ID")
            command(step)
            path = "tools/check.ps1" if step["kind"] == "check" else step["script"]
            if not (ROOT / path).is_file():
                raise ValueError(f"missing stage script {path}")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"stage {name}: {error}") from error


def invoke(step, log):
    args = command(step)
    with log.open("xb") as output:
        subprocess.run(["git", "diff", "HEAD", "--check"], cwd=ROOT, stdout=output,
                       stderr=subprocess.STDOUT, check=True, timeout=10)
        sources = [ROOT / step["script"]] if step["kind"] == "python" else [
            ROOT / path for path in read(ROOT / "ops/task.json")["allowed"]
            if path.endswith(".py") and (ROOT / path).is_file()]
        for source in sources:
            compile(source.read_bytes(), str(source), "exec")
        result = subprocess.run(args, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                timeout=step["seconds"], check=False)
    if result.returncode:
        raise RuntimeError(f"exit {result.returncode}; see {log}")
    if step.get("verdict"):
        lines = log.read_text(encoding="utf8", errors="replace").splitlines()
        verdicts = [line.removeprefix("JUNQI_VERDICT ") for line in lines if line.startswith("JUNQI_VERDICT ")]
        if verdicts == ["rejected"]:
            raise CandidateRejected(f"candidate rejected; see {log}")
        if verdicts != ["accepted"]:
            raise ValueError(f"missing or ambiguous business verdict; see {log}")
    if step["kind"] == "check" and not any(
            line.startswith("CHECK PASS ") for line in log.read_text(encoding="utf8", errors="replace").splitlines()):
        raise RuntimeError(f"full check receipt missing; see {log}")


def execute_stage(plan, index, folder, run=invoke, *, resume=False, recheck=lambda: None):
    if index is None:
        return "ignored", "event does not match the active plan"
    for earlier in range(index):
        previous = folder / f"{earlier}.json"
        if not previous.is_file() or read(previous)["status"] != "passed":
            return "blocked", "predecessor has no successful receipt"
    path = folder / f"{index}.json"
    log = folder / f"{index}.log"
    step = plan["steps"][index]
    attempt = 1
    if path.exists():
        prior = read(path)
        if not (resume and step.get("resumable") and prior["status"] == "running"):
            return prior["status"], prior["message"]
        preserved = folder / f"{index}.interrupted.json"
        if preserved.exists():
            return "blocked", "one recovery already attempted; inspect the preserved receipts"
        attempt = 2
        with preserved.open("xb") as output:
            output.write(path.read_bytes())
        if log.is_file():
            log.rename(folder / f"{index}.interrupted.log")
    folder.mkdir(parents=True, exist_ok=True)
    receipt = dict(ticket=plan["ticket"], baseline=plan["baseline"], stage=step["name"], index=index,
                   status="running", attempt=attempt,
                   message="stage interrupted or still running; inspect before resuming")
    if "round" in step:
        receipt["round"] = step["round"]
    write_receipt(path, receipt)
    began=perf_counter()
    try:
        run(step, log)
        recheck()
        if "research" in plan and index+1==len(plan["steps"]):
            verdict=[line.removeprefix("JUNQI_RESEARCH ") for line in
                     log.read_text(encoding="utf8",errors="replace").splitlines() if line.startswith("JUNQI_RESEARCH ")]
            if len(verdict)!=1 or verdict[0] not in RESEARCH_OUTCOMES:
                raise ValueError("independent research disposition is missing or ambiguous")
            receipt.update(research_outcome=verdict[0], next_condition=plan["research"]["routes"][verdict[0]])
        receipt.update(status="passed", message=f'{step["name"]} passed')
    except CandidateRejected as error:
        receipt.update(status="rejected", message=f'{step["name"]}: {error}')
    except (OSError, ValueError, SyntaxError, RuntimeError, subprocess.SubprocessError) as error:
        receipt.update(status="blocked", message=f'{step["name"]}: {error}')
    if "research" in plan:
        receipt["elapsed_seconds"]=round(perf_counter()-began,3)
        if receipt["status"]!="passed":
            receipt.update(research_outcome="blocked",next_condition=plan["research"]["routes"]["blocked"])
    write_receipt(path, receipt)
    return receipt["status"], receipt["message"]


def emit(key, index, message):
    print(chr(96) * 3 + "agent-autoloop")
    print(json.dumps(dict(events=[dict(id=event_id(key, index), type=EVENT,
                                      subject="junqi-serial", summary=message)])))
    print(chr(96) * 3)


def main():
    # The service account can use this repository without modifying Git config.
    os.environ.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="safe.directory",
                      GIT_CONFIG_VALUE_0=ROOT.as_posix(), OPENBLAS_NUM_THREADS="1")
    if len(sys.argv) == 3:
        os.environ["JUNQI_PWSH"] = sys.argv[2]
    try:
        plan, key = load_plan()
        context = read(Path(sys.argv[1]))
        count = len(plan["steps"])
        is_resume, target = resume_index(context, key, count)
        index = target if is_resume else requested_stage(context, key, count)
        serial = ROOT / "build/serial"
        folder = serial / key
        def recheck():
            if load_plan()[1] != key:
                raise ValueError("execution inputs changed during the stage")
        with worker_lock(serial):
            recheck()
            if index is not None:
                bind_research_reservation(plan,key,serial)
            os.environ["JUNQI_SERIAL_KEY"] = key
            status, message = execute_stage(plan, index, folder, resume=is_resume, recheck=recheck)
            print(f"{status.upper()} {message} | {folder}")
            if status == "passed" and index + 1 < count:
                recheck()
                emit(key, index + 1, message)
    except (KeyError, IndexError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"BLOCKED invalid dispatch: {error}")


if __name__ == "__main__":
    main()
