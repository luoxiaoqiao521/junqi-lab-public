"""Run the frozen confirmation once and hand its evidence to the commit worker."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from evaluate_transfer import compare, totals
from learn_short import EXE, ROOT, RULES, np

BASELINE = "98172f3561f0f05ad657dfe8ae147853168a82cf"
CANDIDATE = "43e055a01cb41c7ec3c19cc969ebe3fe6adec9e39a103417333fd9aa3fe7e624"
INITIAL = "f7caabe6b9fbc594e0dc431279d21129c38a85327d4c7469c594fb84c2b87fb3"
STEM = "docs/REPORTS/l2c-confirm"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def games(weights, policy):
    started = time.monotonic()
    proc = subprocess.run(
        [str(EXE), "transfer-probe", str(RULES), *map(str, weights),
         "60000", "128", policy, "0"], capture_output=True, text=True, check=True, timeout=180)
    rows = [json.loads(line) for line in proc.stdout.splitlines()]
    if [(r["seed"], r["seat"]) for r in rows] != [(s, v) for s in range(60000, 60128) for v in range(4)]:
        raise ValueError("confirmation blocks mismatch")
    count = totals(rows)
    count["wall_seconds"] = time.monotonic() - started
    count["action_us_mean"] = sum(r["learner_ns"] for r in rows) / max(1, count["decisions"]) / 1000
    count["score_bounds"] = [(count["wins"] + .5 * count["draws"] + unknown) / len(rows)
                             for unknown in (0, count["truncated"])]
    return rows, count


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--handoff", action="store_true")
    args = cli.parse_args()
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    task = json.loads((ROOT / "ops/task.json").read_text(encoding="utf-8-sig"))
    if head != BASELINE or task["baseline"] != BASELINE or task["id"] != "L2C":
        raise ValueError("confirmation ticket or HEAD mismatch")
    candidate, initial = ROOT / "build/l2b-opening/trained.npy", ROOT / "build/l1-learning/initial.npy"
    if (digest(candidate), digest(initial)) != (CANDIDATE, INITIAL):
        raise ValueError("frozen model hash mismatch")
    if not any(line.startswith("CHECK PASS elapsed_seconds=") for line in
               (ROOT / f"{STEM}-check.txt").read_text(encoding="utf-8-sig").splitlines()):
        raise ValueError("engineering check receipt missing")
    output = ROOT / f"{STEM}.json"
    if output.exists():
        raise FileExistsError("confirmation already recorded")
    weights = np.load(initial, allow_pickle=False)
    rows, summary = {}, {}
    for name, params, policy in [
            ("initial", weights, "short"), ("random", weights, "random"),
            ("heuristic-1", weights, "heuristic"),
            ("trained", np.load(candidate, allow_pickle=False), "short")]:
        rows[name], summary[name] = games(params, policy)
    contrasts = {name: compare(rows[name], rows["trained"]) for name in ("initial", "random", "heuristic-1")}
    eligible = all(not counts["truncated"] for counts in summary.values()) and all(
        contrast["exploratory_block_interval"][0] > 0 for contrast in contrasts.values())
    result = dict(eligible=eligible, models={"initial": INITIAL, "trained": CANDIDATE},
                  executable=digest(EXE), rules=digest(RULES), summary=summary, contrasts=contrasts, rows=rows)
    with output.open("x", encoding="utf8") as handle:
        json.dump(result, handle, indent=2)
    receipt = [
        "# L2C 独立确认", "",
        "种子 60000–60127，各轮换四席。其余三席为随机策略。",
        "主要对比为冻结候选对冻结初始化，random 与 heuristic-1 为并列使用基线。", "",
        "|策略|胜|和|负|截断|团队均分下界|平均选招微秒|", "|---|---:|---:|---:|---:|---:|---:|"
    ]
    for name, s in summary.items():
        receipt.append(f"|{name}|{s['wins']}|{s['draws']}|{s['losses']}|{s['truncated']}|"
                       f"{s['score_bounds'][0]:.4f}|{s['action_us_mean']:.2f}|")
    for name, diff in contrasts.items():
        receipt.append(f"\n候选相对 {name} 的配对差界 {diff['difference_bounds']}，"
                       f"区组重采样区间 {diff['exploratory_block_interval']}。")
    conclusion = "达到 L3 准入。" if eligible else "未达到 L3 准入，当前确认链结案。"
    receipt.extend(["", conclusion, "",
                    "行动计时包含实际选招调用，排除输入观察构建及裁判机会诊断。"
                    "统计只覆盖一个固定候选，训练种子波动未测。完整结果与哈希见同名 JSON。"])
    (ROOT / f"{STEM}.md").write_text("\n".join(receipt) + "\n", encoding="utf8")
    next_ticket = "L3" if eligible else None
    (ROOT / "ops/CONTINUE.md").write_text(
        "# 串行续跑现役入口\n\n最新证据：docs/REPORTS/l2c-confirm.md。\n\n" +
        ("下一票 L3：冻结同一候选，分别更换搭档与对手。\n\n" if eligible else
         "本链已结案。新研究仅在有具体瓶颈与对照时另开票。\n\n") +
        f"<!-- junqi-next: {next_ticket or 'END'} -->\n", encoding="utf8")
    path = ROOT / "ops/progress.json"
    status = json.loads(path.read_text(encoding="utf-8-sig"))
    status.update(current="L2C", baseline=BASELINE, next=next_ticket, L2C=conclusion)
    path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(conclusion)
    if args.handoff:
        detail = dict(chain="junqi-20261118-b5300dd", ticket="L2C", baseline=BASELINE, attempt=1,
                      receipt=f"{STEM}.md", log=f"{STEM}-check.txt", next=next_ticket, minutes=120)
        print("\n```agent-autoloop\n" + json.dumps({"events": [{
            "id": f"junqi-l2c-ready:{BASELINE}", "type": "junqi.post-mvp.ready",
            "subject": "junqi-post-mvp", "summary": conclusion, "detail": json.dumps(detail)
        }]}) + "\n```")


if __name__ == "__main__":
    main()
