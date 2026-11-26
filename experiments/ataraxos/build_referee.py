"""Compile the pinned upstream board types and CPU-executable kernel bodies."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "upstream"
PIN = "92db29e8ffc323b1b8a2804b5c3f84695d036b05"
BUILD = ROOT / "build"
VS = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat")


def checked_source(name):
    path = SOURCE / name
    expected = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", f"{PIN}:{name}"], text=True).strip()
    actual = subprocess.check_output(["git", "-C", str(SOURCE), "hash-object", str(path)], text=True).strip()
    if actual != expected:
        raise RuntimeError(f"Upstream source changed: {name}")
    return path.read_text(encoding="utf-8")


def main(review=False, legacy=False):
    BUILD.mkdir(exist_ok=True)
    board = checked_source("src/env/stratego_board.h")
    types = board[board.index("const int BOARD_SIZE"):board.index("/// Checks that")]
    counts = next(line for line in board.splitlines() if "CLASSIC_INITIAL_COUNTS =" in line)
    (BUILD / "upstream_board.inc").write_text(types + counts + "\n", encoding="utf-8")
    for label in ("action", "infostate"):
        body = checked_source(f"src/env/cuda/{label}_kernels.cu")
        body = re.sub(r"^#include[^\n]*\n", "", body, flags=re.M)
        (BUILD / f"upstream_{label}.inc").write_text(body, encoding="utf-8")
    for label, filename, stop in (
        ("two_square", "src/env/rules/twosquare_state.cu", "__global__ void EnvStateDeadCellsKernel"),
        ("chase", "src/env/rules/chase_state.cu", "void ResetChaseState("),
    ):
        body = checked_source(filename).split(stop, 1)[0]
        body = re.sub(r"^#include[^\n]*\n", "", body, flags=re.M)
        (BUILD / f"upstream_{label}.inc").write_text(body, encoding="utf-8")
    terminal = checked_source("src/env/cuda/termination_kernels.cu")
    (BUILD / "upstream_terminal.inc").write_text(
        terminal[terminal.index("__global__ void IncrementTerminationCounterKernel"):], encoding="utf-8")
    source = ROOT / "cpu_referee.cpp"
    artifact = "official_referee.dll"
    if review:
        name = "review_legacy" if legacy else "review_referee"
        target = BUILD / f"{name}.cpp"
        content = source.read_text(encoding="utf-8")
        if legacy:
            content = content.replace("kQuietLimit=200, kCap=4000", "kQuietLimit=70, kCap=3000")
            content = content.replace("if (use_rules) {", "if (false && use_rules) {")
            original = content.index("  std::fill_n(m.chase_actions,MAX_CHASE_LENGTH,-1);", content.index("API int play"))
            content = content[:original] + '\n  if (m.flag) return m.result=m.flag;\n  legal(m,mask,m.actor,false);\n  if (std::none_of(mask,mask+NUM_ACTIONS,[](bool x){return x;})) return m.result=3-m.actor;\n  if (m.quiet>=kQuietLimit) return m.result=3;\n  if (m.ply>=kCap) return m.result=4;\n  return 0;\n}\n'
        target.write_text(content + "\n" + (ROOT / "review_cells.inc").read_text(encoding="utf-8-sig"), encoding="utf-8")
        source, artifact = target, f"{name}.dll"
    command = f'call "{VS}" >nul && cl /nologo /std:c++17 /EHsc /O2 /MT /LD /W3 /utf-8 /I"{BUILD}" "{source}" /link /OUT:{artifact}'
    subprocess.run(f'cmd.exe /d /s /c "{command}"', cwd=BUILD, check=True, timeout=55)


if __name__ == "__main__":
    import sys
    main("--review" in sys.argv[1:], "--legacy" in sys.argv[1:])
