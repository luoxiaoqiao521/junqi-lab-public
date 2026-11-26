"""Descriptive diagnosis; public events only, no scoring or rules duplicated."""
from pathlib import Path
from collections import Counter
import argparse, json, subprocess, tempfile, hashlib, time
ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / 'build/junqi.exe'

def diagnose(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    frames = rows[2:]
    previous = {}
    seen = Counter()
    targets = Counter()
    reverse = repeated = combat = 0
    for frame in frames:
        seat, source, target, outcome, _, _ = frame['event']
        reverse += previous.get(seat) == (target, source)
        repeated += seen[seat, source, target] > 0
        combat += outcome != 0
        previous[seat] = (source, target)
        seen[seat, source, target] += 1
        if frame['no_progress'] > 0:
            targets[target] += 1
    last = rows[-1]
    return dict(plies=len(frames), result=last['result'], combat=combat,
                reverse=reverse, repeated=repeated,
                terminal_no_progress=last['no_progress'],
                quiet_targets=targets.most_common(5))

def terminal_window(path, report, policy):
    frames = [json.loads(line) for line in path.read_text().splitlines()][2:]
    if frames[-1]['result'] != 3 or frames[-1]['no_progress'] != 70:
        return []
    candidates = [tuple(map(int, line.split()[1:])) for line in report.splitlines()
                  if line.startswith('CAND ')]
    if [row[0] for row in candidates] != [frame['seq'] for frame in frames]:
        raise ValueError('Candidate output does not match replay sequence')
    seats = [Counter() for _ in range(4)]
    previous = {}
    for i, (frame, (_, legal, attacks, gap, ties)) in enumerate(zip(frames, candidates)):
        seat, source, target, outcome, _, _ = frame['event']
        reverse = previous.get(seat) == (target, source)
        previous[seat] = (source, target)
        if i < len(frames) - 70:
            continue
        if outcome or not legal:
            raise ValueError('Expected legal quiet move inside collision-free window')
        counts = seats[seat]
        counts['moves'] += 1
        counts['reverse'] += reverse
        counts['attack_available'] += bool(attacks)
        if reverse:
            counts[f'reverse_edge_{source}_{target}'] += 1
            counts['reverse_attack_available'] += bool(attacks)
            if policy != 'random' and frame['decision']['policy'] == 'heuristic-1':
                if attacks:
                    counts[f'attack_gap_{gap}'] += 1
                counts['reverse_has_score_tie'] += bool(ties)
    return seats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seat-window', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    records = []
    with tempfile.TemporaryDirectory(prefix='junqi-diagnose-') as temp:
        for seed in range(1000, 1032):
            for policy in ['random', 'heuristic-ns', 'heuristic-ew']:
                path = Path(temp) / f'{seed}-{policy}.jsonl'
                subprocess.run([str(EXE), 'simulate', str(seed), str(path),
                                str(ROOT/'config/ruleset.json'),
                                str(ROOT/'config/observation.json'), policy],
                               check=True, capture_output=True, timeout=20)
                command = 'replay-candidates' if args.seat_window else 'replay'
                replay = subprocess.run([str(EXE), command, str(path)], check=True,
                                        capture_output=True, text=True, timeout=20)
                record = dict(seed=seed, policy=policy, **diagnose(path))
                if args.seat_window:
                    record['terminal_window'] = terminal_window(path, replay.stdout, policy)
                records.append(record)
    groups = {}
    for policy in ['random', 'heuristic-ns', 'heuristic-ew']:
        games = [r for r in records if r['policy'] == policy]
        plies = sum(r['plies'] for r in games)
        groups[policy] = dict(games=len(games), plies=plies,
            draws=sum(r['result']==3 for r in games),
            combat_rate=sum(r['combat'] for r in games)/plies,
            reverse_rate=sum(r['reverse'] for r in games)/plies,
            repeated_rate=sum(r['repeated'] for r in games)/plies)
    result = dict(exe_sha256=hashlib.sha256(EXE.read_bytes()).hexdigest(),
        scope='Existing frozen seeds: descriptive diagnosis, not new strength validation',
        definitions={'reverse':'same seat reverses its previous move; not piece identity proof',
                     'repeated':'same seat directed edge previously used in game',
                     'quiet_targets':'destination counts on non-resetting plies; node IDs'},
        groups=groups, records=records)
    name = 'stall-seat-window' if args.seat_window else 'stall-diagnosis'
    if args.seat_window:
        result['elapsed_seconds'] = round(time.monotonic() - started, 3)
        result['definitions']['terminal_window'] = (
            'Last 70 plies of a no-progress draw only; array indexed by absolute seat. '
            'Attack means legal move to an enemy occupied square, not a known winning capture. '
            'Gap is chosen minus best attack; ties include any other legal equal-scored move.')
    (ROOT/f'docs/REPORTS/{name}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(groups, indent=2))

if __name__ == '__main__':
    main()
