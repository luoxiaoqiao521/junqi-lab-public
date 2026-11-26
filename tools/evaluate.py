from pathlib import Path
import subprocess,tempfile,json,time,statistics,random,hashlib
root=Path(__file__).resolve().parents[1]
exe=root/'build/junqi.exe'
policies=['random','heuristic-ns','heuristic-ew','heuristic-south','heuristic-east']
# Frozen validation set, no tuning on results. Each seed is one dependent block.
seeds=range(1000,1032)
records=[]
start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='junqi-eval-') as temp:
 for seed in seeds:
  for policy in policies:
   path=Path(temp)/f'{seed}-{policy}.jsonl'
   before=time.perf_counter()
   p=subprocess.run([str(exe),'simulate',str(seed),str(path),str(root/'config/ruleset.json'),str(root/'config/observation.json'),policy],capture_output=True,text=True)
   if p.returncode: raise RuntimeError(p.stderr)
   elapsed=time.perf_counter()-before
   rows=path.read_text().splitlines();header=json.loads(rows[0]);last=json.loads(rows[-1])
   records.append(dict(seed=seed,policy=policy,first=header['first'],result=last['result'],plies=last['seq'],seconds=elapsed))
  print(f'block {seed} saved',flush=True)
  (root/'docs/REPORTS/R4-results.json').write_text(json.dumps(records,indent=2))
def score(result,team):
 if result==4:return None
 return .5 if result==3 else float(result==team+1)
report={'seed_start':1000,'blocks':32,'planned_games':160,'elapsed_seconds':time.perf_counter()-start,'exe_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'groups':{},'peak_memory':'unmeasured','timing_scope':'whole CLI including JSONL disk IO, not pure decision latency'}
for modes in [('heuristic-ns','heuristic-ew'),('heuristic-south','heuristic-east')]:
 differences=[];outcomes=[0]*5
 for seed in seeds:
  block={x['policy']:x for x in records if x['seed']==seed}
  values=[]
  for team,mode in enumerate(modes):
   result=block[mode]['result'];outcomes[result]+=1
   candidate=score(result,team);baseline=score(block['random']['result'],team)
   if candidate is not None and baseline is not None:values.append(candidate-baseline)
  if len(values)==2:differences.append(statistics.mean(values))
 rng=random.Random(20261117)
 means=sorted(statistics.mean(rng.choices(differences,k=len(differences))) for _ in range(4000)) if len(differences)==32 else []
 report['groups']['+'.join(modes)]={'results_by_enum':outcomes,'paired_gain':statistics.mean(differences) if means else None,'block_bootstrap_95_percent':[means[100],means[3899]] if means else None,'note':'fixed-size approximate block bootstrap; only synthetic layout distribution; no tuning; no inference if truncated blocks'}
for policy in policies:
 times=sorted(x['seconds'] for x in records if x['policy']==policy)
 report.setdefault('timing',{})[policy]={'median_seconds':statistics.median(times),'p95_seconds':times[int(.95*(len(times)-1))]}
(root/'docs/REPORTS/R4-summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
