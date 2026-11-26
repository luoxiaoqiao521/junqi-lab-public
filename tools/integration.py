from pathlib import Path
import json, subprocess, tempfile, time, hashlib
root=Path(__file__).resolve().parents[1]
exe=root/'build/junqi.exe'
def run(args,ok=True):
 p=subprocess.run([str(exe),*map(str,args)],capture_output=True,text=True)
 if (p.returncode==0)!=ok: raise AssertionError(p.stdout+p.stderr)
 return p.stdout
start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='junqi-test-') as d:
 d=Path(d);a=d/'a.jsonl';b=d/'b.jsonl'
 params=[root/'config/ruleset.json',root/'config/observation.json']
 run(['simulate',42,a,*params]);run(['simulate',42,b,*params])
 assert a.read_bytes()==b.read_bytes()
 print(run(['replay',a]).strip())
 rows=[json.loads(l) for l in a.read_text(encoding='utf-8').splitlines()]
 original=a.read_text(encoding='utf-8')
 bad=d/'bad.jsonl'
 variants=[]
 variants.append(original[:-2]);variants.append('\n'.join(original.splitlines()[:-1])+'\n')
 for mutate in [lambda x:x[2].update(seq=999),lambda x:x[2]['event'].__setitem__(1,255),lambda x:x[0].update(rules_hash='0'),lambda x:x[0].update(topology='wrong'),lambda x:x[2]['views'][1].__setitem__(0,11),lambda x:x[2].update(hash='wrong'),lambda x:x[0].update(observation_hash='0')]:
  changed=json.loads(json.dumps(rows));mutate(changed);variants.append('\n'.join(json.dumps(x,ensure_ascii=False) for x in changed)+'\n')
 for text in variants:
  bad.write_text(text,encoding='utf-8');run(['replay',bad],False)
 run(['simulate',42,a,*params],False)
 run(['replay',a,'extra'],False)
 huge=d/'huge.jsonl'
 with huge.open('wb') as f: f.seek(64000000);f.write(b'x')
 run(['replay',huge],False)
 h=d/'heuristic.jsonl';h2=d/'heuristic2.jsonl'
 run(['simulate',42,h,*params,'heuristic-ns'])
 run(['simulate',42,h2,*params,'heuristic-ns'])
 assert h.read_bytes()==h2.read_bytes()
 run(['replay',h])
 altered=[json.loads(x) for x in h.read_text().splitlines()]
 assert altered[1]['views'][0].count(-1) > 0
 altered[2]['decision']['total']+=1
 bad.write_text('\n'.join(json.dumps(x) for x in altered)+'\n')
 run(['replay',bad],False)
 print('PASS heuristic deterministic/legal replay and tampered explanation rejection')
 for version in (2,3):
  for entry,seats in (('simulate-full','all'),('simulate','ns')):
   name=f'heuristic-{version}'
   policy=f'{name}-{seats}'
   path=d/f'{policy}.jsonl'
   run([entry,42,path,*params,policy])
   records=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
   assert records[0]['policy']==policy and any(row['decision']['policy']==name for row in records[2:])
   run(['replay',path])
   records[0]['policy']=f'heuristic-{seats}'
   bad.write_text(''.join(json.dumps(row)+'\n' for row in records),encoding='utf8')
   run(['replay',bad],False)
 print('PASS versioned heuristics in full and dark simulation and replay, including wrong-version rejection')
 rules=json.loads((root/'config/ruleset-selfplay.json').read_bytes())
 obs=json.loads(params[1].read_bytes())
 rules['experimental']['marshal_death.reveal_flag']=True
 obs['marshal_reveals_flag']=True
 profile=d/'practice.json'; view=d/'practice-view.json'
 profile.write_text(json.dumps(rules),encoding='utf-8');view.write_text(json.dumps(obs),encoding='utf-8')
 p=d/'flag-route.jsonl';q=d/'flag-route-repeat.jsonl';review=d/'flag-route-review.jsonl'
 run(['simulate',44,p,profile,view,'flag-route']);run(['simulate',44,q,profile,view,'flag-route'])
 assert p.read_bytes()==q.read_bytes()
 run(['review',p,review]);run(['replay',review]);run(['replay',p])
 practice=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
 assert all(x['decision']['policy']=='flag-route-1' for x in practice[2:])
 assert any(x['event'][5] for x in practice[2:]), 'Practice game must cover public flag reveal'
 practice[2]['decision']['objective']+=1;practice[2]['decision']['total']+=1
 bad.write_text('\n'.join(json.dumps(x) for x in practice)+'\n',encoding='utf-8');run(['replay',bad],False)
 print('PASS flag route with marshal reveal, deterministic replay, review export and tampered score rejection')
 print('PASS deterministic JSONL, overwrite rejection, 9 corruption/incomplete/version cases')
 actual=hashlib.sha256(a.read_bytes()).hexdigest()
 expected=(root/'tests/golden_seed42.sha256').read_text().strip()
 assert actual==expected, 'Golden log changed; investigate before updating baseline'
 print('golden_seed42_sha256='+actual)
print('integration_seconds=%.3f'%(time.perf_counter()-start))
