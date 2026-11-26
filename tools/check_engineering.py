from pathlib import Path
import re,json
root=Path(__file__).resolve().parents[1]
errors=[];files=[];graph={};line_count=0
for folder in ['core','observe','strategy','sim','web','tools']:
 for f in (root/folder).rglob('*'):
  if not f.is_file() or f.suffix not in ['.h','.cpp','.js','.css','.html','.py','.ps1']:continue
  if f.name=='demo.js':continue
  text=f.read_text(encoding='utf-8-sig');lines=text.splitlines();files.append(f);line_count+=len(lines)
  if len(lines)>500:errors.append(f'{f}: exceeds 500 lines')
  if folder=='core' and re.search(r'#include.*(observe|strategy|sim|web)',text):errors.append(f'{f}: core dependency direction')
  includes=[]
  for inc in (re.findall(r'#include\s+"([^"]+)"',text) if f.suffix in ['.h','.cpp'] else []):
   p=(f.parent/inc).resolve()
   if not p.exists():p=(root/'core'/inc).resolve()
   if not p.exists():errors.append(f'{f}: missing include {inc}')
   includes.append(p)
  graph[f.resolve()]=includes
def visit(f,seen):
 if f in seen:return
 seen.add(f)
 if f.name=='state.h' or (f.parent.name=='observe' and f.name=='filter.h') or f.parent.name=='sim':errors.append('player dependency reaches truth: '+str(f))
 for child in graph.get(f,[]):visit(child,seen)
for f in files:
 if 'strategy' in f.relative_to(root).parts or f.name=='observation.h':visit(f.resolve(),set())
dest=root/'dist/junqi-mvp'
allowed={'junqi.exe','config/ruleset.json','config/ruleset-selfplay.json','config/observation.json','web/index.html','web/style.css','web/review.js','web/app.js','web/demo.js','Start-MVP.cmd','start-mvp.ps1','README.md'}
actual={p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()}
if actual!=allowed:errors.append('package whitelist mismatch '+str(actual^allowed))
sizes={str(p.relative_to(dest)):p.stat().st_size for p in dest.rglob('*') if p.is_file()}
if sum(sizes.values())>100000000:errors.append('package hard limit')
if sizes.get('junqi.exe',0)>5000000:errors.append('core budget')
if sum(v for k,v in sizes.items() if k.startswith('web'))>20000000:errors.append('UI budget')
if errors:raise SystemExit('\n'.join(errors))
print(f'PASS dependency/include closure, file line budget, package whitelist; files={len(files)} lines={line_count} bytes={sum(sizes.values())}')
print('Function <=60 lines: manual review of new functions; lexical scan not a proof. Task additions tracked by M1/M2/M3 source scripts. Runtime isolation separately tested.')

