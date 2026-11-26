from pathlib import Path
import hashlib
import json,subprocess,sys
root=Path(__file__).resolve().parents[1]
ticket=json.loads((root/'ops/task.json').read_text(encoding='utf-8-sig'))
def git(*args):
 return subprocess.check_output(['git','-C',str(root),*args],text=True).splitlines()
def allowed(path):
 return any(path==p.rstrip('/') or path.startswith(p.rstrip('/')+'/') for p in ticket['allowed'])
changed=set(git('diff','--name-only',ticket['baseline'],'--'))
untracked=set(git('ls-files','--others','--exclude-standard'))
untracked.difference_update(ticket.get('preexisting_untracked',[]))
if 'local_untracked_manifest' in ticket:
 manifest=ticket['local_untracked_manifest']
 path=(root/manifest['path']).resolve()
 if not path.is_relative_to(root/'build'):
  raise SystemExit('Preexisting untracked manifest must stay in local build')
 raw=path.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=manifest['sha256']:
  raise SystemExit('Preexisting untracked manifest differs from the ticket')
 untracked.difference_update(raw.decode('utf8').splitlines())
changed.update(untracked)
bad=sorted(p for p in changed if not allowed(p))
if bad:raise SystemExit('Out of task scope: '+', '.join(bad))
assert not allowed('data/raw/probe') and not allowed('third_party/probe'), 'protected scope admitted'
print('PASS task scope '+ticket['id']+'; changed paths='+str(len(changed)))
