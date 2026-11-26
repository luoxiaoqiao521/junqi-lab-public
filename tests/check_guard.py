from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parents[1]
probe=root/'strategy'/'guard_fixture'/'bad.h'
if probe.exists(): raise SystemExit('fixture already exists')
probe.parent.mkdir(exist_ok=True)
try:
 probe.write_text('#include "../../core/state.h"\n')
 result=subprocess.run([sys.executable,str(root/'tools/check_engineering.py')],capture_output=True,text=True)
 assert result.returncode != 0 and 'player dependency reaches truth' in result.stdout+result.stderr,result
finally:
 probe.unlink();probe.parent.rmdir()
print('PASS nested forbidden-include rejection')
