"""MIT. Replay one small captured survey bundle at a time on the board."""
import json
from pathlib import Path
import subprocess
from run_oracle import adb, BOARD

ROOT=Path(__file__).resolve().parent/'primitive_survey'
results={}
adb('push',str(ROOT/'replay-survey'),BOARD+'/replay-survey')
for name in json.loads((ROOT/'tasks.json').read_text()):
    bundle=ROOT/(name+'.replay')
    if bundle.stat().st_size>262144:raise ValueError('small replay bundles only')
    adb('push',str(bundle),BOARD+'/survey.replay')
    text=adb('shell',f'cd {BOARD} && chmod +x replay-survey && ./replay-survey survey.replay',timeout=15).decode()
    (ROOT/(name+'_replay.log')).write_text(text)
    # Old board adbd does not reliably propagate the child exit code.
    results[name]=dict(passed='PASS cases=4 ' in text,output=text.strip())
    (ROOT/'replay_results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(name,results[name]['passed'],flush=True)
    if 'submit/sync' in text:raise RuntimeError('hardware submission failed; inspect before continuing')
adb('shell',f'rm -f {BOARD}/survey.replay')
