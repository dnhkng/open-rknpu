"""Stream independent elementwise cases into one reusable board slot."""
from pathlib import Path
import os
import argparse
import subprocess
p=argparse.ArgumentParser()
p.add_argument('op',choices=['add','mul','sub','max','stride2','depthwise_stride2'])
p.add_argument('--adb',default=os.environ.get('ADB','adb'))
a=p.parse_args();root=Path(__file__).resolve().parent
remote='/userdata/open-npu-research/mul_suite'
def adb(*args):
    return subprocess.run([a.adb,*args],check=True,capture_output=True,text=True,timeout=60).stdout
logs=[]
try:
    for i in range(3 if a.op=='stride2' else 12):
        for prefix,suffix in [('model','bin'),('input','u8'),('expected','i8')]:
            adb('push',str(root/(a.op+'_suite')/f'{prefix}{i:03}.{suffix}'),f'{remote}/{prefix}000.{suffix}')
        out=adb('shell',f'/userdata/open-npu-research/board_api_test {remote} 1')
        logs.append(f'Source model {i}\n'+out)
        if f'PASS: 1 models, {16 if a.op=='depthwise_stride2' else 32} inferences, {768 if a.op=='depthwise_stride2' else 1536 if a.op=='stride2' else 6144} output bytes' not in out:
            raise RuntimeError(out)
        print(a.op,i,'passed',flush=True)
    logs.append(('PASS: 12 models, 192 inferences, 9216 output bytes' if a.op=='depthwise_stride2' else 'PASS: 3 models, 96 inferences, 4608 output bytes' if a.op=='stride2' else 'PASS: 12 models, 384 inferences, 73728 output bytes')+'\n')
finally:
    (root/(a.op+'_suite.log')).write_text('\n'.join(logs))
