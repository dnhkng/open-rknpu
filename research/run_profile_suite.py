"""MIT. Stream small independent test cases; preserve errors and stop on failure."""
from pathlib import Path
import os
import argparse,subprocess,json
from open_rknpu.sequence import decode_sequence
p=argparse.ArgumentParser();p.add_argument('suite');p.add_argument('--start',type=int,default=0);a=p.parse_args()
root=Path(__file__).resolve().parent;folder=root/a.suite
adb=os.environ.get('ADB', 'adb')
remote='/userdata/open-npu-research/mul_suite';results=[]
for path in sorted(folder.glob('model*.bin')):
    index=path.stem[5:]
    if int(index)<a.start:continue
    info=decode_sequence(path.read_bytes())
    n=(folder/f'input{index}.u8').stat().st_size//info['input_bytes']
    for prefix,suffix in [('model','bin'),('input','u8'),('expected','i8')]:
        subprocess.run([adb,'push',str(folder/f'{prefix}{index}.{suffix}'),f'{remote}/{prefix}000.{suffix}'],check=True,capture_output=True,timeout=30)
    r=subprocess.run([adb,'shell',f'/userdata/open-npu-research/board_api_test {remote} 1'],capture_output=True,text=True,timeout=40)
    passed=f'PASS: 1 models, {n} inferences, {n*info["output_bytes"]} output bytes' in r.stdout
    results.append(dict(model=index,passed=passed,inferences=n,bytes=n*info['output_bytes'],output=r.stdout+r.stderr))
    (folder/f'board_results_{a.start}.json').write_text(json.dumps(results,indent=2))
    total=sum(e['inferences'] for e in results); exact=sum(e['bytes'] for e in results)
    (folder/'board_summary.txt').write_text(
        f'{"PASS" if all(e["passed"] for e in results) else "FAIL"}: {len(results)} models, '
        f'{total} inferences, {exact} exact output bytes (board_api_test)\n')
    print(index,passed,r.stdout.strip(),flush=True)
    if not passed:raise SystemExit(1)
