import json,os,psutil,subprocess,sys,time
from pathlib import Path
p=Path.cwd();r=p/'logs/pi05_piper_backbone_grad/batch_retry_20260908'
d=json.loads((r/'retry.process.json').read_text())
try:
 owner=psutil.Process(d['pid'])
 if abs(owner.create_time()-d['created'])<.1:owner.wait(timeout=600)
except psutil.NoSuchProcess:pass
cmd=[sys.executable,'scripts/gpu_reservation.py','run-concurrent','--log',str(r/'raw_repeat_managed.log'),'--',sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=8','scripts/train_backbone_gradient.py','--mode','full','--output',str(r/'full_b32_raw_repeat_gate'),'--steps','8','--warmup','1','--global-batch','256','--batch-size','32','--accumulation','1','--memory-fraction','0.9','--checkpoint-every','8','--eval-samples','8','--eval-draws','1','--workers','2','--engineering-smoke']
result=subprocess.run(cmd)
(r/'raw_repeat_exit.json').write_text(json.dumps({'exit_code':result.returncode,'time':time.time()}))
raise SystemExit(result.returncode)
