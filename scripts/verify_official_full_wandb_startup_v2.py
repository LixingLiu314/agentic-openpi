"""One bounded check of full-B step10; exits after cloud metrics are verified."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--identity",type=Path)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--check-existing",action="store_true")
    args=parser.parse_args()
    # Preserve the venv launcher path: resolving its symlink bypasses pyvenv.cfg.
    display_python=os.path.abspath('.stage1_staging/wandb_display_env/bin/python')
    check="import sys,json,wandb; assert callable(getattr(wandb,'Api',None)); assert 'wandb_display_env' in sys.prefix; print(json.dumps({'passed':True,'executable':sys.executable,'prefix':sys.prefix,'wandb_file':wandb.__file__,'wandb_version':wandb.__version__}))"
    result=subprocess.run([display_python,'-c',check],capture_output=True,text=True,timeout=45)
    if result.returncode:raise RuntimeError('W&B interpreter preflight failed: '+result.stderr[-2000:])
    runtime=json.loads(result.stdout)
    if args.preflight:
        args.report.write_text(json.dumps(runtime,indent=2));print(json.dumps(runtime));return
    assert args.output is not None and (args.identity is not None or args.check_existing)
    identity=json.loads(args.identity.read_text()) if args.identity else None
    deadline=time.monotonic()+900
    row=None
    while time.monotonic()<deadline:
        if not args.check_existing:
            process=psutil.Process(identity["pid"])
            assert abs(process.create_time()-identity["created"])<.1
            if process.status()==psutil.STATUS_ZOMBIE:raise RuntimeError("Trainer exited before logging verification")
        path=args.output / "metrics.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                try:item=json.loads(line)
                except json.JSONDecodeError:continue  # The current append can be incomplete.
                if item.get("event")=="train" and item["step"]==10:row=item;break
        if row:break
        if args.check_existing:raise RuntimeError('Existing run has no step10')
        time.sleep(5)
    if not row:raise TimeoutError("No step10 training event")
    config=json.loads((args.output / "run_config.json").read_text())
    assert config["initialization"]=="official_pi05_base" and config["inherited_training_updates"]==0
    initialization=json.loads((args.output / "initialization.json").read_text())
    assert initialization["decoder_sha256"]=="bab5a0bce442eb7e5759a15deea98c1ff0e75b326941e9abff2e3eacc7eb53ee"
    expected={"train/action_flow_mse_all32":row["loss_action"],"train/subtask_ce":row["loss_subtask"]}
    for name,suffix in [("action","a"),("backbone","b"),("backbone_vision","b_vision"),("backbone_language","b_language")]:
        value=row["grad_norms"][name]
        assert math.isfinite(value) and value>0,(name,value)
        expected["optim/grad_norm_"+suffix]=value
    snapshot=args.report.with_suffix(".expected.json")
    snapshot.write_text(json.dumps(expected,indent=2))
    # The isolated display runtime has the supported public W&B API.
    code='''import json,sys,math,wandb
expected=json.load(open(sys.argv[1]))
run=wandb.Api().run('xiahy23-tsinghua-university/agentic-openpi-pi05-subtask/'+sys.argv[2])
assert run.config['observability_version']==2
assert run.config['initialization']=='official_pi05_base'
rows=list(run.scan_history(keys=['trainer/step',*expected],min_step=0,max_step=12,page_size=30))
row=next(r for r in rows if r.get('trainer/step')==10 and all(r.get(k) is not None for k in expected))
for k,v in expected.items():assert math.isclose(row[k],v,rel_tol=1e-7,abs_tol=1e-9),(k,row[k],v)
print(json.dumps({'passed':True,'optimizer_step':10,'metrics':expected,'scope':'one startup comparison, no later polling'}))
'''
    for attempt in range(3):
        result=subprocess.run([display_python,'-c',code,str(snapshot),args.output.name],
                              capture_output=True,text=True,timeout=90)
        if result.returncode==0:
            record=json.loads(result.stdout)
            record.update(runtime=runtime,existing_run_audit=args.check_existing,run_id=args.output.name)
            args.report.write_text(json.dumps(record,indent=2));print(json.dumps(record));return
        if attempt<2:time.sleep(20*(attempt+1))
    raise RuntimeError("Full-B W&B startup verification failed: "+result.stderr[-2000:])


if __name__=="__main__":
    main()
