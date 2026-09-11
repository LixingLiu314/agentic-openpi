"""Bounded local/cloud startup check; no recurring progress reader."""
import argparse,json,math,os,time,statistics
from pathlib import Path
# This read-only checker runs in a different W&B SDK environment. Never attach
# its API client to the training writer's version-specific service transport.
os.environ.pop('WANDB_SERVICE',None)
os.environ.pop('_WANDB_SERVICE',None)
import wandb
def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args()
    rows=[json.loads(s) for s in (a.run/"metrics.jsonl").read_text().splitlines()]
    r=next(r for r in rows if r.get("event")=="train" and r["step"]==10)
    expected={"train/subtask_ce":r["loss_subtask"],"train/action_flow_mse_global_only_all32":r["loss_action"],
        "optim/grad_norm_b":r["grad_norms"]["backbone"],"optim/grad_norm_a":r["grad_norms"]["action"]}
    last=None
    for attempt in range(4):
        try:
            run=wandb.Api(timeout=40).run("xiahy23-tsinghua-university/agentic-openpi-pi05-subtask/"+a.run.name)
            history=list(run.scan_history(min_step=0,max_step=100,page_size=100))
            merged={k:v for row in history if row.get("trainer/step")==10 for k,v in row.items() if v is not None}
            for k,v in expected.items():assert math.isclose(v,merged[k],rel_tol=1e-7,abs_tol=1e-9),(k,merged.get(k),v)
            for key in ["media/first_update_inputs","media/first_update_actions","media/subtask_per_class"]:
                assert any(row.get(key) for row in history),key
            (a.run/"startup_verified.json").write_text(json.dumps({"passed":True,"step":10,"metrics":expected,"media":True,"time":time.time()},indent=2)+"\n")
            seconds=[item['seconds'] for item in rows if item.get('event')=='train' and 3<=item['step']<=10]
            mean=statistics.mean(seconds)
            (a.run/'startup_eta.json').write_text(json.dumps(dict(observed_step=10,
                mean_update_seconds=mean,remaining_updates=4990,pure_training_seconds=4990*mean,
                validation_save_gate_allowance_seconds=7200,estimated_completion_unix=time.time()+4990*mean+7200,
                note='Initial estimate only; native autoregressive validation overhead uncertain',time=time.time()),indent=2)+'\n')
            return
        except Exception as e:
            last=e
            if attempt<3:time.sleep(20)
    raise RuntimeError("Startup verification failed") from last
if __name__=="__main__":main()
