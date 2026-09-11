"""One bounded step10 W&B consistency check, triggered by the trainer itself."""
import argparse
import json
import math
from pathlib import Path
import time
import traceback

import wandb


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    try:
        rows=[json.loads(s) for s in (args.output/"metrics.jsonl").read_text().splitlines()]
        row=next(r for r in rows if r.get("event")=="train" and r["step"]==10)
        expected={"train/action_flow_mse_all32":row["loss_action"],"train/subtask_ce":row["loss_subtask"],
            "optim/grad_norm_a":row["grad_norms"]["action"],"optim/grad_norm_b":row["grad_norms"]["backbone"],
            **{"decision/"+k:row[k] for k in ["rank_arm_pairs","rank_object_pairs","rank_arm_hinge","rank_object_hinge",
                "grounding_ce","grounding_distance_px","first25_native_mse","action_weighted_flow"]}}
        failure=None
        for attempt in range(3):
            try:
                run=wandb.Api(timeout=60).run("xiahy23-tsinghua-university/agentic-openpi-pi05-subtask/"+args.output.name)
                records=list(run.scan_history(min_step=0,max_step=80,page_size=80))
                matching=[r for r in records if r.get("trainer/step")==10]
                merged={k:v for r in matching for k,v in r.items() if v is not None}
                for k,value in expected.items():assert math.isclose(merged[k],value,rel_tol=1e-7,abs_tol=1e-9),(k,merged.get(k),value)
                assert any(r.get("media/first_update_inputs") for r in records)
                assert any(r.get("media/first_update_actions") for r in records)
                assert any(r.get("media/subtask_per_class") for r in records)
                result=dict(passed=True,optimizer_step=10,metrics=expected,media_verified=True,
                            scope="One startup comparison; no subsequent progress polling",time=time.time())
                (args.output/"startup_verified.json").write_text(json.dumps(result,indent=2)+"\n")
                print(json.dumps(result));return
            except Exception as error:
                failure=error
                if attempt<2:time.sleep(20*(attempt+1))
        raise RuntimeError("Startup cloud comparison failed") from failure
    except BaseException as error:
        (args.output/"startup_failure.json").write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc(),time=time.time()),indent=2)+"\n")
        raise


if __name__=="__main__":main()
