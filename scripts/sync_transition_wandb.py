"""Reuse durable metric uploads with GPU telemetry disabled for this request."""
import argparse
import json
from pathlib import Path
import psutil
import sync_subtask_wandb as bridge

def active(directory):
    try:
        record=json.loads((directory/"trainer_process.json").read_text())
        process=psutil.Process(record["pid"])
        return abs(process.create_time()-record["created"])<.1 and process.is_running() and process.status()!=psutil.STATUS_ZOMBIE
    except (OSError,ValueError,KeyError,psutil.Error):
        return False

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    bridge.is_current_job=active
    bridge.gpu_metrics=lambda:{}
    bridge.sync(args.output,bridge.DEFAULT_PROJECT,None,True)

if __name__=="__main__":
    main()
