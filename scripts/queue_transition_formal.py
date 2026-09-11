"""Wait on the existing managed-job lock, then run one finite seed42 job."""
import fcntl
import json
from pathlib import Path
import sys
import time
import gpu_reservation

def main():
    root=Path.cwd()
    directory=root/"logs/pi05_subtask_stage1/gpu_reservation"
    logfile=root/"logs/pi05_piper_transition/r1_seed42_queued.log"
    command=[sys.executable,"scripts/run_transition_formal.py","--output",str(root/"checkpoints/pi05_piper_transition/r1_seed42")]
    print(json.dumps({"event":"queued","command":command,"waiting_for":"managed job lock"}),flush=True)
    while True:
        with (directory/"job.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            fcntl.flock(lock,fcntl.LOCK_UN)
        try:
            code=gpu_reservation.run(directory,logfile,command)
            print(json.dumps({"event":"finite_pipeline_exit","exit_code":code}),flush=True)
            raise SystemExit(code)
        except BlockingIOError:
            time.sleep(.5)

if __name__=="__main__":
    main()
