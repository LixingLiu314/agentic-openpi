"""Exercise the real handoff primitives with small CPU-only processes."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import psutil

from continue_official_full_observed import identified, zombie_exit


for exit_code in (0, 7):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        script = '''import subprocess,sys,json,time,signal
from pathlib import Path
p=subprocess.Popen([sys.executable,'-c',"import time,sys;time.sleep(2);sys.exit(int(sys.argv[1]))",sys.argv[2]],start_new_session=True)
Path(sys.argv[1]).write_text(str(p.pid))
signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))
try:p.wait()
finally:
    if p.poll() is None:p.terminate();p.wait()
'''
        parent = subprocess.Popen([sys.executable, '-c', script, str(root/'child'), str(exit_code)], start_new_session=True)
        try:
            limit=time.monotonic()+5
            while not (root/'child').exists():
                assert time.monotonic()<limit
                time.sleep(.02)
            child=psutil.Process(int((root/'child').read_text()))
            record={'pid':child.pid,'created':child.create_time()}
            os.kill(parent.pid,signal.SIGSTOP)
            time.sleep(.05)
            assert psutil.Process(parent.pid).status()==psutil.STATUS_STOPPED
            assert identified(record).status()!=psutil.STATUS_STOPPED
            while zombie_exit(identified(record)) is None:
                assert time.monotonic()<limit
                time.sleep(.05)
            assert zombie_exit(identified(record))==exit_code
            os.kill(parent.pid,signal.SIGTERM)
            os.kill(parent.pid,signal.SIGCONT)
            assert parent.wait(timeout=5)==0
        finally:
            if parent.poll() is None:
                os.kill(parent.pid,signal.SIGCONT)
                parent.terminate();parent.wait(timeout=5)
print(json.dumps({'passed':True,'cases':[0,7],'child_runs_while_scheduler_stopped':True,'zombie_exit_status_verified':True}))
