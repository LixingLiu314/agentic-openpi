"""Stop only the eight identity-verified holders explicitly authorized by user."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import psutil

ROOT=Path('logs/pi05_native_n1_20260912/attempt_02')
EXTERNAL=Path('/media/raid/workspace/surongpeng/ws_liyan/RLinf')

def identity(record):
    proc=psutil.Process(record['pid'])
    assert abs(proc.create_time()-record['created'])<.02, 'PID reused'
    assert proc.cmdline()==record['command'], 'Changed command'
    assert Path(proc.cwd())==EXTERNAL, 'Wrong project'
    return proc

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    old=json.loads(Path('logs/pi05_native_n1_20260912/attempt_01/capacity_terminal.json').read_text())
    records=old['external_holders']
    assert len(records)==8 and {r['pid'] for r in records}==set(range(375186,375194))
    parent=identity(old['external_parent'])
    assert {p.pid for p in parent.children(recursive=True)}=={r['pid'] for r in records}, 'Unexpected other work under parent'
    sources={name:hashlib.sha256((EXTERNAL/name).read_bytes()).hexdigest() for name in
        ['toolkits/gpu_hold.py','toolkits/run_robotwin_hybrid_optimization_a800.sh']}
    driver=EXTERNAL/'logs/hybrid_optimization_a800_20260912/driver.log'
    last=driver.read_text().splitlines()[-1]
    assert 'FINISHED; per-arm summaries and selection.json contain outcomes' in last, last
    processes=[]
    for record in records:
        proc=identity(record)
        assert proc.ppid()==parent.pid
        assert record['command'][1:]==['toolkits/gpu_hold.py','--memory-fraction','0.70','--sleep-ms','20']
        assert not proc.children(), 'Unexpected holder child'
        processes.append(proc)
    ROOT.mkdir(parents=True,exist_ok=True)
    audit=dict(time=time.time(),source_hashes=sources,parent=old['external_parent'],holders=records,
        driver_terminal_line=last,only_holders_under_parent=True,no_training_code_in_reviewed_holder=True)
    before=ROOT/'holder_audit.json'
    if not args.apply:
        assert not before.exists()
        before.write_text(json.dumps(audit,indent=2)+'\n')
        print(json.dumps(dict(dry_run=True,**audit)));return
    previous=json.loads(before.read_text())
    assert previous['source_hashes']==sources and previous['holders']==records
    receipt=ROOT/'holder_stop.json';assert not receipt.exists()
    sent=[]
    for record in records:
        identity(record).send_signal(signal.SIGTERM)
        sent.append(record['pid'])
    gone,alive=psutil.wait_procs(processes,timeout=30)
    # Do not broaden the target set or signal parent/tmux/other tasks.
    assert not alive, 'Holder did not exit after SIGTERM; inspect before escalating'
    try:
        parent.wait(timeout=5)
        parent_exited=True
    except psutil.TimeoutExpired:parent_exited=False
    result=dict(passed=True,time=time.time(),sigterm_pids=sent,all_holders_exited=True,
        parent_naturally_exited=parent_exited,parent_signalled=False,guard_signalled=False,
        files_deleted=False,audit=audit)
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
