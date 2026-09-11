"""Run the authorized pair, then validation-only condition/memory diagnostics."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gpu_reservation import run_concurrent


def main():
    root = Path('logs/pi05_recurrent_20260909')
    cache = Path('.stage1_staging/piper_rgb224_cache_v2').absolute()
    training = [sys.executable, 'scripts/run_recurrent_pair.py', '--root', str(root), '--cache', str(cache),
                '--stage', 'formal', '--steps', '4000']
    subprocess.run(training, check=True)
    for arm in ['stateless', 'recurrent']:
        candidate = json.loads((root / (arm + '_complete.json')).read_text())
        output = root / (arm + '_condition_memory_diagnostics.json')
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
                   'scripts/evaluate_recurrent_candidate.py', '--checkpoint', candidate['checkpoint'],
                   '--output', str(output), '--cache', str(cache)]
        (root / 'diagnostics_state.json').write_text(json.dumps(dict(phase='running', arm=arm, time=time.time())))
        code = run_concurrent(Path('logs/pi05_subtask_stage1/gpu_reservation'),
                              root / (arm + '_diagnostics.log'), command)
        if code:
            (root / 'diagnostics_failure.json').write_text(json.dumps(dict(arm=arm, code=code, time=time.time())))
            raise SystemExit(code)
    (root / 'diagnostics_state.json').write_text(json.dumps(dict(phase='complete', time=time.time())))


if __name__ == '__main__':
    main()
