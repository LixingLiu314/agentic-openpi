"""Real CPU toy gradients uploaded outside the research comparison display set."""
import json
from pathlib import Path

import torch
import wandb

from test_backbone_observability import ToyModel
from openpi.training.backbone_observability import GradientObserver, configure_run, log_event, event_payload

torch.manual_seed(42)
root=Path('logs/pi05_official_gradient_20260909_wandb_fix')
config={'steps':10,'observability_version':2,'display_schema':1,
        'display_set':'backbone-observability-engineering-20260909','scope':'CPU toy upload check, not a research experiment'}
run=wandb.init(entity='xiahy23-tsinghua-university',project='agentic-openpi-pi05-subtask',
    id='backbone-observability-v2-smoke-20260909',name='Engineering backbone metrics upload',
    job_type='engineering',group=config['display_set'],config=config,dir=str(root),
    settings=wandb.Settings(x_disable_stats=True,disable_git=True,save_code=False))
configure_run(run)
model=ToyModel()
subtask=torch.nn.Linear(3,2)
x=torch.randn(5,3)
action_loss=model.loss(x)
subtask_ce=torch.nn.functional.cross_entropy(subtask(x),torch.tensor([0,1,0,1,0]))
(action_loss+subtask_ce).backward()
norms=GradientObserver(model).measure()
row={'event':'train','step':10,'loss_action':float(action_loss.detach()),'loss_subtask':float(subtask_ce.detach()),'grad_norms':norms}
log_event(run,row,config)
expected=event_payload(row,config)
(root/'smoke_expected.json').write_text(json.dumps(expected,indent=2))
run.finish()
print(json.dumps({'uploaded':True,'expected':expected}))
