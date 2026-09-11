"""Unchanged observed training, recording the corrected startup orchestration."""
from pathlib import Path

import train_official_backbone_gradient_observed as trainer

original_config = trainer.build_run_config


def build_run_config(*args, **kwargs):
    config = original_config(*args, **kwargs)
    config['startup_checker_version'] = 2
    config['recovery_reason'] = 'previous full run stopped at16 by venv-launcher diagnostic bug; no saved checkpoint; fresh official initialization'
    for name in ['scripts/train_official_backbone_gradient_observed_v2.py',
                 'scripts/verify_official_full_wandb_startup_v2.py',
                 'scripts/recover_official_full_v2.py']:
        config['sources'][name] = trainer.sha256_file(Path(name))
    return config


if __name__ == '__main__':
    trainer.build_run_config = build_run_config
    trainer.train(trainer.parse_args())
