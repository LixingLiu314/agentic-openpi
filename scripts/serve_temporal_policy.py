"""Serve a verified R2 delta using explicit per-connection session metadata."""
import argparse
import os
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import torch
from openpi.policies.temporal_subtask_policy import create_temporal_policy
from openpi.serving.temporal_policy_server import TemporalPolicyServer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--parent-checkpoint',type=Path)
    p.add_argument('--device',default='cuda:0');p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8001)
    p.add_argument('--experimental',action='store_true',help='Explicitly load an unqualified research model; does not mark gates passed')
    p.add_argument('--allow-engineering',action='store_true')
    args=p.parse_args();torch.set_num_threads(4)
    policy=create_temporal_policy(args.checkpoint,device=args.device,parent_checkpoint=args.parent_checkpoint,
                                  allow_engineering=args.allow_engineering,require_candidate=not args.experimental)
    metadata=dict(policy.metadata,experimental=args.experimental,engineering_override=args.allow_engineering)
    TemporalPolicyServer(policy,args.host,args.port,metadata).serve_forever()


if __name__=='__main__':main()
