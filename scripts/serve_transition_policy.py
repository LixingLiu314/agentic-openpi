"""Serve an R1 candidate with unchanged native14 observation/action contract."""
import argparse
import torch
from openpi.policies.subtask_transition_policy import create_transition_policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--host",default="0.0.0.0")
    parser.add_argument("--port",type=int,default=8000)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    policy=create_transition_policy(args.checkpoint,device="cuda:0",parent_checkpoint=args.parent_checkpoint)
    WebsocketPolicyServer(policy,host=args.host,port=args.port,metadata=policy.metadata).serve_forever()

if __name__=="__main__":
    main()
