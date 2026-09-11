"""Real cached M3 observations: original-map parity, S/T/D update and exact resume."""
import argparse
import copy
import dataclasses
import json
import os
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('HF_HUB_OFFLINE','1')

import safetensors
import torch
from openpi.models.subtask_tokenizer import SubtaskTextCodec
from openpi.models_pytorch.temporal_subtask import TemporalSubtask,TemporalConfig
from openpi.models_pytorch.subtask_decoder import SubtaskDecoderConfig
from openpi.training.transition_feature_cache import EpisodeFeatures
from openpi.training.transition_sequence import assemble_inputs,batch_inputs


def load_parts(parent):
    with safetensors.safe_open(str(parent/'model.safetensors'),framework='pt',device='cpu') as file:
        state={k.removeprefix('decoder.'):file.get_tensor(k).float() for k in file.keys() if k.startswith('decoder.')}
        expected='base.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight'
        key=(file.metadata() or {}).get(expected,expected)
        if key not in file.keys():raise ValueError('Expected frozen vocabulary or its recorded tied-weight alias')
        embedding=file.get_tensor(key).float()
    return state,embedding


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    protocol=json.loads(Path('assets/pi05_piper_transition/eggplant_potato/r2_v1/protocol.json').read_text())
    parent=Path(protocol['parent_checkpoint'])
    metadata=json.loads((parent/'metadata.json').read_text())
    state,embedding=load_parts(parent)
    model=TemporalSubtask();model.decoder.load_state_dict(state,strict=True)
    codec=SubtaskTextCodec()
    ep=EpisodeFeatures(args.cache/'train/episode_000000')
    active=ep.rows[0]['prediction']
    inputs=batch_inputs([assemble_inputs(ep,0,[],'',codec),assemble_inputs(ep,1,[0],active,codec)])
    generation,prob=model.generate(inputs,embedding)
    original=model.decoder.generate(inputs['memory'],inputs['memory_mask'],embedding)
    assert torch.equal(generation.token_ids,original.token_ids)
    assert torch.equal(generation.token_mask,original.token_mask)
    target_ids,target_mask=codec.targets([r['label'] for r in ep.rows[:2]])
    ids=torch.tensor(target_ids);mask=torch.tensor(target_mask);done=torch.tensor([-100,0])
    opt=lambda m:torch.optim.AdamW(m.parameters(),lr=2.5e-5,betas=(.9,.95),weight_decay=1e-10,foreach=False)
    optimizer=opt(model)
    def step(m,o):
        o.zero_grad(set_to_none=True)
        result=m(inputs,ids,mask,done,embedding)
        result['loss'].backward()
        assert m.temporal[0].self_attn.in_proj_weight.grad.abs().sum()>0
        assert m.completion[-1].weight.grad.abs().sum()>0
        assert embedding.grad is None
        torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);o.step()
        return float(result['loss'])
    args.output.mkdir(parents=True,exist_ok=False)
    losses=[step(model,optimizer)]
    checkpoint=args.output/'step1.pt'
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state()},checkpoint)
    losses.append(step(model,optimizer))
    resumed=TemporalSubtask();saved=torch.load(checkpoint,weights_only=False)
    resumed.load_state_dict(saved['model']);resumed_optimizer=opt(resumed);resumed_optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['rng']);resumed_loss=step(resumed,resumed_optimizer)
    equal=all(torch.equal(value,resumed.state_dict()[name]) for name,value in model.state_dict().items())
    assert equal,'Uninterrupted and saved/resumed temporal model differ'
    result={'passed':True,'scope':'CPU engineering; not model quality','real_frames':2,'initial_M3_text_equal':True,
            'S_T_D_gradients':True,'frozen_embedding_gradient_absent':True,'exact_resume_equal':equal,
            'losses':losses,'resumed_loss':resumed_loss,'seed':42}
    (args.output/'gate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
