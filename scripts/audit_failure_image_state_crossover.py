"""CPU text-only 2x2 image/state crossover for recorded queries70 and94."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import shutil
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('HF_HUB_OFFLINE','1')
import jax
import numpy as np
import safetensors.torch
import torch
from openpi.models.model import Observation
from openpi.models_pytorch.subtask_decoder import SubtaskDecoder
from openpi.policies.subtask_policy import create_subtask_policy
from openpi.training.stage1_data import sha256_file


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(42)
    replay=Path('assets/pi05_piper_transition/eggplant_potato/r2_failure_replay_v1')
    protocol=json.loads((replay/'protocol.json').read_text())
    parent=Path('checkpoints/pi05_piper_stage1/m3_pilot_seed42/step_003500');delta=Path('checkpoints/pi05_piper_transition/r1_seed42/step_001000')
    meta=json.loads((delta/'metadata.json').read_text())
    assert sha256_file(parent/'model.safetensors')==meta['config']['parent_weights_sha256']
    assert sha256_file(delta/'decoder.safetensors')==meta['decoder_sha256']
    assert meta['frozen_base_equal'] and meta['base_tensor_hash']==meta['initial_base_tensor_hash']
    policy=create_subtask_policy(parent,device='cpu');policy.model.float().eval()
    r1=SubtaskDecoder(policy.model.decoder.config).float().eval();safetensors.torch.load_model(r1,delta/'decoder.safetensors',strict=True)
    contexts={};prompts={};states={};rows=[]
    for r in protocol['records']:
        if r['query'] not in [70,94]:continue
        source=replay/r['file'];assert sha256_file(source)==r['sha256']
        with np.load(source) as saved:
            obs={'state':saved['state'].copy(),'images':{n:saved[n].copy() for n in ['cam_high','cam_left_wrist','cam_right_wrist']},'prompt':r['prompt']}
        states[r['query']]=obs['state'].tolist();prompts[r['query']]=r['prompt']
        tensors=jax.tree.map(lambda x:torch.as_tensor(np.asarray(x))[None],policy.input_transform(obs))
        contexts[r['query']]=policy.model.prepare_context(Observation.from_dict(tensors),[r['prompt']])
    assert prompts[70]==prompts[94]
    for image_query in [70,94]:
        for state_query in [70,94]:
            context=contexts[image_query]
            if image_query!=state_query:
                state=contexts[state_query].state
                ids,masks=policy.model._tokens(context.global_prompts,state)
                memory,memory_mask,_=policy.model._prefix(context.image_features,context.image_masks,ids,masks,use_cache=False)
                context=dataclasses.replace(context,state=state,memory=memory,memory_mask=memory_mask)
            outputs={}
            for name,decoder in [('m3',policy.model.decoder),('r1',r1)]:
                generated=decoder.generate(context.memory,context.memory_mask,policy.model.embedding_weight)
                text,status=policy.model.codec.decode(generated)
                outputs[name]={'text':text[0],'status':status[0],'mean_token_log_probability':float(generated.mean_log_probability[0])}
            row={'image_query':image_query,'state_query':state_query,'outputs':outputs};rows.append(row);print(json.dumps(row),flush=True)
    result={'scope':'CPU FP32 AR text only; same global prompt; 2x2 crossover of all three RGB views and full proprioceptive state',
            'rows':rows,'original_raw_states':states,'parent_sha256':meta['config']['parent_weights_sha256'],'r1_sha256':meta['decoder_sha256'],
            'source_sha256':sha256_file(Path(__file__)),'replay_protocol_sha256':sha256_file(replay/'protocol.json'),
            'limitations':'Approximate lossy reconstructed RGB; crossed image/state pairs are physically inconsistent diagnostic inputs. No hardware, training, or physical-cause claim.'}
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');shutil.copy2(__file__,a.output/Path(__file__).name)


if __name__=='__main__':main()
