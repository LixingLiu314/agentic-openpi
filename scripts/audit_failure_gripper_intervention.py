"""CPU-only text sensitivity on failed-trial crops; no policy actions or controls.

Counterfactual width edits deliberately break image/state consistency. They test
model sensitivity, and are not synthetic success labels or deployable corrections.
"""
import argparse
import dataclasses
import hashlib
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
from openpi.models_pytorch.subtask_decoder import SubtaskDecoder,SubtaskDecoderConfig
from openpi.policies.subtask_policy import create_subtask_policy
from openpi.training.stage1_data import sha256_file


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(42)
    replay=Path('assets/pi05_piper_transition/eggplant_potato/r2_failure_replay_v1')
    protocol=json.loads((replay/'protocol.json').read_text())
    parent=Path('checkpoints/pi05_piper_stage1/m3_pilot_seed42/step_003500')
    delta=Path('checkpoints/pi05_piper_transition/r1_seed42/step_001000')
    meta=json.loads((delta/'metadata.json').read_text())
    assert sha256_file(parent/'model.safetensors')==meta['config']['parent_weights_sha256']
    assert sha256_file(delta/'decoder.safetensors')==meta['decoder_sha256']
    assert meta['frozen_base_equal'] and meta['base_tensor_hash']==meta['initial_base_tensor_hash']
    policy=create_subtask_policy(parent,device='cpu');policy.model.float().eval()
    r1=SubtaskDecoder(SubtaskDecoderConfig(**dataclasses.asdict(policy.model.decoder.config))).float().eval()
    safetensors.torch.load_model(r1,delta/'decoder.safetensors',strict=True)
    records=[]
    for record in protocol['records']:
        if record['query'] not in [70,94,95]:continue
        source=replay/record['file'];assert sha256_file(source)==record['sha256']
        with np.load(source) as saved:
            original={'state':saved['state'].copy(),'images':{name:saved[name].copy() for name in ['cam_high','cam_left_wrist','cam_right_wrist']},'prompt':record['prompt']}
        images_digest={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in original['images'].items()}
        transformed=policy.input_transform(original)
        tensors=jax.tree.map(lambda value:torch.as_tensor(np.asarray(value))[None],transformed)
        context=policy.model.prepare_context(Observation.from_dict(tensors),[original['prompt']])
        for name,width in [('recorded',float(original['state'][13])),('fit_handle_terminal_median',.008750000037252903),
                           ('open_reference',.06965),('zero_reference',0.)]:
            if name=='recorded':modified=context
            else:
                state=original['state'].copy();state[13]=width
                changed=policy.input_transform(dict(original,state=state))
                normalized=torch.as_tensor(np.asarray(changed['state']))[None]
                ids,masks=policy.model._tokens([original['prompt']],normalized)
                memory,memory_mask,_=policy.model._prefix(context.image_features,context.image_masks,ids,masks,use_cache=False)
                modified=dataclasses.replace(context,state=normalized,memory=memory,memory_mask=memory_mask)
            outputs={}
            for model_name,decoder in [('m3',policy.model.decoder),('r1',r1)]:
                generated=decoder.generate(modified.memory,modified.memory_mask,policy.model.embedding_weight)
                texts,statuses=policy.model.codec.decode(generated)
                outputs[model_name]={'text':texts[0],'status':statuses[0],'mean_token_log_probability':float(generated.mean_log_probability[0])}
            row={'query':record['query'],'condition':name,'right_width':width,'outputs':outputs}
            records.append(row);print(json.dumps(row),flush=True)
        assert all(hashlib.sha256(v.tobytes()).hexdigest()==images_digest[k] for k,v in original['images'].items())
    result={'scope':'Approximate reconstructed images held fixed; only raw right gripper state changed; CPU FP32 autoregressive text only',
            'seed':42,'rows':records,'parent_sha256':meta['config']['parent_weights_sha256'],'r1_sha256':meta['decoder_sha256'],
            'source_sha256':sha256_file(Path(__file__)),'replay_protocol_sha256':sha256_file(replay/'protocol.json'),
            'physical_causality_claimed':False,'physical_completion_reviewed':False,
            'limitations':'Image/state counterfactuals are physically inconsistent. Sensitivity is evidence about the model mapping, not the physical cause of failure.'}
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');shutil.copy2(__file__,a.output/Path(__file__).name)


if __name__=='__main__':main()
