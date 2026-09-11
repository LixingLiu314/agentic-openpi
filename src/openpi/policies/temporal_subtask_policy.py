"""Native R2 inference with explicit per-run causal state and chunk acknowledgments."""
from collections import deque
import contextlib
import json
import math
from pathlib import Path

import jax
import numpy as np
import safetensors.torch
import torch

from openpi.models.model import Observation
from openpi.models_pytorch.subtask_decoder import SubtaskDecoderConfig
from openpi.models_pytorch.temporal_subtask import TemporalSubtask,TemporalConfig
from openpi.policies.subtask_policy import create_subtask_policy
from openpi.training.stage1_data import sha256_file
from openpi.training.transition_feature_cache import compress_memory
from openpi.training.transition_sequence import choose_text,select_observed_history


class SessionGuard:
    """Control metadata never becomes a label, stage index, or absolute model time."""
    def __init__(self):self.reset()
    def reset(self):
        self.run_id=None;self.sequence=None;self.time=None;self.prompt=None;self.chunk_id=None;self.mode=None
    def inspect(self,session,prompt):
        required={'run_id','sequence','observation_time','mode'}
        if not isinstance(session,dict) or not required<=session.keys():raise ValueError('R2 requires explicit session metadata')
        if session.keys()-required-{'previous_chunk'}:raise ValueError('Unknown session field; no external stage annotation is accepted')
        run=session['run_id'];seq=session['sequence'];stamp=session['observation_time'];mode=session['mode']
        if not isinstance(run,str) or not run or not isinstance(seq,int) or isinstance(seq,bool) or seq<0:raise ValueError('Invalid run/sequence')
        if not isinstance(stamp,(float,int)) or not math.isfinite(stamp):raise ValueError('Invalid observation time')
        if mode not in {'offline','execute'}:raise ValueError('Specify offline analysis or execute adoption semantics')
        new=run!=self.run_id
        if new and seq not in {0,1}:raise ValueError('New runs must start at sequence 0 or 1')
        if not new and (seq!=self.sequence+1 or stamp<=self.time):raise ValueError('Out-of-order or duplicate observation')
        if not new and mode!=self.mode:raise ValueError('Changing session mode requires a new run')
        reason='new_run' if new else 'task_changed' if prompt!=self.prompt else 'expired_history' if stamp-self.time>4 else None
        if not new and mode=='execute':
            ack=session.get('previous_chunk')
            if not isinstance(ack,dict) or ack.get('id')!=self.chunk_id:raise ValueError('Previous action chunk acknowledgment missing or mismatched')
            steps=ack.get('executed_steps')
            if not isinstance(steps,int) or isinstance(steps,bool) or not 0<=steps<=15:raise ValueError('Invalid published action count')
            if steps<15:reason='partial_or_rejected_chunk'
        return reason
    def commit(self,session,prompt,chunk_id):
        self.run_id=session['run_id'];self.sequence=session['sequence'];self.time=session['observation_time']
        self.prompt=prompt;self.mode=session['mode'];self.chunk_id=chunk_id


class TemporalPolicy:
    def __init__(self,base,temporal,*,stay_threshold,metadata):
        self.base=base;self.temporal=temporal;self.stay_threshold=stay_threshold;self._metadata=metadata
        self.guard=SessionGuard();self.history=deque(maxlen=512);self.active=''
    @property
    def metadata(self):return dict(self._metadata)
    def reset(self):self.guard.reset();self.history.clear();self.active=''
    def new_session(self):
        return TemporalPolicy(self.base,self.temporal,stay_threshold=self.stay_threshold,metadata=self._metadata)

    @torch.no_grad()
    def infer(self,obs,*,noise=None):
        forbidden={'action','actions','subtask','labels','target_ids','target_mask','subtask_target_ids','subtask_target_mask'}
        if forbidden&obs.keys():raise ValueError('Deployment observations must not contain subtask supervision')
        prompt=obs.get('prompt')
        if not isinstance(prompt,str) or not prompt.strip():raise ValueError('Global task is required')
        session=obs.get('session');reason=self.guard.inspect(session,prompt)
        if reason:self.history.clear();self.active=''
        begun=self.base._timestamp()
        clean={k:obs[k] for k in ['state','images','prompt']}
        transformed=self.base.input_transform(clean)
        tensors=jax.tree.map(lambda value:torch.as_tensor(np.asarray(value),device=self.base.device)[None],transformed)
        observation=Observation.from_dict(tensors)
        context=self.base.model.prepare_context(observation,[prompt])
        summary,mask=compress_memory(context.memory,context.memory_mask)
        current={'summary':summary[0],'mask':mask[0],'state':context.state[0],'time':session['observation_time']}
        received=list(self.history)
        selected=select_observed_history([r['time'] for r in received],current['time'])
        history=[received[i] if i is not None else None for i in selected]+[current]
        summaries=[];masks=[];states=[];ages=[];valid=[]
        for frame in history:
            ok=frame is not None and 0<=current['time']-frame['time']<=4
            summaries.append(frame['summary'] if ok else torch.zeros_like(current['summary']))
            masks.append(frame['mask'] if ok else torch.zeros_like(current['mask']))
            states.append(frame['state'] if ok else torch.zeros_like(current['state']))
            ages.append(current['time']-frame['time'] if ok else 0);valid.append(ok)
        ids,active_mask=self.base.model.codec.targets([self.active])
        device=self.base.device
        inputs={'memory':context.memory,'memory_mask':context.memory_mask,'summaries':torch.stack(summaries)[None],
                'summary_masks':torch.stack(masks)[None],'states':torch.stack(states)[None],
                'ages':torch.tensor([ages],dtype=torch.float32,device=device),'frame_valid':torch.tensor([valid],device=device),
                'active_ids':torch.tensor(ids,device=device),'active_mask':torch.tensor(active_mask,device=device)}
        amp=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
        with amp:generation,completion=self.temporal.generate(inputs,self.base.model.embedding_weight)
        texts,statuses=self.base.model.codec.decode(generation)
        probabilities=completion[0].float().cpu().tolist()
        text,adoption=choose_text(texts[0],statuses[0],self.active,probabilities,self.stay_threshold)
        prefix=self.base.model.action_prefix(context,[text])
        if noise is not None:
            noise=torch.as_tensor(noise,device=device,dtype=torch.float32)
            if noise.ndim==2:noise=noise[None]
            if noise.shape!=(1,50,32) or not torch.isfinite(noise).all():raise ValueError('Expected finite [50,32] flow noise')
        actions=self.base.model.sample_actions_from_prefix(context,prefix,noise=noise,num_steps=self.base.num_steps)
        result=self.base.output_transform({'state':tensors['state'][0].cpu().numpy(),'actions':actions[0].float().cpu().numpy()})
        chunk_id=f"{session['run_id']}:{session['sequence']}"
        result.update(subtask=text,raw_subtask_candidate=texts[0],subtask_status=statuses[0],
                      subtask_score=float(generation.mean_log_probability[0]),completion_proxy_probabilities=probabilities,
                      physical_completion_certified=False,adoption_reason=adoption,reset_reason=reason,
                      chunk_id=chunk_id,expected_execution_steps=15,policy_timing={'infer_ms':(self.base._timestamp()-begun)*1000})
        self.history.append(current)
        while self.history and current['time']-self.history[0]['time']>4:self.history.popleft()
        self.active=text;self.guard.commit(session,prompt,chunk_id)
        return result


def create_temporal_policy(checkpoint,*,device='cpu',parent_checkpoint=None,allow_engineering=False,require_candidate=True):
    checkpoint=Path(checkpoint);meta=json.loads((checkpoint/'metadata.json').read_text())
    if meta['schema_version']!=2 or meta['variant']!='r2_temporal_completion_v1':raise ValueError('Not an R2 checkpoint')
    if meta['engineering_smoke'] and not allow_engineering:raise ValueError('Engineering checkpoint cannot be deployed as research')
    if sha256_file(checkpoint/'temporal.safetensors')!=meta['temporal_sha256']:raise ValueError('Temporal weights checksum mismatch')
    protocol_path=checkpoint.parent/'protocol.json'
    if sha256_file(protocol_path)!=meta['config']['protocol_sha256']:raise ValueError('Checkpoint protocol changed')
    protocol=json.loads(protocol_path.read_text());parent=Path(parent_checkpoint or protocol['parent_checkpoint'])
    if sha256_file(parent/'model.safetensors')!=meta['config']['parent_weights_sha256']:raise ValueError('M3 parent changed')
    if require_candidate:
        candidate=json.loads((checkpoint.parent/'candidate.json').read_text())
        if (candidate.get('status')!='qualified_offline_candidate' or candidate['checkpoint']!=checkpoint.name
                or candidate.get('temporal_sha256')!=meta['temporal_sha256']
                or not all(candidate.get(k) for k in ['semantic_gates_passed','action_gate_passed','policy_load_gate_passed','stress_gate_passed'])):
            raise ValueError('This temporal checkpoint has not passed candidate gates')
    base=create_subtask_policy(parent,device=device)
    if torch.device(device).type=='cpu':base.model.float()
    temporal=TemporalSubtask(SubtaskDecoderConfig(**meta['config']['decoder_config']),TemporalConfig(**meta['config']['temporal_config'])).to(device).eval()
    safetensors.torch.load_model(temporal,checkpoint/'temporal.safetensors',strict=True)
    threshold=(meta.get('best') or {}).get('stay_threshold',1.01)
    metadata={**base.metadata,'stage':'r2','variant':'r2_temporal_completion_v1','checkpoint':str(checkpoint.resolve()),
              'parent_checkpoint':str(parent.resolve()),'temporal_sha256':meta['temporal_sha256'],
              'session_protocol':1,'required_session_fields':['run_id','sequence','observation_time','mode'],
              'completion_is_annotation_proxy':True,'physical_readiness_reviewed':False}
    return TemporalPolicy(base,temporal,stay_threshold=threshold,metadata=metadata)
