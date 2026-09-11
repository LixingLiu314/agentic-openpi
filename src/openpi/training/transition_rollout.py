"""Independent, empty-state, causal R2 rollout at each observation cadence."""
from collections import deque
import contextlib

import numpy as np
import torch

from openpi.training.transition_sequence import assemble_inputs,batch_inputs,choose_text,select_observed_history
from openpi.training.transition_metrics_v2 import causal_sample


@torch.no_grad()
def rollout(model,embedding,episodes,codec,device,*,period=None,phase=0.,stay_threshold=1.01,batch_size=8):
    previous=model.training;model.eval()
    result=[]
    try:
        keys=sorted(episodes)
        for chunk in range(0,len(keys),batch_size):
            states=[]
            for ep in keys[chunk:chunk+batch_size]:
                episode=episodes[ep]
                selected=episode.rows if period is None else causal_sample(episode.rows,period,phase)
                states.append({'episode':episode,'indices':[r['frame'] for r in selected],'cursor':0,'history':deque(maxlen=512),'active':''})
            while True:
                alive=[s for s in states if s['cursor']<len(s['indices'])]
                if not alive:break
                examples=[]
                for state in alive:
                    ep=state['episode'];index=state['indices'][state['cursor']]
                    past=list(state['history'])
                    positions=select_observed_history([ep.times[i] for i in past],ep.times[index])
                    history=[past[i] if i is not None else None for i in positions]
                    examples.append(assemble_inputs(ep,index,history,state['active'],codec))
                inputs=batch_inputs(examples,device)
                amp=torch.autocast('cuda',dtype=torch.bfloat16) if torch.device(device).type=='cuda' else contextlib.nullcontext()
                with amp:generation,probability=model.generate(inputs,embedding)
                texts,statuses=codec.decode(generation)
                probs=probability.float().cpu().tolist()
                scores=generation.mean_log_probability.float().cpu().tolist()
                for state,text,status,prob,score in zip(alive,texts,statuses,probs,scores,strict=True):
                    index=state['indices'][state['cursor']]
                    chosen,reason=choose_text(text,status,state['active'],prob,stay_threshold)
                    result.append(dict(state['episode'].rows[index],prediction=chosen,status=status,raw_candidate=text,
                                       previous_active=state['active'],completion_probabilities=prob,adoption_reason=reason,
                                       token_log_probability=score,rollout_period=period,rollout_phase=phase))
                    state['active']=chosen;state['history'].append(index);state['cursor']+=1
        return sorted(result,key=lambda r:(r['episode'],r['frame']))
    finally:model.train(previous)


def previous_rollout_text(predictions,episode,frame):
    """A stale own-model proposal may condition a later sample, never a future one."""
    rows=predictions.get(episode,[])
    if not rows:return None
    index=np.searchsorted([r['frame'] for r in rows],frame,side='left')-1
    return rows[int(index)]['prediction'] if index>=0 else ''
