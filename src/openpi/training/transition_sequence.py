"""Build R2 inputs from cached past/current observations, not label histories."""
import numpy as np
import torch


def select_observed_history(times,current_time,period=.764,count=3,max_age=4.,clock_slack=.08):
    """Select received past observations near fixed lags at both dense and low rates.

    The small slack tolerates video/decision quantization; selected observations
    always strictly precede current_time. Insufficient older history stays masked.
    Returned indices address the supplied history list, oldest slot first.
    """
    if period<=0 or count<1:raise ValueError('Invalid history sampling interval')
    available=[i for i,t in enumerate(times) if 0<current_time-float(t)<=max_age]
    chosen=[];latest=current_time
    for lag in period*np.arange(1,count+1):
        candidates=[i for i in available if float(times[i])<latest and current_time-float(times[i])>=lag-clock_slack]
        if not candidates:chosen.append(None);continue
        index=min(candidates,key=lambda i:(abs(current_time-float(times[i])-lag),float(times[i])))
        chosen.append(index);latest=float(times[index])
    return list(reversed(chosen))


def assemble_inputs(episode,index,history,active_text,codec,max_age=4.):
    current=episode.current(index)
    history=list(history)[-3:]
    history=[None]*(3-len(history))+history
    summaries=[];masks=[];states=[];ages=[];valid=[]
    for slot,item in enumerate([*history,index]):
        age=float(episode.times[index]-episode.times[item]) if item is not None else 0.
        if item is not None and slot<3 and age<=0:raise ValueError('Noncausal history index')
        usable=item is not None and 0<=age<=max_age
        if usable:
            # Past calls use only the compact summary, avoiding full-prefix reads.
            from openpi.training.transition_feature_cache import decode_array
            summaries.append(decode_array(episode.summary[item],episode.dtype))
            masks.append(torch.tensor(episode.summary_mask[item]))
            states.append(torch.tensor(episode.state[item]))
        else:
            summaries.append(torch.zeros_like(current['summary']))
            masks.append(torch.zeros_like(current['summary_mask']))
            states.append(torch.zeros_like(current['state']))
        ages.append(age if usable else 0.);valid.append(usable)
    ids,mask=codec.targets([active_text])
    return {'memory':current['memory'],'memory_mask':current['memory_mask'],
            'summaries':torch.stack(summaries),'summary_masks':torch.stack(masks),'states':torch.stack(states),
            'ages':torch.tensor(ages,dtype=torch.float32),'frame_valid':torch.tensor(valid,dtype=torch.bool),
            'active_ids':torch.tensor(ids[0]),'active_mask':torch.tensor(mask[0])}


def batch_inputs(examples,device='cpu'):
    return {k:torch.stack([x[k] for x in examples]).to(device) for k in examples[0]}


def completion_target(rows,index,active_text,uncertainty_frames=3):
    """Annotation proxy with persistent positives; not physical ready supervision."""
    if not active_text:return -100
    label=rows[index]['label']
    changes=[i for i in range(1,len(rows)) if rows[i]['label']!=rows[i-1]['label']]
    if changes and min(abs(index-i) for i in changes)<=uncertainty_frames:return -100
    if active_text==label:return 0
    previous=[i for i in changes if i<=index]
    if previous and active_text==rows[previous[-1]-1]['label']:return 1
    return 2


def choose_text(candidate,status,active,completion_prob,stay_threshold):
    """A calibrated keep decision; no hard-coded stage ordering or timeout advance."""
    if status!='ok':return '', 'invalid_generation'
    if active and float(completion_prob[0])>=stay_threshold:return active,'keep_active'
    return candidate,'generated_successor_or_reidentification'
