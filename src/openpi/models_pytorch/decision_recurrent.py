"""Target/actor decisions and executed-prefix training; optional S-only grounding.

B/A and the discrete subtask-to-action interface keep their existing structure.
All targets, phase weights and contrastive alternatives remain training-only.
"""
import dataclasses
import math
import re

import torch
from torch import nn
import torch.nn.functional as F

from openpi.models_pytorch.recurrent_subtask import RecurrentSubtaskDecoder
from openpi.models_pytorch.semantic_recurrent import SemanticRecurrentModel, semantic_rules, select_context

VARIANT="official_pi05_recurrent_decision_v1"
SCHEMA_VERSION=9
DISPLAY_SET="official-pi05-decision-grounding-pair-s42-v1"
EXPERIMENTS=("decision_prefix","decision_grounded")


def condition_rules(vocabulary, prompts):
    """Existing train vocabulary determines objects; no deployment task rules."""
    vocab,goals=set(vocabulary),set(prompts)
    parsed={label:re.fullmatch(r"reach the (.+) with the (left|right) arm",label) for label in vocab}
    entities=sorted({m.group(1) for m in parsed.values() if m and any(m.group(1) in goal for goal in goals)})
    rules={}
    for label in sorted(vocab):
        arm=[];objects=[];m=parsed[label]
        if m:
            other=label.rsplit(" with the ",1)[0]+" with the "+("right" if m.group(2)=="left" else "left")+" arm"
            if other in vocab:arm.append(other)
            entity=m.group(1)
            if entity in entities:
                for alternate in entities:
                    candidate="reach the "+alternate+" with the "+m.group(2)+" arm"
                    if alternate!=entity and candidate in vocab:
                        objects.append(dict(label=candidate,source_entity=entity,target_entity=alternate))
        rules[label]=dict(arm=arm,object=objects)
    return rules,entities


def decision_pairs(labels, prompts, generated, statuses, dropped, stable, rules, valid_prompts, *, offset=0, per_type=2):
    if stable is None:raise ValueError("Ranking requires the training-only stable-prefix mask")
    selected=[]
    for kind in ("arm","object"):
        count=0
        for n in range(len(labels)):
            i=(n+offset)%len(labels)
            if not stable[i] or dropped[i] or statuses[i]!="ok" or generated[i]!=labels[i]:continue
            choices=rules.get(labels[i],{}).get(kind,[])
            if not choices:continue
            choice=choices[(offset+i)%len(choices)]
            if kind=="arm":alternative,goal=choice,prompts[i]
            else:
                if choice["source_entity"] not in prompts[i]:continue
                alternative=choice["label"]
                goal=prompts[i].replace(choice["source_entity"],choice["target_entity"])
                if goal not in valid_prompts:continue
            selected.append(dict(index=i,label=alternative,prompt=goal,kind=kind));count+=1
            if count==per_type:break
    return selected


def prefix_weighted_flow(errors):
    weights=errors.new_ones((1,errors.shape[1],1))
    weights[:,:25]=2.0
    return (errors*weights/weights.mean()).mean()


class GroundedDecoder(RecurrentSubtaskDecoder):
    """A task-conditioned spatial query over the final B features, inside S.

    The target feature is an extra S context token. It never enters A as a hidden
    state; A continues to receive only the generated ordinary subtask text.
    """
    def __init__(self,config=None,*,recurrent=True,memory_tokens=4):
        super().__init__(config,recurrent=recurrent,memory_tokens=memory_tokens)
        w=self.config.width
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4201)
            self.ground_query=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,w,bias=False))
            self.ground_key=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,w,bias=False))
            self.ground_position=nn.Linear(2,w,bias=False)
            self.ground_type=nn.Parameter(torch.randn(1,1,w)*.02)
        y,x=torch.meshgrid((torch.arange(16)+.5)/16,(torch.arange(16)+.5)/16,indexing="ij")
        self.register_buffer("ground_grid",torch.stack((x,y),-1).reshape(256,2),persistent=False)

    def localize_projected(self,projected,mask):
        if projected.shape[1]<769:
            raise ValueError("Grounding expects three RGB224 cameras followed by task/state tokens")
        language_mask=mask[:,768:].to(projected.dtype)
        task=(projected[:,768:]*language_mask[:,:,None]).sum(1)/language_mask.sum(1,keepdim=True).clamp_min(1)
        keys=self.ground_key(projected[:,:256])
        scores=torch.einsum("bd,bnd->bn",self.ground_query(task),keys).float()/math.sqrt(keys.shape[-1])
        scores=scores.masked_fill(~mask[:,:256].bool(),-1e9)
        probabilities=scores.softmax(-1)
        point=probabilities@self.ground_grid.float()
        selected=torch.einsum("bn,bnd->bd",probabilities.to(projected.dtype),projected[:,:256])
        target_token=selected+self.ground_position(point.to(projected.dtype))
        return scores,point,target_token[:,None]+self.ground_type

    def compose_sequence(self,memory,mask,carry=None,resets=None,*,unroll=1):
        projected,full_mask,next_carry=super().compose_sequence(memory,mask,carry,resets,unroll=unroll)
        # Exclude recurrent tokens from the language query used for localization.
        scores,point,target=self.localize_projected(projected[:,:memory.shape[1]],mask)
        return torch.cat((projected,target),1),torch.cat((full_mask,torch.ones_like(full_mask[:,:1])),1),next_carry

    def localization_loss(self,memory,mask,points):
        if points.shape!=(memory.shape[0],2) or not torch.isfinite(points).all() or (points<0).any() or (points>1).any():
            raise ValueError("Invalid localization targets")
        projected=self.memory_projection(memory.detach().to(self.memory_projection.weight.dtype))
        scores,predicted,_=self.localize_projected(projected,mask)
        delta=(self.ground_grid[None].float()-points[:,None].float())/(1.0/16)
        target=(-.5*delta.square().sum(-1)).softmax(-1)
        loss=-(target*scores.log_softmax(-1)).sum(-1).mean()
        distance=torch.linalg.vector_norm(predicted-points,dim=-1).mean()*224
        return loss,distance.detach(),scores


class DecisionRecurrentModel(SemanticRecurrentModel):
    def __init__(self,base,decoder_config=None,*,grounded=False,recurrent=True,seed=42,unroll=4):
        super().__init__(base,decoder_config,recurrent=recurrent,seed=seed,unroll=unroll)
        self.grounded=grounded
        if grounded:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                decoder=GroundedDecoder(decoder_config,recurrent=recurrent)
            self.decoder=decoder.to(base.action_in_proj.weight.device)

    def configure_decisions(self,vocabulary,prompts,experiment):
        if experiment not in EXPERIMENTS or self.grounded!=(experiment=="decision_grounded"):
            raise ValueError("Decision experiment and S architecture disagree")
        self.experiment=experiment
        self.semantic_weights,_,self.semantic_audit=semantic_rules(self.codec,vocabulary)
        self.decision_rules,self.entities=condition_rules(vocabulary,prompts)
        self.valid_prompts=set(prompts)

    def decision_s_loss(self,ids,valid,projected,mask,row_weights):
        decoder=self.decoder
        inputs=torch.full_like(ids,decoder.config.pad_id);inputs[:,0]=decoder.config.bos_id
        inputs[:,1:]=torch.where(valid[:,:-1],ids[:,:-1],decoder.config.pad_id)
        logits=decoder.logits_projected(inputs,projected,mask,self.embedding_weight)
        targets=torch.where(valid,ids,decoder.config.pad_id)
        losses=F.cross_entropy(logits.float().flatten(0,1),targets.flatten(),reduction="none").view_as(ids)
        weights=torch.tensor([self.semantic_weights[tuple(r)] for r in ids.detach().cpu().tolist()],device=ids.device)*valid
        rows=(losses*weights).sum(1)/weights.sum(1).clamp_min(1)
        row_weights=torch.as_tensor(row_weights,device=ids.device,dtype=rows.dtype)
        if row_weights.shape!=rows.shape or (row_weights<=0).any():raise ValueError("Invalid decision weights")
        ordinary=((losses*valid).sum(1)/valid.sum(1).clamp_min(1)).mean()
        return (rows*row_weights).sum()/row_weights.sum(),ordinary.detach()

    def forward(self,batch,*,carry=None,drop_condition=None,rank_offset=0,rank_stable_mask=None,
                decision_weights=None,ground_batch=None,ground_points=None):
        context=self.prepare_context(batch.observation,batch.global_prompts)
        projected,mask,next_carry=self.decoder.compose_sequence(context.memory,context.memory_mask,carry,
                                                               batch.reset_mask,unroll=self.unroll)
        if decision_weights is None:decision_weights=[1.0]*len(batch.labels)
        semantic,ordinary=self.decision_s_loss(batch.target_ids,batch.target_mask,projected,mask,decision_weights)
        generation=self.decoder.generate_projected(projected.detach(),mask,self.embedding_weight)
        texts,statuses=self.codec.decode(generation)
        dropped=[False]*len(texts) if drop_condition is None else drop_condition
        conditions=["" if drop else text for text,drop in zip(texts,dropped,strict=True)]
        noise=self.base.sample_noise(batch.actions.shape,batch.actions.device)
        time=self.base.sample_time(batch.actions.shape[0],batch.actions.device)
        errors=self.action_errors(context,conditions,batch.actions,noise=noise,time=time)
        action=prefix_weighted_flow(errors)
        pairs=decision_pairs(batch.labels,batch.global_prompts,texts,statuses,dropped,rank_stable_mask,
                             self.decision_rules,self.valid_prompts,offset=rank_offset)
        rank_losses={kind:action.new_zeros(()) for kind in ("arm","object")}
        counts={kind:sum(p["kind"]==kind for p in pairs) for kind in rank_losses}
        if pairs:
            indices=[p["index"] for p in pairs]
            negative_context=select_context(context,indices)
            negative_context=dataclasses.replace(negative_context,global_prompts=tuple(p["prompt"] for p in pairs))
            negative=self.action_errors(negative_context,[p["label"] for p in pairs],batch.actions[indices],
                                        noise=noise[indices],time=time[indices])
            hinge=F.relu(.01+errors[indices,:25,:14].mean((1,2))-negative[:,:25,:14].mean((1,2)))
            for kind in rank_losses:
                selected=[i for i,p in enumerate(pairs) if p["kind"]==kind]
                if selected:rank_losses[kind]=hinge[selected].mean()
        ranking=sum(rank_losses.values())/max(1,sum(v>0 for v in counts.values()))
        grounding=action.new_zeros(());ground_distance=action.new_zeros(())
        if ground_batch is not None:
            if not self.grounded or ground_points is None:raise ValueError("Unexpected grounding batch")
            ground_context=self.prepare_context(ground_batch.observation,ground_batch.global_prompts)
            grounding,ground_distance,_=self.decoder.localization_loss(ground_context.memory,ground_context.memory_mask,ground_points)
        return dict(loss_subtask=semantic+.05*grounding,loss_action=action+.1*ranking,
            loss_action_main=errors.mean().detach(),loss_action_weighted=action.detach(),
            loss_subtask_unweighted=ordinary,loss_semantic=semantic.detach(),loss_action_rank=ranking.detach(),
            rank_pairs=len(pairs),rank_arm_pairs=counts["arm"],rank_object_pairs=counts["object"],
            rank_arm_hinge=rank_losses["arm"].detach(),rank_object_hinge=rank_losses["object"].detach(),
            grounding_ce=grounding.detach(),grounding_distance_px=ground_distance,
            first25_native_mse=errors[:,:25,:14].mean().detach(),
            carry=None if next_carry is None else next_carry.detach(),generated_count=len(texts),
            invalid_generation_count=sum(x!="ok" for x in statuses),empty_condition_count=sum(not x for x in conditions))
