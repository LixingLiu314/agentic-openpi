"""Parallel S/A heads: observation CE trains B; every A-to-B K/V edge is detached."""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.models.gemma import modeling_gemma

from openpi.models_pytorch.recurrent_subtask import RecurrentSubtaskModel, RecurrentSubtaskDecoder
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

VARIANT = "official_pi05_parallel_multitask_v1"
SCHEMA_VERSION = 10
DISPLAY_SET = "official-pi05-parallel-action-stop-s42-v1"


class ParallelDecoder(RecurrentSubtaskDecoder):
    def compose_sequence(self, memory, mask, carry=None, resets=None, *, unroll=1):
        projected = self.memory_projection(memory.to(self.memory_projection.weight.dtype))
        if memory.shape[0] % unroll:
            raise ValueError("Sequence batch is not divisible by unroll")
        streams = memory.shape[0] // unroll
        initial = self.initial_memory[None].expand(streams, -1, -1)
        carry = initial if carry is None else carry
        if carry.shape != initial.shape:
            raise ValueError("Recurrent state shape mismatch")
        resets = torch.zeros(memory.shape[0], dtype=torch.bool, device=memory.device) if resets is None else resets
        states = []
        for t in range(unroll):
            section = slice(t * streams, (t + 1) * streams)
            carry = torch.where(resets[section, None, None], initial, carry)
            current = self.observation_norm(projected[section])
            observed, _ = self.memory_attention(self.query_norm(carry), current, current,
                key_padding_mask=~mask[section].bool(), need_weights=False)
            carry = self.memory_update(observed.flatten(0,1), carry.flatten(0,1)).view_as(initial)
            states.append(self.memory_norm(carry) + self.memory_type)
        return (torch.cat([projected, torch.cat(states)], 1),
                torch.cat([mask.bool(), torch.ones((memory.shape[0], self.memory_tokens),
                          device=memory.device, dtype=torch.bool)], 1), carry)


def parallel_layer(wrapper, layer_idx, prefix, suffix, attention_mask, position_ids, cond):
    """Separate attentions make the A->B gradient cut explicit at every layer."""
    models = [wrapper.paligemma.language_model, wrapper.gemma_expert.model]
    queries, keys, values, gates = [], [], [], []
    for model, hidden, condition in zip(models, [prefix, suffix], [None, cond]):
        layer = model.layers[layer_idx]
        normalized, gate = layer.input_layernorm(hidden, cond=condition)
        shape = (*normalized.shape[:-1], -1, layer.self_attn.head_dim)
        queries.append(layer.self_attn.q_proj(normalized).view(shape).transpose(1,2))
        keys.append(layer.self_attn.k_proj(normalized).view(shape).transpose(1,2))
        values.append(layer.self_attn.v_proj(normalized).view(shape).transpose(1,2))
        gates.append(gate)
    q, k = torch.cat(queries,2), torch.cat(keys,2)
    dummy = q.new_zeros((q.shape[0], q.shape[2], q.shape[-1]))
    cos, sin = wrapper.paligemma.model.language_model.rotary_emb(dummy, position_ids)
    q, k = modeling_gemma.apply_rotary_pos_emb(q,k,cos,sin,unsqueeze_dim=1)
    n = prefix.shape[1]
    attn = models[0].layers[layer_idx].self_attn
    p_out, _ = modeling_gemma.eager_attention_forward(attn,q[:,:,:n],k[:,:,:n],values[0],
        attention_mask[:,:,:n,:n],attn.scaling)
    a_out, _ = modeling_gemma.eager_attention_forward(attn,q[:,:,n:],
        torch.cat([k[:,:,:n].detach(), k[:,:,n:]],2),
        torch.cat([values[0].detach(), values[1]],2),attention_mask[:,:,n:,:],attn.scaling)
    outputs = []
    for i, (hidden, attended) in enumerate(zip([prefix,suffix],[p_out,a_out])):
        layer = models[i].layers[layer_idx]
        attended = attended.reshape(hidden.shape[0],hidden.shape[1],-1).to(layer.self_attn.o_proj.weight.dtype)
        out = modeling_gemma._gated_residual(hidden,layer.self_attn.o_proj(attended),gates[i])
        residual = out
        out, gate = layer.post_attention_layernorm(out,cond=[None,cond][i])
        out = layer.mlp(out.to(layer.mlp.up_proj.weight.dtype))
        outputs.append(modeling_gemma._gated_residual(residual,out,gate))
    return outputs[0], outputs[1]


class ParallelSubtaskModel(RecurrentSubtaskModel):
    def __init__(self, base, decoder_config=None, *, seed=42, unroll=4, **kwargs):
        super().__init__(base,decoder_config,recurrent=True,seed=seed,unroll=unroll)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.decoder = ParallelDecoder(decoder_config,recurrent=True).to(base.action_in_proj.weight.device)
        self.enable_backbone("action_stop")

    def enable_backbone(self, mode, **kwargs):
        if mode not in {"frozen","action_stop"}:
            raise ValueError("Parallel experiment supports action_stop only")
        self.set_stage("m3")
        self.backbone_mode = "action_stop"
        for p in self.base.paligemma_with_expert.paligemma.parameters():
            p.requires_grad_(True)
        self.train()

    def _tokens(self, global_prompts, state, subtasks=None):
        # Explicitly ignore internal diagnostic labels; never serialize them.
        return super()._tokens(global_prompts,state,None)

    def action_prefix(self, context, subtasks=None):
        return super().action_prefix(context,None)

    def prepare_features(self, observation, prompts):
        images, image_masks, _, _, state = self.base._preprocess_observation(observation,train=False)
        wrapper = self.base.paligemma_with_expert
        features = tuple(checkpoint(wrapper.embed_image,x,use_reentrant=False) if torch.is_grad_enabled()
                         else wrapper.embed_image(x) for x in images)
        tokens, token_mask = self._tokens(prompts,state)
        language = wrapper.embed_language_tokens(tokens)
        prefix = torch.cat([*features,language * math.sqrt(language.shape[-1])],1)
        mask = torch.cat([*(m[:,None].expand(f.shape[:2]) for m,f in zip(image_masks,features)),token_mask],1)
        return state, prefix, mask

    def joint_outputs(self, observation, prompts, actions, *, noise, time, detached=True):
        state, prefix, mask = self.prepare_features(observation,prompts)
        noisy = time[:,None,None]*noise + (1-time[:,None,None])*actions
        suffix, suffix_mask, suffix_ar, cond = self.base.embed_suffix(state,noisy,time)
        all_mask = torch.cat([mask,suffix_mask],1)
        ar = torch.cat([torch.zeros_like(mask),suffix_ar],1)
        attention = self.base._prepare_attention_masks_4d(make_att_2d_masks(all_mask,ar))
        positions = all_mask.cumsum(1)-1
        wrapper = self.base.paligemma_with_expert
        dtype = wrapper.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        prefix,suffix = prefix.to(dtype),suffix.to(dtype)
        if not detached:
            (prefix,suffix),_ = wrapper(attention_mask=attention,position_ids=positions,
                inputs_embeds=[prefix,suffix],use_cache=False,adarms_cond=[None,cond])
        else:
            for idx in range(len(wrapper.paligemma.language_model.layers)):
                def run(p,a,index=idx):
                    return parallel_layer(wrapper,index,p,a,attention,positions,cond)
                prefix,suffix = checkpoint(run,prefix,suffix,use_reentrant=False) if torch.is_grad_enabled() else run(prefix,suffix)
            prefix,_ = wrapper.paligemma.language_model.norm(prefix,cond=None)
            suffix,_ = wrapper.gemma_expert.model.norm(suffix,cond=cond)
        velocity = self.base.action_out_proj(suffix[:,-self.base.config.action_horizon:].float())
        return prefix,mask,velocity

    def forward(self,batch,*,carry=None,drop_condition=None,noise=None,time=None):
        if drop_condition is not None and any(drop_condition):
            raise ValueError("No subtask condition dropout in parallel action_stop")
        noise = self.base.sample_noise(batch.actions.shape,batch.actions.device) if noise is None else noise
        time = self.base.sample_time(batch.actions.shape[0],batch.actions.device) if time is None else time
        hidden,mask,velocity = self.joint_outputs(batch.observation,batch.global_prompts,batch.actions,noise=noise,time=time)
        projected,full_mask,next_carry = self.decoder.compose_sequence(hidden,mask,carry,batch.reset_mask,unroll=self.unroll)
        ce = self.decoder.loss_projected(batch.target_ids,batch.target_mask,projected,full_mask,self.embedding_weight)
        return dict(loss_subtask=ce,loss_action=(velocity.float()-(noise-batch.actions)).square().mean(),
                    carry=next_carry.detach(),generated_count=0,invalid_generation_count=0,empty_condition_count=0)

