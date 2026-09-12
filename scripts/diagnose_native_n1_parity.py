"""Read-only full-checkpoint N1 parity diagnosis; run through GPU reservation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
import torch.nn.functional as F
from openpi.policies.native_subtask_policy import create_native_policy
from openpi.training import config, data_loader
from openpi.training.reach_arm_data import data_config
from openpi.training.subtask_batch import SubtaskTrainingDataset, collate_subtask


def difference(a, b):
    d = (a.float() - b.float()).abs()
    return dict(shape=list(a.shape), unequal=int((a != b).sum()), elements=a.numel(),
                max_abs=float(d.max()), mean_abs=float(d.mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4); torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(.9, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    policy = create_native_policy(a.checkpoint, device='cuda:0', allow_engineering=True)
    model = policy.model.eval()
    dc = data_config(config.get_config('pi05_piper_stage1').model, split='val')
    raw = data_loader.create_torch_dataset(dc, 50, model.base.config)
    batch = collate_subtask([SubtaskTrainingDataset(raw, dc)[0]]).to('cuda:0')
    fixed = torch.randn_like(batch.actions); times = torch.full((1,), .4, device='cuda:0')
    captured = []
    original = model.text_logits
    def capture(hidden):
        captured.append(hidden.detach().clone())
        return original(hidden)
    model.text_logits = capture
    with torch.no_grad():
        joint, velocity = model.joint_outputs(batch.observation, batch.global_prompts, batch.actions,
            batch.target_ids, batch.target_mask, noise=fixed, time=times)
        context = model.prepare_context(batch.observation, batch.global_prompts)
        ids, valid = model.teacher_inputs(batch.target_ids, batch.target_mask)
        cached, _ = model.logits_cached(context, ids, valid)
        model.text_logits = original
        start = len(model.cue_ids) - 1
        jh, ch = captured
        selected = ch[:, start:]
        aligned = original(selected)
        head = model.base.paligemma_with_expert.paligemma
        jn, _ = head.language_model.norm(jh, cond=None)
        cn, _ = head.language_model.norm(ch, cond=None)
        weight32 = head.lm_head.weight.float()
        j32 = F.linear(jn.float(), weight32)
        c32 = F.linear(cn.float(), weight32)[:, start:]
        noisy = times[:, None, None] * fixed + (1-times[:, None, None]) * batch.actions
        native = model.base.denoise_step(context.state, context.mask, context.cache, noisy, times)
        mask = batch.target_mask
        result = dict(checkpoint=str(a.checkpoint), metadata=policy.metadata,
            source_sha256=hashlib.sha256(Path('src/openpi/models_pytorch/native_subtask.py').read_bytes()).hexdigest(),
            projection_input_shapes=[list(t.shape) for t in captured],
            hidden=difference(jh[mask], selected[mask]),
            normalized_hidden=difference(jn[mask], cn[:, start:][mask]),
            original_bf16=difference(joint[mask], cached[:, start:][mask]),
            same_shape_bf16=difference(joint[mask], aligned[mask]),
            fp32_head_tf32_disabled=difference(j32[mask], c32[mask]),
            action=difference(velocity, native),
            argmax_disagreements=int((joint[mask].argmax(-1) != cached[:, start:][mask].argmax(-1)).sum()),
            ce_joint=float(model.ce(joint, batch.target_ids, mask)),
            ce_original_cached=float(model.ce(cached[:, start:], batch.target_ids, mask)),
            ce_same_shape_cached=float(model.ce(aligned, batch.target_ids, mask)))
        result['head_shape_isolated'] = all(result[k]['unequal'] == 0 for k in
            ['hidden', 'normalized_hidden', 'same_shape_bf16', 'action'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
if __name__ == '__main__': main()
