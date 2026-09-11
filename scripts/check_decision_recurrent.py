"""Validate the exact exported checkpoint through native policy inference."""

import argparse
import dataclasses
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import safetensors

from openpi.policies.decision_recurrent_policy import create_decision_recurrent_policy
from openpi.training.reach_arm_data import data_config
from openpi.training.stage1_data import sha256_file
from openpi.models_pytorch.backbone_gradient import LoRALinear
from openpi.training import config, data_loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--allow-engineering", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    policy = create_decision_recurrent_policy(args.checkpoint, device=args.device,
                                             allow_engineering=args.allow_engineering)
    metadata = json.loads((args.checkpoint / "metadata.json").read_text())
    frozen_tensors = 0
    mode = metadata["config"]["mode"]
    if mode in {"frozen", "limited"}:
        # The unmerged LoRA file keeps original B separate from its learned delta.
        filename = "training_model.safetensors" if mode == "limited" else "model.safetensors"
        with safetensors.safe_open(str(args.checkpoint / filename), framework="pt", device="cpu") as trained:
            with safetensors.safe_open("checkpoints/pi05_base_pytorch/model.safetensors", framework="pt", device="cpu") as official:
                for key in official.keys():
                    if not key.startswith("paligemma_with_expert.paligemma."):
                        continue
                    target = "base." + key
                    if target not in trained.keys():
                        target = target.rsplit(".", 1)[0] + ".original." + target.rsplit(".", 1)[1]
                    actual = trained.get_tensor(target)
                    expected = official.get_tensor(key).to(actual.dtype)
                    if not torch.equal(actual, expected):
                        raise AssertionError(f"Frozen original B changed: {key}")
                    frozen_tensors += 1
        # safetensors stores the tied language embedding / LM head only once.
        assert frozen_tensors == 603, frozen_tensors
        backbone = policy.model.base.paligemma_with_expert.paligemma
        assert backbone.lm_head.weight.data_ptr() == backbone.language_model.embed_tokens.weight.data_ptr()
    cfg = config.get_config("pi05_piper_stage1")
    dc = data_config(cfg.model, split="val")
    dataset = data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    sample = dataset[0]
    observation = {"prompt":sample["task"], "state":np.asarray(sample["observation.state"]).copy(),
                   "images":{name:np.asarray(sample[f"observation.images.{name}"]).copy()
                             for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")}}
    state = observation["state"].copy()
    images = {name:value.copy() for name, value in observation["images"].items()}
    noise = np.random.default_rng(4250).standard_normal((50, 32)).astype(np.float32)
    left, right = policy.new_session(), policy.new_session()
    first = left.infer(observation, noise=noise)
    second = right.infer(observation, noise=noise)
    next_left = left.infer(observation, noise=noise)
    next_right = right.infer(observation, noise=noise)
    np.testing.assert_array_equal(next_left["actions"], next_right["actions"])
    if policy.model.decoder.recurrent:
        assert policy._memory is None
        assert left._memory is not right._memory
        torch.testing.assert_close(left._memory, right._memory, rtol=0, atol=0)
    left.reset()
    restarted = left.infer(observation, noise=noise)
    np.testing.assert_array_equal(first["actions"], restarted["actions"])
    assert first["actions"].shape == (50, 14) and np.isfinite(first["actions"]).all()
    np.testing.assert_array_equal(first["actions"], second["actions"])
    np.testing.assert_array_equal(state, observation["state"])
    for name, image in images.items():
        np.testing.assert_array_equal(image, observation["images"][name])
    assert first["subtask"] == second["subtask"]
    gradient_contract = None
    export_exact = None
    if args.allow_engineering:
        from openpi.training.recurrent_sequence import collate_sequence
        from openpi.training.subtask_batch import SubtaskTrainingDataset
        training_weights = args.checkpoint / "training_model.safetensors"
        assert sha256_file(training_weights) == metadata["training_weights_sha256"]
        model = policy.model
        model.enable_backbone("limited", lora_rank=16, lora_alpha=32, last_layers=2)
        safetensors.torch.load_model(model, training_weights, strict=True)
        modules = [(name, module) for name,module in model.named_modules() if isinstance(module,LoRALinear)]
        assert len(modules) == 14
        allowed = {name+"."+suffix for name,_ in modules for suffix in ("lora_a","lora_b")}
        trainable_b = {"base.paligemma_with_expert.paligemma."+name for name,p in
                       model.base.paligemma_with_expert.paligemma.named_parameters() if p.requires_grad}
        assert trainable_b == allowed
        depth = len(model.base.paligemma_with_expert.paligemma.language_model.layers)
        assert all(any(f".layers.{i}." in name for i in [depth-2,depth-1]) for name in allowed)
        assert all(module.lora_a.shape[0] == 16 and module.scale == 2 for _,module in modules)
        model.eval()
        unmerged = policy.new_session().infer(observation, noise=noise)
        np.testing.assert_array_equal(first["actions"],unmerged["actions"])
        assert first["subtask"] == unmerged["subtask"]
        export_exact = True
        wrapped = SubtaskTrainingDataset(dataset,dc)
        sequence = collate_sequence([(wrapped[i], i == 0) for i in [0,21,42,63]]).to(args.device)
        model.train()
        context = model.prepare_context(sequence.observation,sequence.global_prompts)
        projected,mask,carry = model.decoder.compose_sequence(context.memory,context.memory_mask,
                               resets=sequence.reset_mask,unroll=4)
        ce,ordinary = model.decision_s_loss(sequence.target_ids,sequence.target_mask,projected,mask,[2.,1.,1.,1.])
        ce.backward()
        assert all(p.grad is None for p in model.base.parameters())
        assert model.decoder.memory_update.weight_hh.grad.abs().sum() > 0
        assert carry.shape == (1,4,512)
        model.zero_grad(set_to_none=True)
        generated = model.codec.decode(model.decoder.generate_projected(projected.detach(),mask,model.embedding_weight))[0]
        fixed_noise = torch.randn_like(sequence.actions)
        fixed_time = torch.full((4,),.4,device=args.device)
        model.eval()
        with torch.no_grad():
            cached = model.action_loss(context,model.action_prefix(context,generated),sequence.actions,
                                       noise=fixed_noise,time=fixed_time)
            joint = model.action_loss_joint(sequence.observation,context,generated,sequence.actions,
                                           noise=fixed_noise,time=fixed_time)
        torch.testing.assert_close(cached,joint,rtol=.005,atol=2e-5)
        with torch.no_grad():
            elementwise=model.action_errors(context,generated,sequence.actions,noise=fixed_noise,time=fixed_time)
        torch.testing.assert_close(elementwise.mean(),joint,rtol=0,atol=0)
        model.train()
        model.action_loss_joint(sequence.observation,context,generated,sequence.actions,
                                noise=fixed_noise,time=fixed_time).backward()
        assert all(p.grad is None for p in model.subtask_parameters())
        nonzero_b = sum(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in model.backbone_parameters())
        nonzero_a = sum(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in model.action_parameters())
        assert nonzero_b > 0 and nonzero_a > 0
        assert all(p.grad is None for p in model.base.paligemma_with_expert.paligemma.vision_tower.parameters())
        for _,module in modules:
            assert all(not p.requires_grad and p.grad is None for p in module.original.parameters())
        model.zero_grad(set_to_none=True)
        from openpi.training.decision_data import load_assets, GroundingSamples
        from openpi.models_pytorch.decision_recurrent import decision_pairs, select_context, prefix_weighted_flow
        decisions,points,_=load_assets(args.checkpoint/"assets/decision")
        table=wrapped.raw_dataset.hf_dataset
        labels=list(table["subtask"])
        lid=next(i for i,label in enumerate(labels) if "reach the handle of the lid with" in label)
        obj=next(i for i,label in enumerate(labels) if "reach the " in label and "handle" not in label and " with the " in label)
        # One causal stream of four actual frames; engineering branch fixture only.
        sequence=collate_sequence([(wrapped[i],n==0) for n,i in enumerate([lid,lid+21,obj,obj+21])]).to(args.device)
        context=model.prepare_context(sequence.observation,sequence.global_prompts)
        positives=list(sequence.labels)
        pairs=decision_pairs(positives,sequence.global_prompts,positives,["ok"]*4,[False]*4,[True]*4,
                             model.decision_rules,model.valid_prompts)
        assert sum(r["kind"]=="arm" for r in pairs)==2 and sum(r["kind"]=="object" for r in pairs)==2
        fixed_noise=torch.randn_like(sequence.actions);fixed_time=torch.full((4,),.4,device=args.device)
        positive_errors=model.action_errors(context,positives,sequence.actions,noise=fixed_noise,time=fixed_time)
        indices=[p["index"] for p in pairs]
        negative_context=dataclasses.replace(select_context(context,indices),global_prompts=tuple(p["prompt"] for p in pairs))
        negative_errors=model.action_errors(negative_context,[p["label"] for p in pairs],sequence.actions[indices],
                                           noise=fixed_noise[indices],time=fixed_time[indices])
        hinge=torch.relu(.01+positive_errors[indices,:25,:14].mean((1,2))-negative_errors[:,:25,:14].mean((1,2)))
        ranking=hinge.mean()
        assert torch.isfinite(ranking) and ranking>0,float(ranking)
        ranking.backward()
        assert all(p.grad is None for p in model.subtask_parameters())
        rank_b=sum(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in model.backbone_parameters())
        rank_a=sum(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in model.action_parameters())
        assert rank_b>0 and rank_a>0
        model.zero_grad(set_to_none=True)
        ground_batch=ground_points=None
        if model.grounded:
            grounding=GroundingSamples(wrapped,points,split="val")
            ground_batch,ground_points=grounding.batch(grounding.select(0,0),args.device)
            gctx=model.prepare_context(ground_batch.observation,ground_batch.global_prompts)
            auxiliary,_,_=model.decoder.localization_loss(gctx.memory,gctx.memory_mask,ground_points)
            auxiliary.backward()
            assert all(p.grad is None for p in model.base.parameters())
            for name in ["ground_query","ground_key"]:
                assert any(p.grad is not None and p.grad.abs().sum()>0 for p in getattr(model.decoder,name).parameters())
            model.zero_grad(set_to_none=True)
        from unittest.mock import patch
        from openpi.models_pytorch.subtask_decoder import SubtaskGeneration
        fixture=SubtaskGeneration(sequence.target_ids,sequence.target_mask,
                   torch.ones(4,dtype=torch.bool,device=args.device),torch.zeros(4,device=args.device))
        with patch.object(model.decoder,"generate_projected",return_value=fixture):
            result=model(sequence,drop_condition=[False]*4,rank_stable_mask=[True]*4,
                         decision_weights=[2.]*4,ground_batch=ground_batch,ground_points=ground_points)
            assert result["rank_arm_pairs"]==2 and result["rank_object_pairs"]==2
            if model.grounded:assert result["grounding_ce"]>0
            (result["loss_subtask"]+result["loss_action"]).backward()
            assert torch.isfinite(result["loss_subtask"]+result["loss_action"])
        model.zero_grad(set_to_none=True)
        gradient_contract = dict(grounding_only_S=model.grounded, both_arm_object_pairs_verified=True, prefix25_rank_only=True, weighted_ce_only_S=True,ce_only_S=True,recurrence_gradient_nonzero=True,
                                 native14_ranking_only_A_B=True,ranking_nonzero_A_tensors=rank_a,ranking_nonzero_B_tensors=rank_b,
                                 actual_forward_ranking_fixture=True,elementwise_joint_exact=True,action_only_A_and_limited_B=True,
                                 action_gradients_in_S=False,nonzero_B_tensors=nonzero_b,nonzero_A_tensors=nonzero_a,
                                 LoRA_modules=14,trainable_B_tensors=28,joint_cache_flow_parity=True,
                                 joint_cache_parity_rtol=.005)
    try:
        policy.infer({**observation, "subtask":"external target"})
    except ValueError:
        pass
    else:
        raise AssertionError("External subtask supervision was accepted")
    result = {"passed":True, "checkpoint":str(args.checkpoint), "metadata":policy.metadata,
              "gradient_contract":gradient_contract, "merged_unmerged_inference_exact":export_exact,
              "unchanged_original_B_tensors":frozen_tensors,
              "shared_embedding_head_alias_verified":mode in {"frozen", "limited"},
              "native_shape":[50,14], "fixed_noise_repeat_equal":True, "inputs_unchanged":True,
              "external_subtask_rejected":True, "session_isolation":True, "reset_reproducible":True, "subtask":first["subtask"], "subtask_status":first["subtask_status"],
              "scope":"loadable experimental robot policy; not a physical readiness or task-success result"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
