"""Read-only gradient diagnostics; the optimizer retains joint A+B clipping."""

import math

import torch

from openpi.training import wandb_standard


OBSERVABILITY_VERSION = 2
GRADIENT_METRICS = {
    "action": "optim/grad_norm_a",
    "backbone": "optim/grad_norm_b",
    "backbone_vision": "optim/grad_norm_b_vision",
    "backbone_language": "optim/grad_norm_b_language",
}


class GradientObserver:
    def __init__(self, model):
        action = list(model.action_parameters())
        backbone = list(model.backbone_parameters())
        visual_ids = {
            id(p) for name, p in model.base.paligemma_with_expert.paligemma.named_parameters()
            if "vision_tower" in name or "multi_modal_projector" in name
        }
        vision = [p for p in backbone if id(p) in visual_ids]
        language = [p for p in backbone if id(p) not in visual_ids]
        assert not ({id(p) for p in action} & {id(p) for p in backbone})
        self.groups = {"action":action, "backbone_vision":vision, "backbone_language":language}
        self.backbone = backbone

    @staticmethod
    def norm(parameters):
        gradients = [p.grad.detach() for p in parameters if p.grad is not None]
        if not gradients:
            return 0.0
        # dtype controls accumulation without changing or replacing any .grad tensor.
        norms = [torch.linalg.vector_norm(g, dtype=torch.float32) for g in gradients]
        value = float(torch.linalg.vector_norm(torch.stack(norms)))
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite diagnostic gradient norm")
        return value

    @torch.no_grad()
    def measure(self):
        result = {name:self.norm(parameters) for name, parameters in self.groups.items()}
        result["backbone"] = math.hypot(result["backbone_vision"], result["backbone_language"])
        result["backbone_tensors_with_grad"] = sum(p.grad is not None for p in self.backbone)
        result["backbone_trainable_tensors"] = len(self.backbone)
        return result


def event_payload(row, config):
    payload = wandb_standard.event_payload(row, config)
    if row.get("event") == "train":
        norms = row.get("grad_norms", {})
        for source, target in GRADIENT_METRICS.items():
            if source in norms:
                payload[target] = wandb_standard.number(norms[source])
        if "backbone_tensors_with_grad" in norms:
            count = wandb_standard.number(norms["backbone_tensors_with_grad"])
            total = wandb_standard.number(norms["backbone_trainable_tensors"])
            if not 0 <= count <= total:
                raise ValueError("Invalid backbone gradient tensor count")
            payload["optim/b_tensors_with_grad"] = count
            payload["optim/b_grad_tensor_fraction"] = count / total if total else 0.0
    return payload


def configure_run(run):
    wandb_standard.configure_run(run)
    run.summary.update({
        "observability/version":OBSERVABILITY_VERSION,
        "observability/b_loss_source":"train/action_flow_mse_all32; shared with A; no independent B objective",
        "observability/gradient_norm_scope":"DDP-reduced gradients on rank0 before unchanged joint A+B clipping",
        "observability/vision_scope":"SigLIP vision_tower plus multi_modal_projector",
        "observability/language_scope":"remaining shared B, including token embeddings",
    })


def log_event(run, row, config, *, upload_step=None):
    if row.get("event") == "ready":
        run.summary.update({f"trainable/{key}":value for key,value in row["trainable_parameters"].items()})
        return True
    payload = event_payload(row, config)
    if not payload:
        return False
    if row.get("per_class"):
        import wandb
        payload["media/subtask_per_class"] = wandb.Table(
            columns=["optimizer_step", "subtask", "f1", "recall", "support"],
            data=wandb_standard.class_rows(row))
    # Shared losses and B diagnostics use one row and the same optimizer-step axis.
    run.log(payload, step=upload_step)
    return True
