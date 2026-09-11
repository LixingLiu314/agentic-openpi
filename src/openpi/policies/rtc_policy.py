"""Session-local RTC protocol; clients reference only this server's last output."""
import copy

import jax
import numpy as np
import torch

from openpi.models.model import Observation
from openpi.models_pytorch.rtc_sampling import sample_rtc

PROTOCOL = "piper_rtc_v1"


def native_overlap_to_model(policy, normalized_state, previous):
    """Invert the audited affine output transform at the new measured state.

    This handles train-only quantiles, 12 relative joints and absolute grippers
    together. Reusing the previous normalized delta would use the wrong origin.
    """
    shape = (policy.model.base.config.action_horizon, policy.model.base.config.action_dim)
    state = np.asarray(normalized_state).copy()
    zero = policy.output_transform({"state": state.copy(), "actions": np.zeros(shape, np.float32)})["actions"]
    one = policy.output_transform({"state": state.copy(), "actions": np.ones(shape, np.float32)})["actions"]
    scale = one - zero
    if not np.isfinite(scale).all() or (np.abs(scale) < 1e-9).any():
        raise ValueError("RTC cannot invert the action normalization")
    return ((previous - zero[:len(previous)]) / scale[:len(previous)]).astype(np.float32)


class RTCPolicy:
    def __init__(self, policy):
        self.policy = policy
        # Serving only: VJP needs gradients with respect to noisy actions,
        # not model weights. Avoid storing weight-gradient activations.
        self.policy.model.requires_grad_(False)
        self.reset()

    @property
    def model(self):
        return self.policy.model

    @property
    def metadata(self):
        return dict(self.policy.metadata, rtc_protocol=PROTOCOL, rtc_supported=True,
                    rtc_method="soft_mask_endpoint_vjp", rtc_max_execution_horizon=25)

    def reset(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self._previous = None
        self._query_id = 0
        self._prompt = None

    def new_session(self):
        underlying = self.policy.new_session() if hasattr(self.policy, "new_session") else copy.copy(self.policy)
        return type(self)(underlying)

    @torch.no_grad()
    def infer(self, obs, *, noise=None):
        observation_dict = dict(obs)
        request = observation_dict.pop("rtc", None)
        if request is None:
            # Switching modes on one socket would make an old reference unsafe.
            self._previous, self._query_id, self._prompt = None, 0, None
            return self.policy.infer(observation_dict, noise=noise)
        if not isinstance(request, dict) or set(request) != {"protocol", "query_id", "previous_query_id", "consumed_steps", "delay_steps"}:
            raise ValueError("Invalid RTC request fields")
        if request["protocol"] != PROTOCOL:
            raise ValueError("Unsupported RTC protocol")
        for key in ("query_id", "consumed_steps", "delay_steps"):
            if type(request[key]) is not int:
                raise ValueError("RTC counters must be integers")
        query_id, consumed, delay = (request[k] for k in ("query_id", "consumed_steps", "delay_steps"))
        if query_id != self._query_id + 1:
            raise ValueError("Out-of-order RTC query")
        prompt = observation_dict.get("prompt")
        if self._previous is None:
            if request["previous_query_id"] is not None or consumed != 0 or delay != 0:
                raise ValueError("First RTC query cannot reference a previous chunk")
            result = self.policy.infer(observation_dict, noise=noise)
            overlap = 0
        else:
            if prompt != self._prompt:
                raise ValueError("Start a new connection when changing the RTC task")
            if request["previous_query_id"] != self._query_id or type(request["previous_query_id"]) is not int:
                raise ValueError("RTC previous chunk does not belong to this session")
            if not 1 <= consumed < 50 or not 0 <= delay < 50-consumed:
                raise ValueError("RTC delay leaves no usable overlap")
            forbidden = {"action", "actions", "subtask", "subtask_target_ids", "subtask_target_mask", "target_ids", "target_mask", "labels"}
            if forbidden & observation_dict.keys():
                raise ValueError("Deployment observations must not contain supervision")
            p = self.policy
            begun = p._timestamp()
            inputs = p.input_transform(observation_dict)
            tensors = jax.tree.map(lambda value: torch.as_tensor(np.asarray(value), device=p.device)[None], inputs)
            observation = Observation.from_dict(tensors)
            after_inputs = p._timestamp()
            timings = {"input_ms": (after_inputs-begun)*1000}
            context = p.model.prepare_context(observation, [prompt], timings=timings)
            start_text = p._timestamp()
            recurrent = hasattr(p, "_memory") and hasattr(p.model, "generate_with_memory")
            if recurrent:
                texts, statuses, generation, next_memory = p.model.generate_with_memory(context, p._memory)
            else:
                texts, statuses, generation = p.model.generate_subtask(context)
            after_text = p._timestamp()
            prefix = p.model.action_prefix(context, texts)
            after_prefix = p._timestamp()
            prior = native_overlap_to_model(p, tensors["state"][0].cpu().numpy(), self._previous[consumed:])
            actions = sample_rtc(p.model, context, prefix,
                                 torch.as_tensor(prior, device=p.device)[None], delay,
                                 noise=noise, num_steps=p.num_steps)
            after_actions = p._timestamp()
            result = p.output_transform({"state": tensors["state"][0].cpu().numpy(),
                                         "actions": actions[0].float().cpu().numpy()})
            result.update(subtask=texts[0], subtask_status=statuses[0],
                          subtask_score=float(generation.mean_log_probability[0]))
            ended = p._timestamp()
            timings.update(subtask_ms=(after_text-start_text)*1000, prefix2_ms=(after_prefix-after_text)*1000,
                           flow_ms=(after_actions-after_prefix)*1000, output_ms=(ended-after_actions)*1000,
                           infer_ms=(ended-begun)*1000)
            result["policy_timing"] = timings
            if recurrent:
                p._memory = None if next_memory is None else next_memory.detach()
                p._task = prompt
            overlap = len(prior)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise FloatingPointError("RTC returned invalid native actions")
        self._previous = actions.copy()
        # The robot clips grippers identically; constrain to the commands it
        # actually receives, not unreachable values outside its native range.
        self._previous[:, [6, 13]] = np.clip(self._previous[:, [6, 13]], 0, .09)
        self._query_id, self._prompt = query_id, prompt
        result["rtc"] = dict(protocol=PROTOCOL, query_id=query_id, previous_query_id=request["previous_query_id"],
                             consumed_steps=consumed, delay_steps=delay, overlap_steps=overlap,
                             guidance="soft_mask_endpoint_vjp" if overlap else "initial_ordinary_flow")
        return result
