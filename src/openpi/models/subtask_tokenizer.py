"""One prompt/target codec for subtask training and deployment."""

import numpy as np

from openpi.models.tokenizer import PaligemmaTokenizer


class SubtaskTextCodec:
    def __init__(self, max_prompt_tokens=200, max_subtask_tokens=16, state_dim=14):
        self.max_prompt_tokens = max_prompt_tokens
        self.max_subtask_tokens = max_subtask_tokens
        self.state_dim = state_dim
        # One extra position makes overflow an error instead of silently accepting truncation.
        self.prompt_tokenizer = PaligemmaTokenizer(max_prompt_tokens + 1)
        self.processor = self.prompt_tokenizer._tokenizer  # noqa: SLF001
        self.bos_id, self.eos_id, self.pad_id = (
            self.processor.bos_id(),
            self.processor.eos_id(),
            self.processor.pad_id(),
        )
        if (self.bos_id, self.eos_id, self.pad_id) != (2, 1, 0):
            raise ValueError("Unexpected PaliGemma special-token mapping")

    def prompts(self, global_prompts, normalized_states, subtasks=None):
        """State is already normalized; serialize only its real robot dimensions."""
        if len(global_prompts) != len(normalized_states):
            raise ValueError("Prompt/state batch mismatch")
        if subtasks is not None and len(subtasks) != len(global_prompts):
            raise ValueError("Subtask condition batch mismatch")
        ids, masks = [], []
        for index, (global_prompt, state) in enumerate(zip(global_prompts, normalized_states, strict=True)):
            prompt = global_prompt
            if subtasks is not None and subtasks[index]:
                prompt = f"{global_prompt}, Subtask: {subtasks[index]}"
            tokens, mask = self.prompt_tokenizer.tokenize(prompt, np.asarray(state[: self.state_dim]))
            if mask.sum() > self.max_prompt_tokens:
                raise ValueError("Prompt exceeds the configured budget; do not silently truncate state/condition")
            ids.append(tokens[: self.max_prompt_tokens])
            masks.append(mask[: self.max_prompt_tokens])
        return np.stack(ids), np.stack(masks)

    def targets(self, labels):
        ids = np.full((len(labels), self.max_subtask_tokens), self.pad_id, dtype=np.int64)
        mask = np.zeros_like(ids, dtype=bool)
        for row, label in enumerate(labels):
            if label is None or not label.strip():
                continue
            tokens = [*self.processor.encode(label), self.eos_id]
            if len(tokens) > self.max_subtask_tokens:
                raise ValueError(f"Subtask target exceeds budget: {label!r}")
            ids[row, : len(tokens)] = tokens
            mask[row, : len(tokens)] = True
        return ids, mask

    def decode(self, generation):
        ids = generation.token_ids.cpu().numpy()
        masks = generation.token_mask.cpu().numpy()
        ended = generation.ended.cpu().numpy()
        texts, statuses = [], []
        for row, mask, finished in zip(ids, masks, ended, strict=True):
            tokens = [int(token) for token in row[mask] if token not in {self.bos_id, self.eos_id, self.pad_id}]
            text = self.processor.decode(tokens).strip()
            status = "ok" if finished and text else "empty" if finished else "truncated"
            # Invalid/unfinished generation uses the condition-empty path, never a GT fallback.
            texts.append(text if status == "ok" else "")
            statuses.append(status)
        return texts, statuses
