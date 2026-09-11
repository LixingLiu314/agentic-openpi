"""CPU-only tests of display semantics and durable-log replay."""
import json
import tempfile
import unittest
from pathlib import Path

from openpi.training.wandb_standard import AXIS, class_rows, configure_run, event_payload


class DisplayContractTest(unittest.TestCase):
    def test_train_validation_are_distinct_at_same_optimizer_step(self):
        train = event_payload({"event": "train", "step": 500, "loss_subtask": 0.1, "loss_action": 0.2}, {})
        val = event_payload({"event": "validation", "step": 500, "loss_subtask": 0.3, "flow_generated_native14_normalized": 0.4}, {})
        self.assertEqual(set(train) & set(val), {AXIS})
        self.assertEqual(train[AXIS], val[AXIS])
        self.assertNotEqual(next(k for k in train if "flow" in k), next(k for k in val if "flow" in k))

    def test_counts_are_global_rates_and_speed_is_update_only(self):
        row = {"event": "train", "step": 20, "generated_count": 256, "invalid_generation_count": 2,
               "empty_condition_count": 24, "seconds": 8, "grad_norms": {"subtask": 0.2, "action_backbone": 0.1}}
        payload = event_payload(row, {"global_batch_size": 256, "steps": 5000})
        self.assertEqual(payload["train/invalid_generation_rate"], 2 / 256)
        self.assertEqual(payload["perf/examples_per_second"], 32)
        self.assertEqual(payload["progress/fraction"], 20 / 5000)

    def test_metadata_and_classes_never_expand_to_scalar_charts(self):
        self.assertEqual(event_payload({"event": "ready", "start": 0, "train_frames": 123}, {}), {})
        row = {"event": "validation", "step": 0, "per_class": {"reach": {"f1": .8, "recall": .9, "support": 10}}}
        self.assertEqual(event_payload(row, {}), {AXIS: 0})
        self.assertEqual(class_rows(row), [[0, "reach", .8, .9, 10]])

    def test_rejects_wrong_units_nonfinite_and_missing_step(self):
        for row in ({"event": "train"}, {"event": "train", "step": -1},
                    {"event": "validation", "step": 0, "macro_f1": 80},
                    {"event": "train", "step": 1, "loss_subtask": float("nan")}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                event_payload(row, {})

    def test_every_namespace_has_explicit_axis(self):
        class Fake:
            def __init__(self): self.calls = []
            def define_metric(self, name, **kw): self.calls.append((name, kw))
        fake = Fake()
        configure_run(fake)
        for name, settings in fake.calls:
            if name.endswith("/*"):
                self.assertEqual(settings["step_metric"], AXIS)
                self.assertFalse(settings["step_sync"])

    def test_partial_jsonl_waits_and_resume_skips_transport_indices(self):
        from sync_wandb_standard import read_complete_rows
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "metrics.jsonl"
            p.write_bytes(b'{"event":"ready"}\n{"event":"train","step":1}\n{"event":')
            self.assertEqual([n for n, _ in read_complete_rows(p, 0)], [1, 2])
            self.assertEqual([n for n, _ in read_complete_rows(p, 2)], [2])
            with p.open("ab") as f: f.write(b'"validation","step":1}\n')
            self.assertEqual([n for n, _ in read_complete_rows(p, 3)], [3])
            p.write_text('{bad}\n')
            with self.assertRaises(ValueError): list(read_complete_rows(p, 0))


if __name__ == "__main__":
    unittest.main()
