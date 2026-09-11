"""No-motion catalog and handshake checks for the paired S experiments."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from piper_checkpoint import checkpoint_info, OFFICIAL_SHA256
from piper_gui_support import discover_checkpoints, server_command, validate_server, MODEL_SCRIPTS


class DecisionDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'checkpoints/pi05_piper_decision/decision_prefix_seed42_v1/step_005000'
        self.path.mkdir(parents=True)
        self.metadata = dict(schema_version=9, stage='recurrent_subtask', variant='official_pi05_recurrent_decision_v1',
                             completed_steps=5000, weights_sha256='a' * 64,
                             config=dict(engineering_smoke=False, mode='limited', arm='recurrent', label_version='reach_arm_v1', experiment='decision_prefix', engineering_condition_fixture=False,
                                         initialization='official_pi05_base', inherited_training_updates=0,
                                         official_weights_sha256=OFFICIAL_SHA256, parent_weights_sha256=OFFICIAL_SHA256))
        for name in ['model.safetensors', 'assets/eggplant_potato/norm_stats.json', 'assets/eggplant_potato/split.json','assets/decision/READY.json','assets/decision/decisions.json','assets/decision/reviewed_points.json']:
            p = self.path / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('{}')
        self.write()

    def write(self):
        (self.path / 'metadata.json').write_text(json.dumps(self.metadata))

    def handshake(self):
        return dict(checkpoint_info(self.path), checkpoint=str(self.path), state_dim=14, action_horizon=50)

    def test_both_arms_are_discovered_and_verified(self):
        for experiment in ['decision_prefix','decision_grounded']:
            self.metadata['config']['experiment']=experiment
            arm='recurrent'
            self.metadata['config']['arm'] = arm
            self.write()
            self.assertEqual(checkpoint_info(self.path)['arm'], arm)
            self.assertEqual(len(discover_checkpoints(self.root)), 1)
            validate_server(self.handshake(), self.root, self.path)
            self.assertNotIn('--parent-checkpoint', server_command(self.root, self.path, 18765))

    def test_mismatched_arm_weights_or_provenance_are_rejected(self):
        correct = self.handshake()
        for key, value in [('arm', 'stateless'), ('experiment', 'decision_grounded'), ('label_version', 'original'), ('weights_sha256', 'b' * 64), ('stage', 'm3'),
                           ('initialization', 'm3'), ('inherited_training_updates', 1), ('mode', 'full')]:
            with self.assertRaises(ValueError):
                validate_server(dict(correct, **{key: value}), self.root, self.path)

    def test_invalid_or_partial_candidates_are_hidden(self):
        original = copy.deepcopy(self.metadata)
        for key, value in [('engineering_smoke', True), ('arm', 'unknown'), ('experiment', 'unknown'), ('engineering_condition_fixture', True), ('label_version', 'original'), ('mode', 'full'),
                           ('parent_weights_sha256', 'b' * 64)]:
            self.metadata = copy.deepcopy(original)
            self.metadata['config'][key] = value
            self.write()
            self.assertEqual(discover_checkpoints(self.root), [])
        self.metadata = original
        self.write()
        (self.path / 'model.safetensors').rename(self.path / 'model.safetensors.partial')
        self.assertEqual(discover_checkpoints(self.root), [])

    def test_direct_recurrent_service_is_covered_by_port_cleanup(self):
        self.assertIn('serve_recurrent_subtask_piper.py', MODEL_SCRIPTS)


if __name__ == '__main__':
    unittest.main()
