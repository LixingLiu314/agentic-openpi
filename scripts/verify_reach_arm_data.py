"""Exercise the actual training data path without loading a policy or using GPUs."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from openpi.shared import normalize
from openpi.training import config, data_loader
from openpi.training.stage1_data import load_split, sha256_file
from openpi.training.subtask_batch import SubtaskTrainingDataset, collate_subtask

torch.set_num_threads(2)
root = Path('Datasets/eggplant_potato_reach_arm_v1').resolve()
assets = Path('assets/pi05_piper_reach_arm_v1/eggplant_potato').resolve()
manifest = json.loads((assets/'split.json').read_text())
audit = json.loads((assets/'audit.json').read_text())
records = {r['episode_index']:r for r in manifest['episodes']}
annotations = [json.loads(s) for s in (root/'meta/reach_arm_annotations.jsonl').read_text().splitlines()]
source = Path(json.loads((root/'meta/reach_arm_provenance.json').read_text())['source_root'])
cfg = config.get_config('pi05_piper_stage1')
base_data = cfg.data.create(cfg.assets_dirs,cfg.model)
norm = normalize.load(assets)
results = []
seen_labels = set()
for split in ('train','val'):
    batch_samples = []
    ids = load_split(assets/'split.json',split,root)
    d = dataclasses.replace(base_data,repo_id='local/eggplant_potato_reach_arm_v1',
                            local_root=str(root),split_manifest=str(assets/'split.json'),
                            split=split,norm_stats=norm)
    raw = data_loader.create_torch_dataset(d,50,cfg.model)
    wrapped = SubtaskTrainingDataset(raw,d)
    assert len(raw) == audit['split_frames'][split]
    assert set(raw.episodes) == set(ids)
    chosen = {}
    for ep in ids:
        r=records[ep]
        key=(r['task_index'],r['layout_visual'],r['lid_arm'],r['object_arm'])
        chosen.setdefault(key,ep)
    checks = 0
    examples=[]
    for ep in chosen.values():
        r=records[ep]
        original=pq.read_table(source/r['parquet_path']).to_pydict()
        actions=np.asarray(original['action'],dtype=np.float32)
        states=np.asarray(original['observation.state'],dtype=np.float32)
        ep_annotations=[a for a in annotations if a['episode_index']==ep]
        frames={0,r['length']-1}
        for a in ep_annotations:
            frames.update((a['start_frame'],a['end_frame_exclusive']-1,a['end_frame_exclusive']))
        offset=int(raw.episode_data_index['from'][raw.episode_positions[ep]])
        for frame in sorted(frames):
            if frame >= r['length']: continue
            sample=raw[offset+frame]
            np.testing.assert_array_equal(np.asarray(sample['observation.state']),states[frame])
            indices=np.minimum(frame+np.arange(50),r['length']-1)
            np.testing.assert_array_equal(np.asarray(sample['action']),actions[indices])
            original_label=original['subtask'][frame]
            annotated=[a for a in ep_annotations if a['start_frame']<=frame<a['end_frame_exclusive']]
            expected=annotated[0]['subtask'] if annotated else original_label
            assert sample['subtask']==expected
            seen_labels.add(expected)
            item=wrapped[offset+frame]
            assert item['supervision']['label']==expected
            assert item['supervision']['global_prompt']==r['task']
            assert item['model']['state'].shape==(32,)
            assert item['model']['actions'].shape==(50,32)
            assert np.isfinite(item['model']['state']).all()
            assert np.isfinite(item['model']['actions']).all()
            assert not ({'label','subtask','actor','target_ids','target_mask'} & set(item['model']))
            ids2=item['supervision']['target_ids'][item['supervision']['target_mask']]
            assert wrapped.codec.processor.decode(ids2[:-1].tolist())==expected
            if frame==0: examples.append(dict(episode=ep,frame=frame,label=expected))
            checks+=1
            if len(batch_samples)<2: batch_samples.append(item)
    batch=collate_subtask(batch_samples)
    assert batch.actions.shape==(2,50,32) and batch.target_ids.shape==(2,16)
    results.append(dict(split=split,episodes=len(ids),frames=len(raw),tested_strata=len(chosen),
                        sampled_training_items=checks,examples=examples))
    print(json.dumps(results[-1]),flush=True)

assert {a['subtask'] for a in annotations}.issubset(seen_labels)
assert any('with the' not in label for label in seen_labels)
# Independent train-only state moment calculation, from the untouched source.
state_arrays=[]
for ep in manifest['splits']['train']:
    a=pq.read_table(source/records[ep]['parquet_path'],columns=['observation.state']).to_pydict()
    state_arrays.append(np.asarray(a['observation.state'],dtype=np.float64))
joined=np.concatenate(state_arrays)
np.testing.assert_allclose(norm['state'].mean,joined.mean(axis=0),atol=2e-6,rtol=2e-6)
np.testing.assert_allclose(norm['state'].std,joined.std(axis=0),atol=2e-6,rtol=2e-6)
assert len(joined)==audit['split_frames']['train']
assert sha256_file(assets/'norm_stats.json')==audit['norm_stats_sha256']
assert sha256_file(assets/'split.json')==audit['split_file_sha256']
result=dict(passed=True,results=results,total_sampled_training_items=sum(r['sampled_training_items'] for r in results),
            all_six_new_reach_strings_covered=True,unchanged_phase_labels_covered=True,
            native_absolute_state_and_50_step_action_windows_match_source=True,
            episode_tail_repeat_matches_source=True,training_target_roundtrip=True,
            supervision_excluded_from_observation=True,independent_train_only_state_moments_match=True,
            actual_collation_shapes={'actions':[2,50,32],'target_ids':[2,16]},
            gpu_used=False,policy_loaded=False,training_started=False)
(assets/'native_loader_check.json').write_text(json.dumps(result,indent=2)+'\n')
(assets/'READY.json').write_text(json.dumps(dict(status='ready_for_training_configuration',
    dataset_root=str(root),assets_root=str(assets),split_sha256=audit['split_file_sha256'],
    norm_stats_sha256=audit['norm_stats_sha256'],native_loader_check_sha256=sha256_file(assets/'native_loader_check.json')),indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
