"""Version only the three user-selected reach labels; preserve all sensor/action data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.models.subtask_tokenizer import SubtaskTextCodec
from openpi.shared import normalize
from openpi.training.stage1_data import action_windows, load_split, manifest_digest, sha256_file

LABELS = {'reach the handle of the lid', 'reach the eggplant', 'reach the sweet potato'}
METHOD = 'reach_arm_v1_state_action_consensus'


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def actor_from_reach(array, start, end):
    anchor = max(0, start - 1)
    x = array[anchor:end]
    displacement = [float(np.linalg.norm(x[:, i:i+6] - array[anchor, i:i+6], axis=1).max())
                    for i in (0, 7)]
    i = int(np.argmax(displacement))
    if displacement[i] <= .1 or displacement[i] <= 2 * max(displacement[1-i], 1e-6):
        raise ValueError(f'Ambiguous reach movement: {displacement}')
    return ('left', 'right')[i], displacement


def stratified_split(records, seed=42):
    # Refuse unreviewed duplicates rather than silently leaking them across splits.
    # The previously audited current source has one episode per exact/source group.
    for key in ('source_path', 'trajectory_sha256', 'video_triple_sha256'):
        values = [r[key] for r in records if r.get(key)]
        if len(values) != len(set(values)):
            raise ValueError(f'Duplicate {key}; requires grouped allocation before proceeding')
    strata = defaultdict(list)
    for r in records:
        strata[(r['task_index'], r['layout_visual'], r['lid_arm'], r['object_arm'])].append(r['episode_index'])
    splits = {'train': [], 'val': []}
    allocation = []
    rng = np.random.default_rng(seed)
    for key, episodes in sorted(strata.items()):
        ids = np.array(sorted(episodes))
        rng.shuffle(ids)
        count = max(1, int(np.floor(len(ids) * .1 + .5)))
        assert count < len(ids)
        splits['val'].extend(ids[:count].tolist())
        splits['train'].extend(ids[count:].tolist())
        allocation.append(dict(task_index=key[0], layout=key[1], lid_arm=key[2], object_arm=key[3],
                               total=len(ids), train=len(ids)-count, val=count))
    splits = {k: sorted(v) for k, v in splits.items()}
    assert len(splits['train']) == 178 and len(splits['val']) == 20
    assert set(splits['train']).isdisjoint(splits['val'])
    assert sorted(splits['train'] + splits['val']) == sorted(r['episode_index'] for r in records)
    return splits, allocation


def prepare(args):
    source, output, assets = args.source.resolve(), args.output.resolve(), args.assets_output.resolve()
    if output.exists() or assets.exists():
        raise FileExistsError('Refusing to replace an existing dataset or asset version')
    source_manifest = json.loads(args.source_manifest.read_text())
    assert source_manifest['manifest_sha256'] == manifest_digest(source_manifest)
    reference = {r['episode']: r for r in json.loads(args.reference.read_text())['episodes']}
    info = json.loads((source/'meta/info.json').read_text())
    sources = {r['episode_index']: r for r in source_manifest['episodes']}
    assert len(sources) == 198 and set(sources) == set(reference)
    source_meta_hashes = {p.name: sha256_file(p) for p in (source/'meta').iterdir() if p.is_file()}
    codec = SubtaskTextCodec()
    staging = output.with_name(output.name + '.preparing-' + uuid.uuid4().hex[:8])
    asset_staging = assets.with_name(assets.name + '.preparing-' + uuid.uuid4().hex[:8])
    staging.mkdir(parents=True)
    asset_staging.mkdir(parents=True)
    shutil.copytree(source/'meta', staging/'meta')
    (staging/'videos').symlink_to(source/'videos', target_is_directory=True)
    records, annotations, arrays = [], [], {}
    counts, frame_counts = Counter(), Counter()
    frames = changed_frames = videos_checked = 0
    video_before = {}
    for episode, old_record in sorted(sources.items()):
        rel = old_record['parquet_path']
        original = source/rel
        assert sha256_file(original) == old_record['parquet_sha256'], f'Source changed: {episode}'
        table = pq.read_table(original)
        columns = table.to_pydict()
        state = np.asarray(columns['observation.state'], dtype=np.float32)
        action = np.asarray(columns['action'], dtype=np.float32)
        labels = columns['subtask']
        length = len(labels)
        assert state.shape == action.shape == (length, 14)
        assert np.isfinite(state).all() and np.isfinite(action).all()
        assert columns['frame_index'] == list(range(length))
        assert set(columns['episode_index']) == {episode}
        assert length == old_record['length']
        trajectory = hashlib.sha256(state.tobytes() + action.tobytes()).hexdigest()
        assert trajectory == old_record['trajectory_sha256']
        new_labels = list(labels)
        starts = [0] + [i for i in range(1, length) if labels[i] != labels[i-1]] + [length]
        episode_annotations = []
        for start, end in zip(starts[:-1], starts[1:]):
            label = labels[start]
            if label not in LABELS:
                continue
            actor_s, ds = actor_from_reach(state, start, end)
            actor_a, da = actor_from_reach(action, start, end)
            assert actor_s == actor_a, f'State/action disagree: {episode}/{label}'
            expected = reference[episode]['lid_arm_resolved' if 'lid' in label else 'object_arm_motion']
            assert actor_s == expected, f'Independent prior audit disagrees: {episode}/{label}'
            text = f'{label} with the {actor_s} arm'
            ids, mask = codec.targets([text])
            assert codec.processor.decode(ids[0, mask[0]][:-1].tolist()) == text
            new_labels[start:end] = [text] * (end - start)
            record = dict(episode_index=episode, start_frame=start, end_frame_exclusive=end,
                          original_subtask=label, subtask=text, actor=actor_s, frames=end-start,
                          state_joint_displacement_LR=ds, action_joint_displacement_LR=da,
                          method=METHOD, prior_role_audit_agrees=True, tokens_with_eos=int(mask.sum()))
            episode_annotations.append(record)
            annotations.append(record)
            counts[text] += 1
            frame_counts[text] += end-start
        assert len(episode_annotations) == 2, f'Expected two reach segments: {episode}'
        assert {a['original_subtask'] for a in episode_annotations} == {
            'reach the handle of the lid',
            'reach the eggplant' if old_record['task_index'] == 0 else 'reach the sweet potato'}
        changed = [a != b for a, b in zip(labels, new_labels)]
        assert changed == [a in LABELS for a in labels]
        assert sum(changed) == sum(a['frames'] for a in episode_annotations)
        index = table.schema.get_field_index('subtask')
        new_table = table.set_column(index, table.schema.field(index), pa.array(new_labels, type=table.schema.field(index).type))
        dest = staging/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new_table, dest, compression='snappy')
        loaded = pq.read_table(dest)
        assert loaded.equals(new_table, check_metadata=True)
        assert loaded.drop(['subtask']).equals(table.drop(['subtask']), check_metadata=True)
        assert sha256_file(original) == old_record['parquet_sha256']
        for v in old_record['videos']:
            p = source/v['path']
            assert (staging/v['path']).resolve() == p.resolve()
            before = p.stat()
            assert sha256_file(p) == v['sha256'], f'Video changed: {p}'
            with av.open(str(staging/v['path'])) as container:
                stream = container.streams.video[0]
                assert stream.frames == length and float(stream.average_rate) == info['fps']
                next(container.decode(stream))
            after = p.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            video_before[v['path']] = (after.st_size, after.st_mtime_ns)
            videos_checked += 1
        record = copy.deepcopy(old_record)
        record.update(parquet_sha256=sha256_file(dest), source_parquet_sha256=old_record['parquet_sha256'],
                      layout_visual=reference[episode]['layout_visual'],
                      lid_arm=reference[episode]['lid_arm_resolved'], object_arm=reference[episode]['object_arm_motion'])
        records.append(record)
        arrays[episode] = state, action
        frames += length
        changed_frames += sum(changed)
        if (episode+1) % 50 == 0:
            print(json.dumps({'episodes_written_and_checked':episode+1, 'changed_frames':changed_frames}), flush=True)
    assert len(annotations) == 396 and frames == 132939 and videos_checked == 594
    assert {p.name:sha256_file(p) for p in (source/'meta').iterdir() if p.is_file()} == source_meta_hashes
    splits, allocation = stratified_split(records)
    manifest = {k:copy.deepcopy(v) for k,v in source_manifest.items()
                if k not in {'splits','episodes','manifest_sha256'}}
    manifest.update(seed=42, splits=splits, episodes=records, dataset_root=str(output),
                    dataset_version='eggplant_potato_reach_arm_v1',
                    split_method='seed42 shuffle within task/layout/lid-actor/object-actor strata; 10% rounded per stratum',
                    duplicate_check='No repeated source_path, exact state/action hash or exact three-camera video triple',
                    strata=allocation, source_manifest_sha256=sha256_file(args.source_manifest))
    manifest['manifest_sha256'] = manifest_digest(manifest)
    dump(asset_staging/'split.json',manifest)
    stats = {'state':normalize.RunningStats(), 'actions':normalize.RunningStats()}
    for i, episode in enumerate(splits['train']):
        states, actions = arrays[episode]
        stats['state'].update(states)
        stats['actions'].update(action_windows(actions,states,50))
        if (i+1)%50 == 0: print(json.dumps({'normalization_train_episodes':i+1}),flush=True)
    norm = {key:value.get_statistics() for key,value in stats.items()}
    normalize.save(asset_staging,norm)
    annotation_path = staging/'meta/reach_arm_annotations.jsonl'
    annotation_path.write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in annotations))
    info['repo_id'] = 'local/eggplant_potato_reach_arm_v1'
    info['subtask_label_version'] = 'reach_arm_v1'
    dump(staging/'meta/info.json',info)
    provenance = dict(schema_version=1, method=METHOD, source_root=str(source), output_root=str(output),
                      source_manifest_path=str(args.source_manifest.resolve()),
                      source_manifest_sha256=sha256_file(args.source_manifest),
                      prior_role_audit_sha256=sha256_file(args.reference),
                      builder_sha256=sha256_file(Path(__file__)),
                      modified_columns=['subtask'], allowed_original_labels=sorted(LABELS),
                      suffix_template=' with the {left|right} arm',
                      source_metadata_sha256=source_meta_hashes,
                      annotations_sha256=sha256_file(annotation_path),
                      shared_video_root=str(source/'videos'),
                      notes=['All other subtask strings and every other parquet column are unchanged.',
                             'Videos are referenced through an absolute directory symlink; source dataset must remain available.',
                             'meta/info splits denotes full LeRobot data availability; actual experiment train/val membership is in assets split.json.',
                             'Actor is derived offline from demonstrated reach motion; no future action or true label is added to inference inputs.'])
    dump(staging/'meta/reach_arm_provenance.json',provenance)
    report = dict(schema_version=1, dataset_root=str(output), repo_id=info['repo_id'], assets_root=str(assets),
                  episodes=198, frames=frames, modified_segments=len(annotations), modified_frames=changed_frames,
                  unchanged_label_frames=frames-changed_frames, segment_counts=dict(counts), frame_counts=dict(frame_counts),
                  max_target_tokens=max(a['tokens_with_eos'] for a in annotations),
                  split_episodes={k:len(v) for k,v in splits.items()},
                  split_frames={k:sum(len(arrays[e][0]) for e in v) for k,v in splits.items()},
                  strata=allocation, norm_training_episodes=splits['train'],
                  normalization='Existing OpenPI RunningStats over every train state and every 50-step same-episode endpoint-repeat action window; 12 joints delta, grippers absolute.',
                  norm_stats_sha256=sha256_file(asset_staging/'norm_stats.json'),
                  split_file_sha256=sha256_file(asset_staging/'split.json'),
                  manifest_sha256=manifest['manifest_sha256'], annotations_sha256=sha256_file(annotation_path),
                  checks=dict(exact_non_subtask_columns_all_episodes=True, only_allowed_labels_modified=True,
                              each_episode_exactly_two_reach_segments=True, raw_state_action_actor_consensus=396,
                              prior_role_audit_agrees=396, token_roundtrip_passed=396,
                              source_parquet_hashes_unchanged=198, video_hash_header_decode_checks=594,
                              source_metadata_unchanged=True, duplicate_groups_absent=True,
                              train_val_disjoint_complete=True, test_split_absent=True),
                  training_started=False)
    dump(asset_staging/'audit.json',report)
    for split in splits:
        assert load_split(asset_staging/'split.json',split,staging) == splits[split]
    for rel, before in video_before.items():
        p=source/rel; assert (p.stat().st_size,p.stat().st_mtime_ns) == before
    for r in records:
        assert sha256_file(source/r['parquet_path']) == r['source_parquet_sha256']
    staging.rename(output)
    asset_staging.rename(assets)
    dump(assets/'PREPARED.json',{'dataset_root':str(output),'split_sha256':report['split_file_sha256'],
                                 'norm_stats_sha256':report['norm_stats_sha256'],
                                 'status':'integrity_passed_pending_native_loader_check'})
    print(json.dumps({k:v for k,v in report.items() if k not in {'norm_training_episodes','strata'}},ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['source','output','assets-output','source-manifest','reference']:
        parser.add_argument('--'+key,type=Path,required=True)
    prepare(parser.parse_args())
