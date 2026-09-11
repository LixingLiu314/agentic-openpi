"""Freeze R2 exploration protocol and re-score archived R0/R1 without GPU use."""
import json
from pathlib import Path
import shutil

import numpy as np

from openpi.training.stage1_data import sha256_file
from openpi.training.subtask_transition import read_jsonl
from openpi.training.transition_metrics_v2 import causal_sample, report, semantic_gates


def main():
    root=Path.cwd()
    output=root/'assets/pi05_piper_transition/eggplant_potato/r2_v1'
    output.mkdir(parents=True,exist_ok=False)
    original=root/'checkpoints/pi05_piper_transition/r1_seed42/transition_assets'
    rows=read_jsonl(original/'frames_train.jsonl')
    train_episodes={}
    for r in rows:train_episodes.setdefault(r['task'],set()).add(r['episode'])
    rng=np.random.default_rng(42)
    calibration=[]
    for task,eps in sorted(train_episodes.items()):
        calibration.extend(rng.choice(sorted(eps),size=10,replace=False).tolist())
    fit=sorted({r['episode'] for r in rows}-set(calibration))
    old=json.loads((original/'protocol.json').read_text())
    protocol={'schema_version':2,'variant':'r2_temporal_completion_v1','seed':42,
              'parent_checkpoint':str(root/'checkpoints/pi05_piper_stage1/m3_pilot_seed42/step_003500'),
              'parent_weights_sha256':old['parent_weights_sha256'],'split_sha256':old['split_sha256'],'norm_sha256':old['norm_sha256'],
              'fit_episodes':fit,'calibration_episodes':sorted(calibration),
              'calibration_note':'Held out of R2 fitting; parent M3 previously trained on these original-train episodes.',
              'test_access':False,'physical_readiness_reviewed':False,
              'supervision':'Original annotation CE and persistent active-stage label-proxy completion; no physical relabeling claims.',
              'model':{'autoregressive_text':True,'frozen_B_A':True,'history_observations':4,'history_period_seconds':[.55,1.0],
                       'history_tokens_per_view':16,'active_stage_source':'past model outputs only in deployment and evaluation',
                       'future_observations_allowed':False,'external_subtask_allowed':False},
              'training':{'steps':5000,'global_batch':256,'save_interval':500,'warmup':500,'peak_lr':2.5e-5,'decay_lr':2.5e-6,
                          'schedule':'official CosineDecaySchedule','boundary_fraction':.5,'weight_decay':1e-10},
              'evaluation':{'window_frames':15,'hold_seconds':.3,'low_period':.764,'low_phases':[0.,.255,.509],
                            'boundary_em_gain_min':.08,'paired_episode_ci95_lower_min':0.,'stable_em_drop_max':.01,
                            'failure_rate_increase_max':0.,'early_increase_max':.02,'delay_failure_cost_seconds':2.5,
                            'low_rate_failure_inclusive_delay_reduction':.30,'action_mse_increase_max':.05,
                            'wrong_order_and_far_stage_increase_max':0.,
                            'history_protocol':'Independent empty-history causal rollout for each episode and each input rate; no future smoothing.',
                            'selection':'Internal original-train calibration only; original val comparison after selecting a candidate. Seal results by version.'},
              'files':{}}
    for name in ['frames_train.jsonl','frames_val.jsonl','boundary_index.jsonl','handoff_annotations.jsonl']:
        shutil.copy2(original/name,output/name)
        protocol['files'][name]=sha256_file(output/name)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    val=read_jsonl(output/'frames_val.jsonl')
    events=[r for r in read_jsonl(output/'boundary_index.jsonl') if r['split']=='val']
    prior=root/'logs/pi05_subtask_stage1/m3_native_dense_val'
    predictions=[r for path in sorted(prior.glob('semantics_rank_*.json')) for r in json.loads(path.read_text())]
    lookup={(r['episode'],r['frame']):r for r in predictions}
    r0=[dict(r,prediction=lookup[(r['episode'],r['frame'])]['prediction']) for r in val]
    r1=json.loads((root/'checkpoints/pi05_piper_transition/r1_seed42/predictions_001000.json').read_text())
    perfect=[dict(r,prediction=r['label']) for r in val]
    results={}
    for name,preds in [('R0',r0),('R1_1000',r1),('perfect_annotation_scorer_check',perfect)]:
        reports={'dense':report(preds,events)}
        reports.update({f'low_{phase}':report(causal_sample(preds,.764,phase),events) for phase in [0.,.255,.509]})
        results[name]=reports
        (output/(name+'_metrics_v2.json')).write_text(json.dumps(reports,indent=2)+'\n')
    gate=semantic_gates(results['R1_1000'],results['R0'])
    (output/'R1_vs_R0_gates_v2.json').write_text(json.dumps(gate,indent=2)+'\n')
    for key,score in results['perfect_annotation_scorer_check'].items():
        assert score['events']['model_missed_or_unstable']==0,key
        assert score['events']['short_correct']==score['events']['short_observed'],key
    print(json.dumps({'protocol':str(output/'protocol.json'),'fit_episodes':len(fit),'calibration_episodes':len(calibration),
                      'perfect_scorer_no_false_model_failures':True,'R1_still_rejected':not gate['passed'],
                      'r0':{k:{n:v['events'][n] for n in ['confirmable','confirmed','model_missed_or_unstable','short_observed','short_correct']} for k,v in results['R0'].items()}},indent=2))


if __name__=='__main__':main()
