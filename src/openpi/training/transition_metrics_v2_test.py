from openpi.training.transition_metrics_v2 import causal_sample, report, semantic_gates


def example():
    rows=[dict(episode=0,frame=i,timestamp=i/10,task="task",label="a" if i<10 else "b" if i<16 else "c",
               prediction="a" if i<10 else "b" if i<16 else "c",boundary_event="event" if 7<=i<=19 else None) for i in range(30)]
    events=[dict(event_id="ab",episode=0,task="task",transition="a -> b",old="a",new="b",t_label=1.,new_end_time=1.6,cell_start=0,cell_end=1.3),
            dict(event_id="bc",episode=0,task="task",transition="b -> c",old="b",new="c",t_label=1.6,new_end_time=3.,cell_start=1.3,cell_end=3.)]
    return rows,events


def test_perfect_dense_and_single_observation_have_no_model_failure():
    rows,events=example()
    dense=report(rows,events)["events"]
    assert dense["confirmed"]==2 and dense["model_missed_or_unstable"]==0
    low=report(causal_sample(rows,.764),events)["events"]
    assert low["short_observed"]>=1 and low["short_correct"]==low["short_observed"]
    assert low["model_missed_or_unstable"]==0


def test_wrong_single_observation_does_not_disappear():
    rows,events=example();sample=causal_sample(rows,.764)
    for r in sample:
        if r["label"]=="b":r["prediction"]="a"
    metrics=report(sample,events)["events"]
    assert metrics["short_correct"]<metrics["short_observed"]
    assert any(e["status"]=="observed_wrong_unconfirmable" for e in metrics["events_detail"])


def test_stuck_prediction_has_failure_cost_and_wrong_order_is_counted():
    rows,events=example()
    for r in rows:r["prediction"]="a"
    score=report(rows,events)
    assert score["events"]["model_missed_or_unstable"]==2
    assert score["events"]["capped_late_cost_mean"]==2.5
    rows[11]["prediction"]="c"
    score=report(rows,events)
    assert score["frames"]["forward_jumps"]==1
    assert score["frames"]["backward_changes"]==1


def test_sampling_never_uses_future_frame():
    rows,_=example();sample=causal_sample(rows,.764)
    assert [r["frame"] for r in sample]==[0,7,15,22]


def test_identity_model_does_not_pass_improvement_gate():
    rows,events=example();r=report(rows,events)
    result=semantic_gates({"dense":r,"low_0":r},{"dense":r,"low_0":r})
    assert not result["passed"]
    assert not result["checks"]["boundary_gain_8pp"]
