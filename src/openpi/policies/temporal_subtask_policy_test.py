import pytest
from openpi.policies.temporal_subtask_policy import SessionGuard


def first(mode='execute'):
    return dict(run_id='run',sequence=1,observation_time=10.,mode=mode)


def test_new_runs_and_action_adoption():
    guard=SessionGuard();session=first()
    assert guard.inspect(session,'task')=='new_run';guard.commit(session,'task','run:1')
    next_session={**session,'sequence':2,'observation_time':10.7,'previous_chunk':{'id':'run:1','executed_steps':15}}
    assert guard.inspect(next_session,'task') is None
    next_session['previous_chunk']['executed_steps']=8
    assert guard.inspect(next_session,'task')=='partial_or_rejected_chunk'


def test_missing_ack_duplicate_and_external_annotation_rejected():
    guard=SessionGuard();s=first();guard.commit(s,'task','run:1')
    with pytest.raises(ValueError,match='duplicate'):guard.inspect(s,'task')
    with pytest.raises(ValueError,match='acknowledgment'):guard.inspect({**s,'sequence':2,'observation_time':10.7},'task')
    with pytest.raises(ValueError,match='annotation'):guard.inspect({**s,'subtask':'grasp'},'task')


def test_offline_mode_is_explicit_and_long_gap_resets():
    guard=SessionGuard();s=first('offline');guard.commit(s,'task','run:1')
    assert guard.inspect({**s,'sequence':2,'observation_time':15.},'task')=='expired_history'
    assert guard.inspect({**s,'sequence':2,'observation_time':10.7},'new task')=='task_changed'
    assert guard.inspect({**s,'run_id':'fresh'},'task')=='new_run'
    with pytest.raises(ValueError,match='mode'):guard.inspect({**s,'sequence':2,'observation_time':11.,'mode':'execute'},'task')
