import numpy as np
from openpi.training.transition_sequence import select_observed_history,completion_target


def test_dense_and_low_history_choose_same_observed_times():
    dense=np.arange(91)/30
    ids=select_observed_history(dense,3.)
    assert ids==[21,44,67]
    low=dense[[0,21,44,67]]
    other=select_observed_history(low,3.)
    assert np.array_equal(dense[ids],low[other])


def test_startup_and_quantized_low_rate_are_causal():
    assert select_observed_history([0.,.72],.72)==[None,None,0]
    assert select_observed_history([0.,.033],.033)==[None,None,None]
    assert select_observed_history([0.,1.,2.,3.,4.],3.)==[0,1,2]
    assert select_observed_history([1.,2.],8.)==[None,None,None]


def test_completion_proxy_stays_positive_after_boundary_and_ignores_uncertainty():
    rows=[{'label':'reach' if i<10 else 'grasp' if i<30 else 'move'} for i in range(40)]
    assert completion_target(rows,10,'reach')==-100
    assert completion_target(rows,15,'reach')==1
    assert completion_target(rows,25,'reach')==1
    assert completion_target(rows,25,'grasp')==0
    assert completion_target(rows,35,'reach')==2
