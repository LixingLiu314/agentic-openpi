"""Meaningful orchestration guards: failed quality, PID reuse and exact resume."""
import json
import numpy as np
import pytest
import torch
from compare_temporal_resume import equal
import queue_temporal_experiment as queue
from run_temporal_experiment import candidate_record


def checkpoint(tmp_path):
    directory=tmp_path/'step_001000';directory.mkdir()
    (directory/'metadata.json').write_text(json.dumps({'engineering_smoke':False,'temporal_sha256':'weights',
             'config':{'parent_weights_sha256':'parent','protocol_sha256':'protocol'}}))
    return directory


def test_failed_or_engineering_evidence_cannot_qualify(tmp_path):
    c=checkpoint(tmp_path);semantic={'passed':True};action={'passed':True,'engineering_smoke':False}
    native={'passed':True};stress={'passed':True,'engineering_smoke':False,'observations':99}
    assert candidate_record(c,semantic,action,native,stress)['status']=='qualified_offline_candidate'
    assert candidate_record(c,{'passed':False},action,native,stress)['status']=='failed_candidate'
    assert candidate_record(c,semantic,{**action,'engineering_smoke':True},native,stress)['status']=='failed_candidate'
    assert candidate_record(c,semantic,action,native,{**stress,'observations':3})['status']=='failed_candidate'
    assert candidate_record(c,semantic,action,native,{**stress,'passed':False})['status']=='failed_candidate'


def test_pid_reuse_and_incomplete_cache_do_not_launch(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);cache=tmp_path/'cache';cache.mkdir()
    (tmp_path/'cache_queue.process.json').write_text(json.dumps({'pid':1,'created':123.}))
    calls=[]
    def alive(pid,created):calls.append((pid,created));return False
    monkeypatch.setattr(queue.gpu_reservation,'alive',alive)
    with pytest.raises(RuntimeError,match='without complete'):queue.dependency_state(tmp_path,cache)
    assert calls==[(1,123.)]
    (cache/'cache_complete.json').write_text(json.dumps({'frames':{'train':106079,'val':13515},'episodes':178}))
    monkeypatch.setattr(queue.gpu_reservation,'active_job',lambda _:True)
    assert queue.dependency_state(tmp_path,cache)=='waiting_for_managed_job'
    monkeypatch.setattr(queue.gpu_reservation,'active_job',lambda _:False)
    assert queue.dependency_state(tmp_path,cache)=='ready'
    (tmp_path/'cache_failure.json').write_text('{}')
    with pytest.raises(RuntimeError,match='failed'):queue.dependency_state(tmp_path,cache)


def test_resume_comparison_checks_optimizer_and_rng_values():
    state={'optimizer':{'momentum':torch.tensor([1.,2.])},'rng':(np.array([4,5]),42)}
    equal(state,{'optimizer':{'momentum':torch.tensor([1.,2.])},'rng':(np.array([4,5]),42)})
    with pytest.raises(AssertionError,match='Tensor mismatch'):
        equal(state,{'optimizer':{'momentum':torch.tensor([1.,3.])},'rng':(np.array([4,5]),42)})
    with pytest.raises(AssertionError,match='Array mismatch'):
        equal(state,{'optimizer':{'momentum':torch.tensor([1.,2.])},'rng':(np.array([4,6]),42)})
