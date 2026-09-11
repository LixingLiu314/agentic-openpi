"""Match the current user recipe against the actual repository schedule."""
from train_temporal_subtask import cosine_lr
from openpi.training.optimizer import CosineDecaySchedule

reference=CosineDecaySchedule(warmup_steps=500,peak_lr=2.5e-5,decay_steps=5000,decay_lr=2.5e-6).create()
for step in [0,1,499,500,501,2500,4999,5000]:
    actual=cosine_lr(step)
    expected=float(reference(step))
    assert abs(actual-expected)<2e-11,(step,actual,expected)
print('Official learning-rate schedule equality passed at initial, warmup, midpoint and final boundaries')
