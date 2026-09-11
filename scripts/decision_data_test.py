"""CPU checks of causal replay, split isolation and S-only localization."""
import os
os.environ.setdefault("JAX_PLATFORMS","cpu")
os.environ.setdefault("HF_HUB_OFFLINE","1")
import unittest
from collections import Counter

import numpy as np
import torch

from openpi.models_pytorch.decision_recurrent import GroundedDecoder,condition_rules,decision_pairs,prefix_weighted_flow
from openpi.models_pytorch.recurrent_subtask import RecurrentSubtaskDecoder
from openpi.models_pytorch.subtask_decoder import SubtaskDecoderConfig
from openpi.training.decision_data import load_assets,DecisionStreamSampler,row_weights,GroundingSamples
from openpi.training.recurrent_sequence import episode_rows


class DecisionTests(unittest.TestCase):
    def test_counterfactual_conditions_and_gates(self):
        vocab=["reach the "+n+" with the "+a+" arm" for n in ["handle of the lid","eggplant","sweet potato"] for a in ["left","right"]]
        prompts=["Put the "+n+" into the box" for n in ["eggplant","sweet potato"]]
        rules,entities=condition_rules(vocab,prompts)
        labels=[vocab[0],vocab[2],vocab[3],vocab[5]];goals=[prompts[0]]*3+[prompts[1]]
        pairs=decision_pairs(labels,goals,labels,["ok"]*4,[False]*4,[True]*4,rules,set(prompts))
        self.assertEqual(Counter(p["kind"] for p in pairs),{"arm":2,"object":2})
        for p in pairs:
            if p["kind"]=="arm":self.assertEqual(p["prompt"],goals[p["index"]])
            else:
                self.assertNotEqual(p["prompt"],goals[p["index"]])
                self.assertEqual(p["label"].rsplit(" with ",1)[1],labels[p["index"]].rsplit(" with ",1)[1])
        for generated,dropped,stable in [([""]*4,[False]*4,[True]*4),(labels,[True]*4,[True]*4),(labels,[False]*4,[False]*4)]:
            self.assertEqual(decision_pairs(labels,goals,generated,["ok"]*4,dropped,stable,rules,set(prompts)),[])

    def test_prefix_weights_and_gradients(self):
        errors=torch.ones(2,50,32,requires_grad=True)
        loss=prefix_weighted_flow(errors);self.assertEqual(float(loss),1.)
        loss.backward();torch.testing.assert_close(errors.grad[:,0],2*errors.grad[:,49])

    def test_grounding_gradients_and_common_initialization(self):
        cfg=SubtaskDecoderConfig(memory_dim=32,embedding_dim=32,width=32,heads=4,layers=1,mlp_dim=64,max_tokens=4)
        torch.manual_seed(42);base=RecurrentSubtaskDecoder(cfg,recurrent=True)
        torch.manual_seed(42);decoder=GroundedDecoder(cfg)
        for name,value in base.state_dict().items():torch.testing.assert_close(value,decoder.state_dict()[name],rtol=0,atol=0)
        memory=torch.randn(4,780,32,requires_grad=True);mask=torch.ones(4,780,dtype=torch.bool)
        points=torch.tensor([[.2,.3],[.8,.5],[.3,.4],[.7,.4]])
        loss,distance,scores=decoder.localization_loss(memory,mask,points)
        loss.backward();self.assertIsNone(memory.grad)
        self.assertGreater(float(decoder.ground_query[1].weight.grad.abs().sum()),0)
        decoder.zero_grad(set_to_none=True)
        projected,full_mask,carry=decoder.compose_sequence(memory,mask,unroll=4)
        self.assertEqual(projected.shape,(4,785,32));self.assertEqual(carry.shape,(1,4,32))
        changed=memory.detach().clone();changed[3]+=7
        before=projected.detach().clone();after=decoder.compose_sequence(changed,mask,unroll=4)[0]
        torch.testing.assert_close(before[:3],after[:3],rtol=0,atol=0)
        self.assertTrue(torch.isfinite(scores).all())

    def test_real_assets_sampler_and_auxiliary_split(self):
        from openpi.models.pi0_config import Pi0Config
        from openpi.training.reach_arm_data import data_config
        from openpi.training.decoded_video_cache import create_cached_dataset
        from openpi.training.subtask_batch import SubtaskTrainingDataset
        decisions,points,ready=load_assets()
        dc=data_config(Pi0Config(pi05=True));raw=create_cached_dataset(dc,50,".stage1_staging/piper_rgb224_reach_arm_v1")
        rows=episode_rows(raw.hf_dataset);dataset=SubtaskTrainingDataset(raw,dc)
        make=lambda start:DecisionStreamSampler(rows,decisions,steps=32,start=start)
        all_batches=list(make(0));self.assertEqual(all_batches[11:],list(make(11)))
        ep=np.asarray(raw.hf_dataset["episode_index"]);frame=np.asarray(raw.hf_dataset["frame_index"])
        prior=[None]*8
        for batch in all_batches:
            for i,(index,reset) in enumerate(batch):
                stream=i%8
                if reset:self.assertEqual(int(frame[index]),0)
                else:
                    self.assertEqual(ep[index],ep[prior[stream]])
                    self.assertGreater(frame[index],frame[prior[stream]])
                prior[stream]=index
        initial=all_batches[0][:8];lookup={r["episode"]:r for r in decisions}
        keys={(lookup[int(ep[i])]["prompt"],lookup[int(ep[i])]["layout"],lookup[int(ep[i])]["lid_arm"]) for i,_ in initial}
        self.assertEqual(len(keys),8)
        ground=GroundingSamples(dataset,points)
        self.assertEqual(len(ground.samples),117)
        self.assertTrue({r["episode"] for r in ground.samples}<={r["episode"] for r in decisions if r["split"]=="train"})
        batch,xy=ground.batch(ground.select(3,0),"cpu")
        self.assertEqual(tuple(xy.shape),(2,2));self.assertEqual(batch.actions.shape,(2,50,32))
        self.assertTrue(all(k not in vars(batch.observation) for k in ["points","labels","layout"]))
        print("Verified 80 reviewed frames, 117 train points, causal replay, 8 initial strata and native sparse batch")


if __name__=="__main__":
    torch.set_num_threads(4)
    unittest.main()
