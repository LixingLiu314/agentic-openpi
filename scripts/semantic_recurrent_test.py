"""Focused loss-direction and supervision-gating checks; no GPU or robot I/O."""
import unittest
import torch
from openpi.models.subtask_tokenizer import SubtaskTextCodec
from openpi.models_pytorch.semantic_recurrent import eligible_pairs, ranking_loss, semantic_rules, stable_training_frames


class SemanticTests(unittest.TestCase):
    def test_actor_and_object_tokens_receive_weight_without_changing_labels(self):
        codec=SubtaskTextCodec()
        labels=[f"reach the {obj} with the {arm} arm" for obj in ["eggplant","sweet potato"] for arm in ["left","right"]]
        weights,negatives,audit=semantic_rules(codec,labels)
        for label in labels:
            ids,mask=codec.targets([label])
            self.assertEqual(weights[tuple(ids[0])][int(mask.sum())-1],1.)
            pieces=audit[label]["important_pieces"]
            self.assertTrue(any("left" in p or "right" in p for p in pieces),pieces)
            self.assertTrue(all(x in labels and x!=label for x in negatives[label]))

    def test_no_auxiliary_supervision_for_wrong_empty_dropped_or_invalid_generation(self):
        negative={"reach left":["reach right"]}
        for text,status,drop in [("reach right","ok",False),("","ok",False),("reach left","truncated",False),("reach left","ok",True)]:
            self.assertEqual(eligible_pairs(["reach left"],[text],[status],[drop],negative),[])
        self.assertEqual(eligible_pairs(["reach left"],["reach left"],["ok"],[False],negative),[(0,"reach right")])

    def test_ranking_gradient_and_margin_stop(self):
        positive=torch.tensor([.03],requires_grad=True)
        negative=torch.tensor([.035],requires_grad=True)
        ranking_loss(positive,negative).backward()
        self.assertGreater(positive.grad.item(),0)
        self.assertLess(negative.grad.item(),0)
        self.assertEqual(ranking_loss(positive,torch.tensor([.1])).item(),0.)

    def test_boundaries_are_excluded_without_crossing_episodes(self):
        table=dict(episode_index=[0]*8+[1]*3,frame_index=list(range(8))+list(range(3)),subtask=["a"]*4+["b"]*4+["a"]*3)
        stable=stable_training_frames(table,horizon=3,past=1)
        self.assertEqual(stable,{(0,0),(0,1),(0,5),(0,6),(0,7),(1,0),(1,1),(1,2)})
        self.assertEqual(eligible_pairs(["x"],["x"],["ok"],[False],{"x":["y"]},stable=[False]),[])

    def test_pair_limit_and_rotation(self):
        pairs=eligible_pairs(["x"]*32,["x"]*32,["ok"]*32,[False]*32,{"x":["y"]},offset=30)
        self.assertEqual([i for i,_ in pairs],[30,31,0,1])


if __name__=="__main__":unittest.main()
