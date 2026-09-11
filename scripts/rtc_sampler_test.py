import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("rtc_sampling",ROOT/"src/openpi/models_pytorch/rtc_sampling.py")
rtc=importlib.util.module_from_spec(spec);spec.loader.exec_module(rtc)


class SamplerTests(unittest.TestCase):
    def test_soft_mask_has_frozen_prefix_and_free_tail(self):
        w=rtc.overlap_weights(50,25,8)
        torch.testing.assert_close(w[:8],torch.ones(8))
        torch.testing.assert_close(w[25:],torch.zeros(25))
        self.assertTrue(torch.all(w[8:25]>0))
        self.assertTrue(torch.all(w[8:24]>w[9:25]))

    def test_zero_mask_matches_original_reverse_euler(self):
        noise=torch.randn(1,50,32);target=torch.randn_like(noise);w=torch.zeros_like(noise)
        weight=torch.nn.Parameter(torch.tensor(.3))
        def velocity(x,t):return x*weight+t[:,None,None]
        expected=noise.clone()
        with torch.no_grad():
            for i in range(10):expected-=.1*velocity(expected,torch.tensor([1-i/10]))
        actual=rtc.guided_flow(velocity,noise,target,w)
        torch.testing.assert_close(actual,expected)
        self.assertIsNone(weight.grad)
        self.assertFalse(actual.requires_grad)

    def test_guidance_moves_toward_prior_and_leaves_unmasked_dimensions_free(self):
        noise=torch.zeros(1,50,32);target=torch.ones_like(noise)
        w=torch.zeros_like(noise);w[:,:25,:14]=1
        actual=rtc.guided_flow(lambda x,t:x*0,noise,target,w)
        self.assertLess(float((actual[:,:25,:14]-1).abs().mean()),.02)
        self.assertEqual(float(actual[:,:,14:].abs().max()),0)
        self.assertEqual(float(actual[:,25:].abs().max()),0)

    def test_invalid_overlap_rejected(self):
        for overlap,delay in [(50,1),(0,0),(20,21),(-1,0)]:
            with self.assertRaises(ValueError):rtc.overlap_weights(50,overlap,delay)

    def test_endpoint_jacobian_is_used(self):
        x=torch.zeros(1,1,1);target=torch.ones_like(x)
        # At t=1, v=x makes the estimated endpoint constant (zero): its VJP is
        # zero, unlike simply adding a residual or blending the final actions.
        actual=rtc.guided_flow(lambda x,t:x,x,target,torch.ones_like(x),num_steps=1)
        torch.testing.assert_close(actual,x)


if __name__=="__main__":unittest.main()
