"""Rollout contracts needed by the GRPO importance ratio and completion loss."""
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
import torch
from trainer.rollout_engine import TorchRolloutEngine

class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.kwargs={}
    def generate(self, input_ids, **kwargs):
        self.kwargs=kwargs
        return torch.cat([input_ids,torch.tensor([[4,2,2]]*len(input_ids),device=input_ids.device)],1)
    def forward(self, input_ids, **kwargs):
        self.forward_mask=kwargs.get('attention_mask')
        n=kwargs.get('logits_to_keep',0) or input_ids.shape[1]
        scores=torch.tensor([0.13,-0.4,-0.79,0.23,0.53],dtype=torch.bfloat16,device=input_ids.device)
        return SimpleNamespace(logits=scores.expand(len(input_ids),n,5))

class RolloutContract(unittest.TestCase):
    def setUp(self):
        self.model=Policy();self.tokenizer=SimpleNamespace(pad_token_id=0,eos_token_id=2,batch_decode=lambda ids,**kwargs:['response']*len(ids))
        self.result=TorchRolloutEngine(self.model,self.tokenizer,'cpu',nullcontext()).rollout(torch.tensor([[1,3]]),torch.ones(1,2,dtype=torch.long),2,3,temperature=1.)
    def test_eos_is_included_and_following_tokens_are_excluded(self):
        self.assertEqual(self.result.completion_mask.tolist(),[[1,1,0],[1,1,0]])
    def test_unwarped_sampling_matches_the_probability_used_in_loss(self):
        self.assertEqual(self.model.kwargs.get('temperature'),1.)
        self.assertEqual(self.model.kwargs.get('top_p'),1.)
        self.assertEqual(self.model.kwargs.get('top_k'),0)
    def test_old_policy_masks_eos_padding_like_current_policy(self):
        self.assertEqual(self.model.forward_mask[:,-3:].tolist(),[[1,1,0],[1,1,0]])
    def test_old_log_probabilities_use_float32(self):
        self.assertEqual(self.result.per_token_logps.dtype,torch.float32)
        scores=torch.tensor([0.13,-0.4,-0.79,0.23,0.53],dtype=torch.bfloat16).float()
        expected=scores.log_softmax(0)[torch.tensor([4,2,2])]
        torch.testing.assert_close(self.result.per_token_logps[0],expected,rtol=1e-6,atol=1e-6)

if __name__=='__main__':unittest.main()
