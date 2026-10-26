import importlib.util
from pathlib import Path
import unittest
import tempfile
import json
from unittest.mock import patch

import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

spec=importlib.util.spec_from_file_location('gsm8k_grpo_run',Path(__file__).with_name('run.py'))
run=importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class RolloutContractTests(unittest.TestCase):
    def test_status_accepts_timestamp_from_metrics(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(run,'ROOT',Path(folder)):
            run.status('training',step=1,timestamp=123.)
            saved=json.loads((Path(folder)/'status.json').read_text())
            self.assertEqual(saved['step'],1)
            self.assertGreater(saved['timestamp'],123.)

    def test_single_update_ratio_has_policy_gradient(self):
        logits=torch.tensor([-2.,-3.],requires_grad=True)
        advantages=torch.tensor([1.,-1.])
        ratio=(logits-logits.detach()).exp()
        loss=-(ratio*advantages).mean()
        loss.backward()
        torch.testing.assert_close(ratio,torch.ones_like(ratio))
        torch.testing.assert_close(logits.grad,-advantages/2)

    def test_first_eos_inclusive(self):
        ids=torch.tensor([[5,2,2,2],[5,6,7,8],[2,2,2,2]])
        expected=torch.tensor([[1,1,0,0],[1,1,1,1],[1,0,0,0]],dtype=torch.bool)
        self.assertTrue(torch.equal(run.mask_completion(ids,2),expected))

    def test_generated_tokens_support_backward_after_clone(self):
        torch.set_num_threads(2)
        model=MiniMindForCausalLM(MiniMindConfig(hidden_size=32,num_hidden_layers=1,
            vocab_size=32,num_attention_heads=4,num_key_value_heads=2,intermediate_size=64,
            max_position_embeddings=32)).eval()
        generated=model.generate(inputs=torch.tensor([[1,4]]),max_new_tokens=2,
            do_sample=False,top_k=0,top_p=1.,temperature=1.)
        self.assertTrue(generated.is_inference())
        outputs=generated.clone()
        self.assertFalse(outputs.is_inference())
        probabilities,_=run.logps(model,outputs,torch.ones_like(outputs),outputs.size(1)-2)
        (-probabilities.mean()).backward()
        self.assertIsNotNone(model.lm_head.weight.grad)

    def test_moe_train_eval_logits_and_router_gradient(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        model=MiniMindForCausalLM(MiniMindConfig(hidden_size=32,num_hidden_layers=1,
            vocab_size=32,num_attention_heads=4,num_key_value_heads=2,intermediate_size=64,
            max_position_embeddings=32,use_moe=True))
        x=torch.tensor([[1,4,5,6,2],[1,7,8,9,2]])
        model.eval()
        with torch.no_grad(): old,_=run.logps(model,x,torch.ones_like(x),3)
        model.train()
        current,aux=run.logps(model,x,torch.ones_like(x),3)
        self.assertGreater(aux.item(),0)
        torch.testing.assert_close(current,old)
        (-current.mean()+aux).backward()
        gate=model.model.layers[0].mlp.gate.weight
        self.assertTrue(torch.isfinite(gate.grad).all())
        self.assertGreater(gate.grad.abs().sum().item(),0)

    def test_likelihood_alignment_and_gradient(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        model=MiniMindForCausalLM(MiniMindConfig(hidden_size=32,num_hidden_layers=1,
            vocab_size=32,num_attention_heads=4,num_key_value_heads=2,intermediate_size=64,
            max_position_embeddings=32)).eval()
        x=torch.tensor([[1,4,5,6,2],[1,7,8,9,2]])
        mask=torch.ones_like(x)
        with torch.no_grad():
            old,_=run.logps(model,x,mask,3)
            expected=model(x).logits[:,:-1].float().log_softmax(-1).gather(-1,x[:,1:,None]).squeeze(-1)[:,-3:]
        current,_=run.logps(model,x,mask,3)
        torch.testing.assert_close(current,expected)
        torch.testing.assert_close((current-old).exp(),torch.ones_like(old))
        advantages=torch.tensor([1.,-1.])[:,None]
        loss=-((current-old).exp()*advantages).mean()
        loss.backward()
        norm=sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        self.assertGreater(norm,0.)


if __name__=='__main__':
    unittest.main()
