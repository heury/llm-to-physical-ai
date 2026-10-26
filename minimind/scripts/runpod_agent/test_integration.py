import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from trainer.rollout_engine import SGLangRolloutEngine
from trainer.train_agent import convert_unit, rollout_batch, backward_policy_microbatches


class IntegrationTests(unittest.TestCase):
    def engine(self, response):
        engine=object.__new__(SGLangRolloutEngine)
        engine.base_url='http://localhost';engine.timeout=5
        engine.tokenizer=Mock(eos_token_id=2,pad_token_id=0)
        engine.tokenizer.decode.side_effect=lambda ids,**kw: str(ids)
        engine.http=Mock()
        engine.http.post.return_value.json.return_value=response
        return engine

    def test_sglang_tuple_ids_and_masks(self):
        engine=self.engine([{'meta_info':{'output_token_logprobs':[[-0.4,7,None],[-0.1,2,None]]}},
                            {'meta_info':{'output_token_logprobs':[[-0.2,8,None]]}}])
        result=engine.rollout(torch.tensor([[0,1,3],[1,4,5]]),torch.tensor([[0,1,1],[1,1,1]]),1,8)
        self.assertEqual(result.completion_ids.tolist(),[[7,2],[8,0]])
        self.assertEqual(result.completion_mask.tolist(),[[1,1],[1,0]])
        self.assertEqual(result.prompt_lens.tolist(),[2,3])
        self.assertAlmostEqual(result.per_token_logps[0,0].item(),-.4,places=6)

    def test_missing_logprobs_fails(self):
        engine=self.engine({'output_ids':[7,2],'meta_info':{'output_token_logprobs':[[-.4,7,None]]}})
        with self.assertRaises(RuntimeError):
            engine.rollout(torch.tensor([[1,3]]),torch.ones(1,2,dtype=torch.long),1,8)

    def test_invalid_tool_call_shapes(self):
        from trainer.train_agent import parse_tool_calls
        for payload in ['[]', 'null', '42', '{"name": []}', '{"name":"calculate_math","arguments":[]}']:
            self.assertEqual(parse_tool_calls("<tool_call>" + payload + "</tool_call>"), [])

    def test_temperature_conversion(self):
        self.assertEqual(convert_unit({'value':0,'from_unit':'celsius','to_unit':'fahrenheit'}),{'result':32.0})
        self.assertEqual(convert_unit({'value':212,'from_unit':'fahrenheit','to_unit':'celsius'}),{'result':100.0})
        self.assertEqual(convert_unit({'value':10,'from_unit':'km','to_unit':'miles'}),{'result':6.2137})

    def test_parallel_rollout_order(self):
        from unittest.mock import patch
        def fake(engine,tokenizer,messages,tools,*args):
            return (messages[0]['content'], 'context', [1], [2], [1], [-.2], ['done'], False)
        with patch('trainer.train_agent.rollout_single',side_effect=fake):
            result=rollout_batch(None,None,[[{'content':'a'}],[{'content':'b'}]],[None,None],2,workers=4)
        self.assertEqual(result[0],['a','a','b','b'])
        self.assertEqual(len(result),8)

    def test_microbatch_gradient_matches_full_batch(self):
        from copy import deepcopy
        from contextlib import nullcontext
        from types import SimpleNamespace
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
        torch.manual_seed(7)
        cfg=MiniMindConfig(hidden_size=32,num_hidden_layers=2,vocab_size=32,num_attention_heads=4,
            num_key_value_heads=2,head_dim=8,intermediate_size=64,max_position_embeddings=32,flash_attn=False)
        full=MiniMindForCausalLM(cfg);micro=deepcopy(full);ref=deepcopy(full).eval().requires_grad_(False)
        ids=torch.tensor([[1,4,5,6,2],[1,7,2,0,0]])
        mask=torch.tensor([[1,1,1,1,1],[1,1,1,0,0]])
        response=torch.tensor([[0.,1.,1.,1.],[1.,1.,0.,0.]])
        old=torch.zeros(2,4);advantages=torch.tensor([1.,-1.])
        for kind in ['cispo','grpo']:
            full.zero_grad();micro.zero_grad()
            args=SimpleNamespace(train_microbatch_size=0,loss_type=kind,epsilon_high=5.,epsilon=.2,beta=.1,accumulation_steps=1)
            loss1,_=backward_policy_microbatches(full,ref,ids,mask,response,old,advantages,args,cfg,nullcontext())
            args.train_microbatch_size=1
            loss2,_=backward_policy_microbatches(micro,ref,ids,mask,response,old,advantages,args,cfg,nullcontext())
            self.assertAlmostEqual(loss1,loss2,places=6)
            for a,b in zip(full.parameters(),micro.parameters()):
                torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=1e-6)

    def test_skywork_score_head_and_conversation(self):
        from types import SimpleNamespace
        from transformers import BatchEncoding
        from trainer.trainer_utils import LMForRewardModel
        reward=object.__new__(LMForRewardModel)
        reward.sequence_classifier=True;reward.device='cpu';reward.score_scale=10.0
        reward.tokenizer=Mock()
        reward.tokenizer.apply_chat_template.return_value='rendered'
        reward.tokenizer.return_value=BatchEncoding({'input_ids':torch.tensor([[1,2]])})
        reward.model=Mock(return_value=SimpleNamespace(logits=torch.tensor([[12.0]])))
        result=reward.get_score([{'role':'system','content':'system rules'},{'role':'user','content':'question'}],'answer')
        self.assertAlmostEqual(result,1.2)
        self.assertEqual(reward.last_raw_score,12.0)
        self.assertEqual(reward.tokenizer.apply_chat_template.call_args.args[0],
            [{'role':'user','content':'question'},{'role':'assistant','content':'answer'}])
        self.assertFalse(reward.model.call_args.kwargs['use_cache'])


if __name__=='__main__':unittest.main()
