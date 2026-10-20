"""Frozen batched Skywork scorer; raw conversations bypass prompt parsing."""
import json
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from dataset.lm_dataset import RLAIFDataset

class GRPOPromptDataset(RLAIFDataset):
    def __getitem__(self,index):
        item=super().__getitem__(index)
        item['messages_json']=json.dumps(self.samples[index]['conversations'][:-1],ensure_ascii=False)
        return item

class BatchedRewardModel:
    def __init__(self,path,device='cuda',batch_size=8):
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to(device).eval().requires_grad_(False)
        self.device=device;self.batch_size=batch_size;self.last_stats={}
    @torch.inference_mode()
    def score_batch(self,conversations,responses):
        assert len(conversations)==len(responses)
        texts=[]
        for messages,response in zip(conversations,responses):
            messages=[dict(m) for m in messages if m['role']!='system']+[dict(role='assistant',content=response)]
            assert messages[0]['role']=='user' and messages[-1]['role']=='assistant'
            text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
            bos=self.tokenizer.bos_token
            if bos and text.startswith(bos):text=text[len(bos):]
            texts.append(text)
        scores=[]
        for start in range(0,len(texts),self.batch_size):
            inputs=self.tokenizer(texts[start:start+self.batch_size],return_tensors='pt',padding=True,truncation=True,max_length=2048,add_special_tokens=False).to(self.device)
            scores.append(self.model(**inputs).logits[:,0].float())
        result=torch.cat(scores).clone()
        assert torch.isfinite(result).all()
        return result
