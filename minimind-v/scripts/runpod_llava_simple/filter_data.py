"""Deterministic short-caption and simple-dialogue selection; explicit audit rules."""
import collections, hashlib, json, random, re
from pathlib import Path
from transformers import AutoTokenizer
REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'outputs/minimind-v/llava-simple-20261025'
RAW=OUT/'raw'
# A word is a whitespace-delimited unit; punctuation stays attached.
def word_count(text):return len(text.strip().split())
def sentence_count(text):
    # Protect common abbreviations and decimal points before splitting terminators.
    text=re.sub(r'(?<=\d)\.(?=\d)','<DOT>',text.strip())
    text=re.sub(r'\b(?:Mr|Mrs|Ms|Dr|St|e\.g|i\.e)\.',lambda m:m[0].replace('.','<DOT>'),text,flags=re.I)
    return len([s for s in re.split(r'[.!?]+(?:["\u201d\u2019\']*)(?:\s+|$)',text) if s.strip()])
REASONING=re.compile(r'\b(why|might|could|would|likely|infer|inference|reason|reasoning|suggest|purpose|relationship|predict|explain|probably|perhaps|possibly|feel|feeling|opinion|imagine|assume)\b',re.I)
QUESTION=re.compile(r'^(what|which|where|who|how many|is|are|does|do|can you see)\b',re.I)
def simple_pair(question,answer):
    question=question.replace('<image>','').strip();answer=answer.strip()
    if not QUESTION.search(question) or REASONING.search(question+' '+answer):return False
    if not 1<=sentence_count(answer)<=2:return False
    if not 1<=word_count(answer)<=40:return False
    if '\n' in answer or re.search(r'(^|\s)(?:[-*•]|\d+[.)])\s',answer):return False
    return True

def canonical(row):return json.dumps(row['conversations'],sort_keys=True,ensure_ascii=False)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(REPO/'outputs/minimind/hf-release-v1.0/inputs/tokenizer',local_files_only=True)
    captions=json.loads((RAW/'blip_laion_cc_sbu_558k.json').read_text())
    eligible=[r for r in captions if len(r['conversations'])==2 and r['conversations'][1]['from']=='gpt' and 0<word_count(r['conversations'][1]['value'])<20]
    # 6K candidate pool; remote pixel-group checks choose exactly 5K training rows.
    rng=random.Random(42);rng.shuffle(eligible);pool=eligible[:6000]
    conversations=json.loads((RAW/'conversation_58k.json').read_text())
    complex_rows=json.loads((RAW/'complex_reasoning_77k.json').read_text());complex_text={canonical(r) for r in complex_rows}
    counts=collections.Counter();selected=[];seen=set()
    for row in conversations:
        turns=row['conversations']
        if canonical(row) in complex_text:counts['complex_reasoning_match']+=1;continue
        if len(turns)%2 or any(t['from']!=('human' if i%2==0 else 'gpt') for i,t in enumerate(turns)):
            counts['unsupported_roles']+=1;continue
        if not all(simple_pair(turns[i]['value'],turns[i+1]['value']) for i in range(0,len(turns),2)):
            counts['not_simple_or_short']+=1;continue
        msgs=[{'role':'user' if t['from']=='human' else 'assistant','content':t['value'].replace('<image>','<|image_pad|>'*64)} for t in turns]
        text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False,open_thinking=False).replace('<think>\n\n</think>\n\n','')
        ids=tok(text,add_special_tokens=False).input_ids
        if len(ids)>340:counts['over_340_tokens']+=1;continue
        if ids.count(12)!=64:counts['invalid_image_markers']+=1;continue
        key=(row['image'],canonical(row))
        if key in seen:counts['duplicate_dialogue']+=1;continue
        seen.add(key);selected.append(row)
    for name,rows in [('pretrain_candidates',pool),('sft_candidates',selected)]:
        (OUT/f'{name}.json').write_text(json.dumps(rows,ensure_ascii=False))
    report=dict(seed=42,pretrain_source_rows=len(captions),pretrain_under20_eligible=len(eligible),pretrain_candidate_pool=len(pool),pretrain_train_requested=5000,word_rule='whitespace-separated assistant caption words, 1..19',sft_source='conversation_58k.json only; complex_reasoning_77k.json excluded',sft_source_rows=len(conversations),complex_reasoning_rows=len(complex_rows),sft_selected_dialogues=len(selected),sft_unique_images=len({r['image'] for r in selected}),excluded=dict(counts),sft_rule='Every answer 1-2 sentences, <=40 whitespace words; direct visible-identification questions; no speculative/reasoning terms; full conversation <=340 tokens; no answer truncation',semantic_filter='Rule-based heuristic, not a guarantee of semantic simplicity',source_sha256={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in RAW.glob('*.json')})
    (OUT/'filter_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
