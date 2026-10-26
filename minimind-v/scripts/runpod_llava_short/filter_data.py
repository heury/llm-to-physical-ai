"""Auditable conservative short-caption / simple standalone visual QA selection."""
import argparse,collections,hashlib,json,random,re
from pathlib import Path

DENY_Q = re.compile(r'\b(why|might|could|would|likely|probably|infer|inference|reason|reasoning|explain|suggest|suggests|assume|assumption|imagine|hypothetical|purpose|significance|implications|consequences|mood|feel|feeling|relationship|benefit|challenge|potential|possible|possibility|future|before|after|happen next)\b',re.I)
DENY_A = re.compile(r'\b(because|therefore|probably|likely|might|may|could|suggests|possibly|implies|inferred|presumably|perhaps)\b',re.I)
SIMPLE = re.compile(r'^(what\b|which\b|where\s+(is|are)\b|how many\b|is\b|are\b|does\b|do\b|can you (see|identify|name)\b|name\b|identify\b)',re.I)
REFERENCE = re.compile(r'\b(it|its|they|them|their|those|these|that|this|other|another|previous|earlier|aforementioned)\b',re.I)

def normalized(text):return ' '.join(text.casefold().split())
def key(image,q,a):return (Path(image).name,normalized(q.replace('<image>','')),normalized(a))
def sentences(text):
    # Conservative punctuation-based heuristic, not a semantic classifier.
    return [s.strip() for s in re.split(r'[.!?]+["\')\]]*(?:\s+|$)',text.strip()) if s.strip()]
def simple_pair(question,answer):
    q=question.replace('<image>','').strip();a=answer.strip()
    if not q or not a:return 'empty'
    if len(q.split())>30:return 'long_question'
    if not 1<=len(a.split())<=40:return 'long_answer'
    if not 1<=len(sentences(a))<=2:return 'sentence_count'
    if re.search(r'(?:^|\n)\s*(?:[-*•]|\d+[.)])\s+',a):return 'list'
    if not SIMPLE.search(q) or DENY_Q.search(q) or DENY_A.search(a):return 'reasoning_or_nonidentification'
    if re.search(r"without a specific count|cannot|can't|unable to|not possible|difficult to|not clear|unclear|hard to determine",a,re.I):return 'uncertain_answer'
    # Extract standalone turns only; avoid losing an antecedent from a prior turn.
    standalone=re.sub(r'\b(this|that|the)\s+(image|picture|photo|photograph|scene)\b','image',q,flags=re.I)
    if REFERENCE.search(standalone):return 'context_dependent'
    if q.count('?')>1:return 'multiple_questions'
    return None

def select(source,out):
    out.mkdir(parents=True,exist_ok=True)
    raw=json.loads((source/'blip_laion_cc_sbu_558k.json').read_text())
    eligible=[i for i,r in enumerate(raw) if len(r['conversations'])==2 and 0<len(r['conversations'][1]['value'].split())<20]
    assert len(eligible)>=50000
    selected=sorted(random.Random(42).sample(eligible,50000))
    pt=[dict(raw[i],source_row=i) for i in selected]
    (out/'pretrain.json').write_text(json.dumps(pt,ensure_ascii=False))
    complex_rows=json.loads((source/'complex_reasoning_77k.json').read_text())
    complex_pairs={key(r['image'],r['conversations'][j]['value'],r['conversations'][j+1]['value']) for r in complex_rows for j in range(0,len(r['conversations'])-1,2)}
    kept=[];seen=set();excluded=collections.Counter();counts={}
    for filename in ['conversation_58k.json','detail_23k.json']:
        rows=json.loads((source/filename).read_text());before=len(kept)
        for row_i,row in enumerate(rows):
            turns=row['conversations'];assert len(turns)%2==0
            for j in range(0,len(turns),2):
                assert turns[j]['from']=='human' and turns[j+1]['from']=='gpt'
                q=turns[j]['value'].replace('<image>','').strip();a=turns[j+1]['value'].strip()
                reason=simple_pair(q,a);k=key(row['image'],q,a)
                if reason:excluded[reason]+=1;continue
                if k in complex_pairs:excluded['matches_complex_reasoning']+=1;continue
                if k in seen:excluded['duplicate_image_qa']+=1;continue
                seen.add(k)
                kept.append(dict(id=f'{filename}:{row_i}:{j//2}',image=row['image'],source_component=filename,source_record_id=str(row['id']),source_row=row_i,source_turn=j//2,conversations=[{'from':'human','value':'<image>\n'+q},{'from':'gpt','value':a}]))
        counts[filename]=dict(source_conversations=len(rows),selected_pairs=len(kept)-before)
    assert kept and all(simple_pair(r['conversations'][0]['value'],r['conversations'][1]['value']) is None for r in kept)
    (out/'sft.json').write_text(json.dumps(kept,ensure_ascii=False))
    report=dict(seed=42,pretrain=dict(source_rows=len(raw),eligible_under_20_words=len(eligible),selected_rows=len(pt),max_caption_words=max(len(r['conversations'][1]['value'].split()) for r in pt)),sft=dict(complex_source_rows_excluded=len(complex_rows),components=counts,selected_pairs=len(kept),unique_images=len({r['image'] for r in kept}),max_answer_words=40,max_question_words=30,answer_sentences='1 or 2; punctuation heuristic',excluded=dict(excluded)),note='Standalone extracted QA turns; original answer text retained. Conservative rule-based visual identification filter, not human semantic labeling. Image-group heldouts are split after selection, so training row counts are smaller than selected counts.',hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'pretrain.json',out/'sft.json']})
    (out/'filter_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();select(args.source,args.out)
