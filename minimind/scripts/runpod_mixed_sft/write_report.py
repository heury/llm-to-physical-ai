"""Re-score final saved math generations and write the mixed SFT report."""
import json
from pathlib import Path
from common import extract
REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'outputs/minimind/mixed-sft-20261024'
def main():
 data={}
 for kind in ['dense','moe']:
  folder=OUT/kind
  if not (folder/'cleanup_verified.json').exists():return
  data[kind]={}
  for phase in ['before','after']:
   metrics=json.loads((folder/f'tests_{phase}.json').read_text());rows=json.loads((folder/f'generations_{phase}.json').read_text());assert len(rows)==1319 and sorted(r['index'] for r in rows)==list(range(1319))
   assert sum(extract(r['response'])==r['gold'] for r in rows)==metrics['strict_correct']
   assert sum(extract(r['response'],False)==r['gold'] for r in rows)==metrics['flexible_correct']
   data[kind][phase]=metrics
  data[kind]['completion']=json.loads((folder/'completion.json').read_text())
 manifest=json.loads((OUT/'prepared.json').read_text())
 lines=['# 3개 데이터셋 혼합 SFT 결과','', '완료된 문서 전체 사전학습 Dense·MoE에서 각각2epoch 전체 파라미터 SFT.','UltraChat train_sft, SmolTalk all/train, GSM8K main/train을 혼합했다. 공식 테스트는 학습 및 체크포인트 선택에서 제외했다.','', '| 모델 | GSM8K 엄격 정답 수 전→후 | 마지막 숫자 정답 수 전→후 | 혼합 검증 PPL 전→후 |','|---|---|---|---|']
 for k,d in data.items():
  b=d['before'];a=d['after'];lines.append(f"| {k} | {b['strict_correct']}→{a['strict_correct']}/1319 ({a['strict_accuracy']:.2%}) | {b['flexible_correct']}→{a['flexible_correct']}/1319 ({a['flexible_accuracy']:.2%}) | {b['validation']['perplexity']:.4f}→{a['validation']['perplexity']:.4f} |")
 lines+=['', '엄격 지표는 ####숫자 최종 줄을 요구한다. 보조 지표는 그 줄이 없으면 마지막 숫자를 사용한다. 두 지표 모두 풀이의 정확성을 보장하지 않는다.','최대길이340 제한의 custom zero-shot평가다. 다른 리더보드와 직접 비교하지 않는다.','대상별 검증 PPL은 공식 테스트에서 고정 선택한 window의 assistant-only loss다.','', '## 데이터 검증','', '원본 질문·답변은 전체 토큰화했고, 각 assistant target/EOS를 정확히 한 번씩 포함하는지 확인했다.','문맥340보다 긴 대화의 continuation window에는 전체 질문이 없을 수 있다. 마지막 짧은 window는 padding해 EOS를 보존한다.','정규화한 테스트 첫 질문 및 GSM8K테스트 질문의 문자열 출현을 필터링했다. 의미상 바꿔 쓴 문장과 upstream pretrain 오염까지 배제했다는 보장은 아니다.','```json',json.dumps(manifest['stats'],indent=2,ensure_ascii=False),'```']
 for kind,d in data.items():
  c=d['completion'];a=d['after'];lines+=['', '## '+kind,'',f"가중치: `{c['weights']}`",f"SHA256: `{c['weights_sha256']}`",c['wandb_url'],'', '### 소스별 검증 PPL','']
  for source,v in a['validation']['per_dataset'].items():lines.append(f"- {source}: {v['perplexity']:.4f}")
  lines+=['', '### 학습 후 영어 답변 원문','']
  for row in a['english']:lines+=['입력: '+row['prompt']+' / '+row['mode'],'','```text',row['response'],'```','']
 lines+=['주 평가5276개 생성문을 로컬에서 다시 채점해 기록과 일치를 확인했다. 두 Pod는 종료404를 확인했고 Global Volume은 보존했다.']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n');(OUT/'scoring_verified.json').write_text(json.dumps(dict(verified=True,main_generations=5276,models=['dense','moe']),indent=2))
if __name__=='__main__':main()
