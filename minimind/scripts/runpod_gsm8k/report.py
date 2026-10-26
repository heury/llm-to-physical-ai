"""Independently re-score saved generations, then write a reviewable report."""
import json,sys
from pathlib import Path
from common import extract
REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'outputs/minimind/gsm8k-sft-20261024'
def main():
 prepared=json.loads((OUT/'prepared.json').read_text());models={}
 for kind in ['dense','moe']:
  folder=OUT/kind;data={}
  for phase in ['before','after']:
   metrics=json.loads((folder/f'tests_{phase}.json').read_text())
   rows=json.loads((folder/f'generations_{phase}.json').read_text());assert len(rows)==1319
   assert sorted(r['index'] for r in rows)==list(range(1319))
   for row in rows:
    strict=extract(row['response']);flex=extract(row['response'],False)
    assert strict==row['strict_prediction'] and flex==row['flexible_prediction']
    assert (strict==row['gold'])==row['strict_correct']
    assert (flex==row['gold'])==row['flexible_correct']
   assert sum(r['strict_correct'] for r in rows)==metrics['strict_correct']
   assert sum(r['flexible_correct'] for r in rows)==metrics['flexible_correct']
   assert metrics['validation']['assistant_target_tokens']==prepared['splits']['test']['assistant_target_tokens']
   data[phase]=dict(metrics=metrics,rows=rows)
  assert all((b['index'],b['question'],b['gold'])==(a['index'],a['question'],a['gold']) for b,a in zip(data['before']['rows'],data['after']['rows']))
  data['completion']=json.loads((folder/'completion.json').read_text())
  data['cleanup']=json.loads((folder/'cleanup_verified.json').read_text());assert data['cleanup']['confirmed_absent']
  server=json.loads((folder/'wandb_server_verified.json').read_text());assert json.loads(server['summaryMetrics'])['complete'] and server['state']=='finished'
  models[kind]=data
 assert all(d['after']['metrics']['revision']==prepared['revision'] for d in models.values())
 assert all((d['question'],d['gold'])==(m['question'],m['gold']) for d,m in zip(models['dense']['after']['rows'],models['moe']['after']['rows']))
 lines=['# GSM8K SFT 결과','', 'UltraChat SFT 완료 체크포인트를 GSM8K main 학습7473문제로 2epoch 전체 SFT한 결과.','공식 테스트1319문제 전체를 동일한 zero-shot greedy 조건으로 학습 전/후 평가했다.','', '| 모델 | 최종답 형식 엄격 정확도 전→후 | 마지막 숫자 보조 정확도 전→후 | 정답 토큰 PPL 전→후 |','|---|---|---|---|']
 for kind,data in models.items():
  b=data['before']['metrics'];a=data['after']['metrics']
  fmt=lambda n:f'{n}/1319 ({n/1319:.2%})'
  lines.append(f"| {kind} | {fmt(b['strict_correct'])} → {fmt(a['strict_correct'])} | {fmt(b['flexible_correct'])} → {fmt(a['flexible_correct'])} | {b['validation']['perplexity']:.4f} → {a['validation']['perplexity']:.4f} |")
 lines+=['','엄격 지표: 유효한 `#### 숫자` 최종 줄의 숫자를 Decimal로 정규화해 비교.','보조 지표: 유효한 최종답 줄이 없으면 마지막 숫자를 사용. 두 지표 모두 중간 풀이의 정확성을 보장하지 않는다.','공식 리더보드와 다른 프롬프트·문맥 제한을 사용한 실험이며 직접 비교할 수 없다.','', '## 실행 및 제한','', f"데이터 revision: `{prepared['revision']}`.",f"학습 {prepared['splits']['train']['rows']}문제 / {prepared['splits']['train']['windows']} window, assistant 정답 토큰 {prepared['splits']['train']['assistant_target_tokens']:,}개.",f"검증 PPL은 테스트 전체 {prepared['splits']['test']['assistant_target_tokens']:,}개 정답 토큰을 사용했다.",'최대 길이340, stride256, overlap84로 정답 토큰을 정확히 한 번씩 포함했다. 긴 풀이의 continuation window는 전체 질문을 포함하지 않을 수 있다.','생성은 입력+출력340토큰 이내, 신규토큰 최대256. 질문은 자르지 않는다. 길이 초과·형식 실패도1319분모에 포함한다.','BF16, AdamW 기본 weight_decay0.01, LR5e-5, effective batch128, gradient clip1, seed42. 기존6400토크나이저 유지. 테스트 데이터로 체크포인트를 선택하지 않았다.','최종 저장 가중치를 다시 로드해 평가했으며, 유한성·strict reload·resume와 가중치 일치 및 예상 update 수를 검증했다.','']
 for kind,data in models.items():
  a=data['after']['metrics'];c=data['completion'];s=json.loads((OUT/kind/'training_settings.json').read_text())
  lines+=[f"### {kind}",'',f"학습시간 {c['training_seconds']/60:.2f}분, optimizer updates {c['optimizer_updates']}, 실제 epoch {c['epochs']}.",f"최종답 형식 출력 {a['valid_final_format']}/1319, 문맥 초과 {a['context_overflow']}, EOS 전 토큰 한도 도달 {a['token_budget_exhausted']}.",f"W&B: https://wandb.ai/mind4u/MiniMind-GSM8K-SFT/runs/{c['wandb_id']}",f"가중치: `{c['weights']}`",f"SHA256: `{c['weights_sha256']}`",f"Pod `{data['cleanup']['pod_id']}` 종료를 새404조회로 확인. Global Volume 보존.",'','기초 계산 별도 probe (표준 벤치마크 아님):','']
  for b,p in zip(data['before']['metrics']['probes'],a['probes']):
   lines += [f"- `{p['prompt']}` / 정답 `{p['gold']}` / 전 {b['correct']}, 후 {p['correct']}",'','```text',p['response'],'```','']
 lines+=['## 고정 샘플 3개: 학습 후 실제 생성문','']
 for i in [0,1,2]:
  q=models['dense']['after']['rows'][i]
  lines += [f"### test index {i}",'',q['question'],'',f"정답: `{q['gold']}`",'']
  for kind,data in models.items():
   row=data['after']['rows'][i]
   lines += [f"**{kind}** (엄격 정답: {row['strict_correct']}, 보조 정답: {row['flexible_correct']})",'','```text',row['response'],'```','']
 lines+=['## 저장 문제 복구 기록','', '최초 MoE 실행은 최종 재개 파일에서 update100이 읽혀 검증을 중단했다. 원본 UltraChat 체크포인트에서 동일한2epoch를 다시 학습했다. 원본 모델의 기존 기준 평가는 재사용했다. 로컬 디스크에서 최종126update·가중치/optimizer를 검증한 뒤 공유 볼륨의 새 verified 파일로 복사하고 SHA256 일치를 확인했다. 최종 완료 기록은 이 검증된 재실행을 가리킨다. Dense는 최초 실행의 검증을 통과했다.','']
 lines+=['전체 문제별 생성문은 각 모델의 `generations_before.json`, `generations_after.json`에 보존했다.','모든 채점을 로컬에서 다시 계산해 저장된 지표와 일치함을 확인했다.']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
 (OUT/'independent_scoring_verified.json').write_text(json.dumps(dict(verified=True,test_rows=1319,models=['dense','moe'],phases=['before','after'],total_generations=5276,wandb_finished=True,pods_absent=True),indent=2))
 print('\n'.join(lines[:10]))
if __name__=='__main__':main()
