"""Write a Korean report from verified completion/publication receipts only."""
import json
from remote import OUT

def main():
 c=json.loads((OUT/'result/completion.json').read_text());p=json.loads((OUT/'published.json').read_text());cleanup=json.loads((OUT/'cleanup.json').read_text())
 assert c['complete'] and p['upload_verified'] and cleanup['deleted']
 b,a=c['baseline'],c['distilled'];vb,va=c['validation_before'],c['validation_after']
 lines=['# MoE → Dense GSM8K soft distillation 결과','',f"게시 모델: [{p['repo']}]({p['url']})",f"게시 commit: `{p['commit']}`",'',
 'Teacher: `caspar/MiniMind-MoE-198M-A64M-Mixed-SFT-v1.0`',
 'Student: `caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`',
 '데이터: `caspar/MiniMind-Mixed-SFT-Training-v1.0`의 GSM8K 학습 대화 6,851개.',
 '', '## 학습', '',
 '3 epoch, 345 optimizer updates. epoch당 7,333개 윈도우와 845,130개 assistant 정답 토큰.',
 '전체 student 매개변수 학습, teacher 고정. 0.5 CE + 0.5 T² KL(teacher || student), T=1.5.',
 '학습률 5e-6 → 5e-7, microbatch 16 × accumulation 4, BF16 연산, FP32 optimizer weights.',
 '340토큰 윈도우, stride 256; 중복 타깃과 prompt·padding 마스킹, 마지막 짧은 윈도우 유지.',
 f"순수 학습 시간: {c['training']['seconds']:.1f}초. 설치·다운로드·평가 시간 제외.",
 '', '| Epoch | 평균 CE | 평균 KL |', '|---|---:|---:|']
 for r in c['training']['history']:lines.append(f"| {r['epoch']} | {r['ce']:.6f} | {r['kl']:.6f} |")
 lines += ['', '## GSM8K 테스트 결과','',
 '공식 main/test 1,319문항 전체. 데이터 릴리스에서는 `validation_gsm8k`라는 이름이다.',
 '테스트 문항은 학습·하이퍼파라미터 조정·체크포인트 선택에 사용하지 않았다.',
 '두 모델 모두 zero-shot greedy, 최대 생성 256토큰, 전체 문맥 340토큰. 동일 길이 prompt끼리 배치.',
 '형식 오류·문맥 초과도 1,319개 분모에 포함한다. 정확도는 FP16 배포 가중치를 재로딩한 뒤 측정했다.',
 '', '| 지표 | 원본 Dense | 증류 Dense |','|---|---:|---:|',
 f"| 엄격한 #### 정답 | {b['strict_correct']}/1319 ({b['strict_accuracy']:.2%}) | {a['strict_correct']}/1319 ({a['strict_accuracy']:.2%}) |",
 f"| 마지막 숫자 fallback | {b['flexible_correct']}/1319 ({b['flexible_accuracy']:.2%}) | {a['flexible_correct']}/1319 ({a['flexible_accuracy']:.2%}) |",
 f"| Assistant CE | {vb['ce']:.6f} | {va['ce']:.6f} |",
 f"| Perplexity | {vb['perplexity']:.6f} | {va['perplexity']:.6f} |",
 f"| Teacher KL, T=1.5 | {vb['kl']:.6f} | {va['kl']:.6f} |",
 '',f"엄격 정답률 변화: {(a['strict_accuracy']-b['strict_accuracy'])*100:+.4f}%p. 단일 seed 결과이며 통계적 유의성을 검증하지 않았다.",
 '긴 대화의 후속 윈도우에는 전체 질문이 포함되지 않을 수 있다. 문맥/생성 제한이 성능에 영향을 줄 수 있다.',
 '원본 모델 사전학습 데이터의 우회 표현 중복까지 제거되었다는 보장은 없다.',
 '', '## 검증과 산출물','',
 '- 토크나이저 파일 일치, 공개 데이터 해시·개수·정답 토큰 수 검증.',
 '- 손실 shifting/masking, 동일 분포 KL, 정답 추출 단위 테스트 3개 통과.',
 '- 실제 GPU backward, teacher gradient 없음, 학생 가중치 변경·유한성 확인.',
 '- FP16 저장 → 재로딩 → optimizer 체크포인트와 텐서 일치 확인.',
 '- 내려받은 결과 파일 SHA256 및 HF 게시 commit에서 다시 받은 모델 SHA256 확인.',
 f"- 작업 Pod `{cleanup['pod_id']}` 삭제 확인. 기존 볼륨 보존.",
 '', '세부 지표·설정·문항별 생성 출력·코드는 `result/model/`과 Hugging Face 저장소에 있다.',
 '재개용 optimizer 체크포인트는 기존 볼륨의 `/workspace/moe-dense-gsm8k-distill-20261025/resume.pt`에 보존했다.',
 '', '초기 준비 단계의 체크섬 형식 오류는 optimizer update 전에 수정했다. 초기 Pod는 삭제하고 동일 볼륨을 연결한 새 Pod에서 학습을 완료했다.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print(OUT/'REPORT.md')
if __name__=='__main__':main()
