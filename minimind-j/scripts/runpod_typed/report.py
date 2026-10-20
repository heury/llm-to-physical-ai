"""Build a concise Korean report from verified downloaded training artifacts."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    root=parser.parse_args().directory
    report=json.loads((root/'report.json').read_text())
    data=json.loads((root/'dataset_manifest.json').read_text())
    cleanup=json.loads((root/'cleanup.json').read_text())
    assert cleanup['deleted'] and cleanup['readback_http']==404
    test=report['test']; prior=report['train_only_prior_test']
    text=[
        '# MiniMind-J · Typed Decisions 학습 결과', '',
        'B300 MIG 34GB에서 기존 MiniMind SFT 768(8 layers)을 초기값으로 full finetuning했습니다. '
        '공통 입력 하나에 질문 5개를 묶어 기존 LM head로 동시에 처리합니다.', '',
        f"- 데이터: [{data['dataset']}](https://huggingface.co/datasets/{data['dataset']}) (`all`), revision `{data['revision']}`.",
        f"- 학습 {data['records']['train']:,}건 / 검증 {data['records']['validation']:,}건 / 공식 test {data['records']['test']:,}건.",
        '- 검증 데이터는 공식 train에서 workflow별로 분리했습니다. 공식 test와 id/state 중복이 없습니다.',
        '- 선택지 설명과 점수 rubric을 보존하고 정답 확률분포에 soft cross-entropy를 적용했습니다.',
        f"- {report['epochs_run']} epochs, {report['optimizer_steps']:,} optimizer updates. 검증 KL로 선택한 최적 epoch: **{report['best_epoch']}**.",
        f"- 파라미터 {report['parameters']:,}개. 전체 실행 {report['total_seconds']/60:.1f}분. 최대 GPU allocated 메모리 {report['gpu_peak_bytes']/1024**3:.2f} GiB.", '',
        '## 공식 test 결과', '',
        '| 지표 | 모델 | 학습 데이터만 사용한 prior baseline |',
        '| --- | ---: | ---: |',
        f"| 정확도 | {test['accuracy']:.2%} | {prior['accuracy']:.2%} |",
        f"| KL(gold ∥ prediction) ↓ | {test['kl_from_gold']:.4f} | {prior['kl_from_gold']:.4f} |",
        f"| Brier ↓ | {test['brier']:.4f} | {prior['brier']:.4f} |",
        f"| ECE ↓ | {test['ece']:.4f} | {prior['ece']:.4f} |", '',
        '| 질문 타입 | 질문 수 | 정확도 | KL | Brier |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for kind,values in test['by_type'].items():
        text.append(f"| {kind} | {values['decisions']:,} | {values['accuracy']:.2%} | {values['kl_from_gold']:.4f} | {values['brier']:.4f} |")
    text += ['', '정확도는 첫 argmax 기준이며, tie-aware 정확도는 '
             f"{test['tie_aware_accuracy']:.2%}입니다. Brier는 선택지별 제곱 오차의 합, ECE는 10개 bin의 hard accuracy 기준입니다. "
             '학습에 사용한 벤치마크의 specialist 결과이며 zero-shot 모델 결과와 구분해야 합니다.', '',
             '## 저장과 Pod 정리', '',
             '- 모델·tokenizer: `/shared-volume/minimind-j-typed-decisions-20261020/model/`.',
             '- 원본 데이터, 변환 데이터, 정확한 실행 소스, 설정, 학습 기록, test 예측과 파일 해시도 같은 상위 경로에 보관했습니다.',
             f"- 모델 SHA256: `{cleanup['verification']['remote']['model_sha256']}`.",
             f"- shared-volume 파일 {cleanup['verification']['remote']['verified_files']}개를 재읽어 해시 검증했습니다. "
             f"가중치를 제외한 로컬 파일 {cleanup['verification']['local_verified_files']}개도 전송 후 검증했습니다.",
             '- 저장 모델 strict 재로딩 후 logits가 정확히 일치했고 원본 SFT 가중치 해시가 변하지 않았습니다.',
             f"- 이번 Pod `{cleanup['pod_id']}`는 **삭제했고 GET 404로 재확인**했습니다. 기존 shared-volume은 유지했습니다.", '',
             '가중치는 shared-volume에 보관하며 로컬에는 작은 보고서·소스·데이터·tokenizer를 내려받았습니다. '
             '상세 값은 `report.json`, 전체 예측은 `test_predictions.jsonl`, 검증 기록은 `transport-verification.json`과 `cleanup.json`을 참고하세요.', '']
    (root/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    print(root/'REPORT.md')


if __name__=='__main__':
    main()
