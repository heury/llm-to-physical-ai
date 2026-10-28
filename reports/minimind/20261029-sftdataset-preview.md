# SFTDataset 실제 샘플 변환 확인

- 작업 디렉토리: `minimind/`.
- 입력: 대화 중 Hugging Face API로 확인한 `caspar/MiniMind-Mixed-SFT-Training-v1.0`의 train 첫 행 GSM8K messages를 복사한 단일 JSONL. 확인된 저장소 리비전: `542384d27d374d668413d8ade5b9973b7b063bee`. 전체 데이터 다운로드는 하지 않았다.
- 코드: 현재 `dataset/lm_dataset.py`의 실제 `SFTDataset` 생성자와 `__getitem__` 실행.
- 토크나이저: 로컬 `minimind/model/`, 외부 다운로드 없음. 모델 가중치 사용 없음.
- 환경: Python 실행 파일 `/tmp/minimind-v-exercise-venv/bin/python`, torch `2.6.0+cpu`, transformers `4.57.6`, datasets `5.0.1`.
- 설정: `max_length=768`, Python random seed `0`. 이 실행에서는 무작위 system 삽입이 발생하지 않고 빈 think 블록은 제거되었다. 다른 seed에서는 결과가 달라질 수 있다.
- 결과: int64 입력·라벨 각 `[768]`, 유효 입력 149토큰, 패딩 619토큰. 라벨 81~148의 68토큰이 학습 대상이며 assistant 답변, EOS, 뒤 개행을 포함한다. 나머지는 `-100`이다.
- 모델 forward/손실 계산 및 학습은 실행하지 않았다. Mixed SFT 배포본의 바이너리 window와의 바이트 단위 비교도 수행하지 않았다. 이 결과는 현재 기본 SFTDataset에서 로컬 토크나이저로 변환한 예시다.
- 산출물: `outputs/minimind/20261029-sftdataset-preview/`의 `preview.py`, `sample.jsonl`, `preview.json`, `preview.log`. preview.json에 전체 input_ids와 labels를 저장했다.

실행 명령:

```bash
cd minimind
HF_DATASETS_CACHE=../outputs/minimind/20261029-sftdataset-preview/cache \
  /tmp/minimind-v-exercise-venv/bin/python \
  ../outputs/minimind/20261029-sftdataset-preview/preview.py \
  > ../outputs/minimind/20261029-sftdataset-preview/preview.log 2>&1
```
