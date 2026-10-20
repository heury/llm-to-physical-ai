# 실제 시뮬레이터 10 episode 평가

`rollout.py`는 학습 모델을 MuJoCo의 현재 이미지·상태로 실행하는 closed-loop 평가입니다.
원본 demonstration action이나 IK를 정책 제어에 사용하지 않습니다.

- 학습 모델: shared-volume `model/best_vla.pt`, 가중치 파일 SHA 검증.
- 환경: 원본 예제 `ady95/physicalai_tutorial`의 SO101 장면·로봇 helper.
- 모델 자산: MuJoCo Menagerie `robotstudio_so101`; 정확한 commit과 자산 archive 저장.
- 새 큐브 위치: 원본 분포의 radius[.17,.27], angle[-.7,.7], position seed1000.
- 10 episodes, noise seed0..9, 20Hz, 각12simulation seconds.
- 16개 행동을 예측하고 앞4개(0.2s) 실행 후 관측 갱신; Euler10steps.
- 상태는 학습 통계로 정규화, 행동은 역정규화 후 원본 actuator control range만 적용.
- top RGB320×240 및 관절6개·지시문만 정책 입력. 물체좌표는 평가에만 사용.
- 성공: cube center와 box center의 XY 차이가 각5cm 미만, 높이5cm 미만인 상태를
  0.5초 이상 연속 유지하고 마지막 프레임에도 상자 안에 있을 것.
- 모든 에피소드를 보존하며 테스트 결과에 따른 재학습·샘플 선택을 하지 않음.

MIG의 OSMesa 시험 실행은 렌더링이 느려 `rollout10_osmesa_pilot`에 별도로 보존했습니다.
정식10개 평가는 A40 NVIDIA EGL에서 같은 가중치·위치·noise 조건으로 실행합니다.

출력(`/shared-volume/minimind-a-so101-20261020/rollout10`):

- `videos/episode_01.mp4` ~ `episode_10.mp4`: 각12초, 외부 시점+정책 top 카메라.
- `all_10_episodes.mp4`: 10개를 모두 이어 붙인120초 영상.
- `overview.mp4`, `overview.gif`: 10개를 동시에 보는12초 요약.
- `episode_XX.json`, `episode_XX_trajectory.npz`: 결과, 관절·행동·물체 위치 기록.
- `settings.json`: 고정 평가 protocol 및 simulator/model 버전.
- `report.json`, `SHA256.json`, `SUCCESS`: 결과 및 영상 decode·무결성 검증.

동영상들을 로컬로 다운로드하고 원격·로컬 해시를 검증한 뒤 평가 Pod를 제거합니다.
10회 성공률은 작은 표본의 특정 seed 평가이며, 광범위한 일반화 성능을 보장하지 않습니다.
