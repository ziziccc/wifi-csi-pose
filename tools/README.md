# tools

이 폴더는 ESP-TX에서 USB CDC로 전달되는 CSI 데이터를 읽고, 카메라 기반 pose 데이터를 함께 수집하거나, 저장된 CSI CSV를 시각화하는 Python 도구를 담고 있다.

## 파일 구성

| 파일 | 역할 |
| --- | --- |
| `sync_csi_pose_gui.py` | CSI 수신, pose 추정, CSV 저장을 한 화면에서 제어하는 메인 GUI |
| `sync_csi_input.py` | ESP-TX USB serial binary frame을 읽고 STATUS/CYCLE/ACK/CSI record로 파싱하는 모듈 |
| `sync_pose_input.py` | 로컬/네트워크 카메라를 열고 MediaPipe Pose Landmarker로 주요 관절 좌표를 추출하는 모듈 |
| `csi_iq_heatmap.py` | 저장된 CSV의 `iq_pairs` 컬럼을 읽어 RX별 CSI amplitude heatmap PNG를 생성하는 후처리 스크립트 |
| `models/pose_landmarker_lite.task` | MediaPipe pose 추론 모델 파일 |

## sync_csi_pose_gui.py

CSI와 pose를 동기화해서 CSV로 저장하는 메인 Tkinter GUI이다.

주요 기능:

- ESP-TX의 USB CDC serial 포트에 연결한다.
- TX에 ASCII 명령을 보낸다.
  - `status`
  - `mode run`
  - `mode wait`
  - `save_nodes`
  - `timeout_us <value>`
  - `slot_gap_us <value>`
  - `channel <1..13> <none|above|below>`
- TX가 보내는 binary frame을 받아 현재 TX 상태와 RX 노드 목록을 표시한다.
- 로컬 카메라 또는 네트워크 카메라 URL에 연결한다.
- 카메라 영상에서 MediaPipe pose keypoint를 추출하고 preview에 표시한다.
- CSI record와 가장 가까운 pose frame 구간을 맞춰 CSV로 저장한다.
- 여러 개의 CSV 파일을 일정 시간 단위로 나누어 저장할 수 있다.

기본 저장 위치는 프로젝트 루트의 `captures/` 폴더이다.

실행:

```powershell
python .\tools\sync_csi_pose_gui.py
```

필요 패키지:

```powershell
python -m pip install pyserial mediapipe opencv-python
```

CSV 컬럼:

- CSI: `csi_host_time`, `trigger_seq`, `rx_index`, `rssi`, `csi_len`, `iq_pairs`
- Pose metadata: `frame_width`, `frame_height`
- Pose keypoints: shoulder, elbow, wrist, hip, knee, ankle의 x/y 좌표

## sync_csi_input.py

ESP-TX와 USB serial로 통신하는 입력/파싱 모듈이다. `sync_csi_pose_gui.py`에서 import해서 사용한다.

주요 역할:

- 사용 가능한 serial port 목록을 조회한다.
- ESP-TX로 ASCII 명령을 보낸다. 실제 송신 문자열은 `CMD <명령>\n` 형식이다.
- ESP-TX가 보내는 binary frame stream에서 magic number를 찾아 frame boundary를 맞춘다.
- checksum32를 검증한다.
- frame type별 payload를 Python 객체로 변환한다.

지원하는 TX frame:

| frame type | 의미 | 파싱 결과 |
| ---: | --- | --- |
| `1` | STATUS | mode, channel, timeout, RX node 목록 등 |
| `2` | CYCLE | trigger sequence와 RX별 CSI record |
| `3` | ACK | 명령 처리 결과 |

CSI record는 `ParsedRecord`로 변환된다.

주요 필드:

- `host_time`: PC에서 record를 받은 시각
- `monotonic_ms`: 동기화용 monotonic timestamp
- `uart_seq`: TX가 붙인 cycle sequence
- `trigger_seq`: CSI trigger sequence
- `rx_index`: RX slot 번호
- `rssi`: 수신 RSSI
- `csi_len`: CSI byte 길이
- `iq_pairs`: signed byte CSI payload를 `(I, Q)` 쌍으로 변환한 리스트

## sync_pose_input.py

카메라 입력과 pose 추론을 담당하는 모듈이다. `sync_csi_pose_gui.py`에서 import해서 사용한다.

주요 역할:

- 로컬 카메라 인덱스를 스캔한다.
- 네트워크 카메라 URL 후보를 만든다.
  - 입력 URL에 path가 없으면 `/video`, `/video_feed`, `/mjpeg`, `/stream`, `/live` 등을 후보로 시도한다.
- OpenCV `VideoCapture`로 프레임을 읽는다.
- MediaPipe Pose Landmarker lite 모델을 사용해 pose keypoint를 추론한다.
- preview용 영상에 pose skeleton을 그릴 수 있도록 landmark 좌표를 제공한다.

추출하는 pose keypoint:

- left/right shoulder
- left/right elbow
- left/right wrist
- left/right hip
- left/right knee
- left/right ankle

모델 파일:

- 기본 경로: `tools/models/pose_landmarker_lite.task`
- 파일이 없으면 Google MediaPipe model URL에서 자동 다운로드한다.

## csi_iq_heatmap.py

수집된 CSV 파일의 `iq_pairs` 컬럼을 읽어 CSI amplitude heatmap을 PNG로 저장하는 후처리 스크립트이다.

동작:

1. CSV에서 `rx_index`, `trigger_seq`, `iq_pairs` 컬럼을 읽는다.
2. `iq_pairs`의 각 `(I, Q)`에 대해 `sqrt(I^2 + Q^2)` amplitude를 계산한다.
3. RX별로 행은 CSI pair index, 열은 `trigger_seq`인 matrix를 만든다.
4. 누락된 trigger는 검은색으로 표시한다.
5. 기본값은 `log(1 + amplitude)` scale을 사용한다.

실행 예시:

```powershell
python .\tools\csi_iq_heatmap.py .\captures\sample.csv
```

여러 CSV를 한 번에 처리:

```powershell
python .\tools\csi_iq_heatmap.py .\captures\a.csv .\captures\b.csv
```

옵션:

| 옵션 | 의미 |
| --- | --- |
| `--output-dir <dir>` | PNG 저장 폴더 지정 |
| `--rx-index <n>` | 특정 RX만 렌더링 |
| `--max-rows <n>` | 표시할 trigger row 수 제한 |
| `--linear` | log scale 대신 raw amplitude scale 사용 |
| `--dpi <n>` | 저장 PNG DPI 지정 |

예시:

```powershell
python .\tools\csi_iq_heatmap.py .\captures\sample.csv --rx-index 0 --max-rows 500 --output-dir .\captures\heatmaps
```

## 일반 사용 흐름

1. ESP-TX와 ESP-RX 펌웨어를 업로드한다.
2. ESP-TX를 PC에 USB로 연결한다.
3. GUI를 실행한다.
4. CSI serial port와 카메라를 연결한다.
5. GUI에서 `Run`을 눌러 CSI 수집을 시작한다.
6. `Start Capture`로 CSI + pose CSV 저장을 시작한다.
7. 저장된 CSV를 `csi_iq_heatmap.py`로 시각화한다.
