# Wi-Fi CSI Based Human Pose Estimation on FPGA

ESP32 다중 수신기로 수집한 Wi-Fi CSI(Channel State Information) 만을 입력으로 사용해 카메라 없이 사람의 2D 자세(12 keypoints)를 추정하는 임베디드 시스템. 추론은 Zybo Z7-20 (Zynq-7020)의 PS+PL 협업 구조에서 수행되며, PL의 HLS 가속기가 INT8 / 고정소수점 연산으로 CNN 추론을 담당한다.

---

## About

기존의 사람 자세 추정 시스템은 대부분 카메라 입력에 의존한다. 그러나 카메라 기반 방식은 **조명 조건의 영향을 받고**, **어두운 환경이나 가림 상황에서 동작이 불안정**하며, **사생활 침해 우려**가 큰 공간(가정, 병원, 화장실, 노인 보호 시설 등)에는 적용하기 어렵다는 한계가 있다.

이 프로젝트는 카메라를 완전히 제거하고, **Wi-Fi 신호가 인체에 의해 반사·산란·차폐되며 만들어지는 채널 변화(CSI)를 직접 입력으로 사용**해 자세를 추정한다. 사람이 움직이면 송신기–수신기 사이의 무선 채널 응답이 미세하게 변하는데, 이 변화 패턴을 신경망에 학습시키면 시각 정보 없이도 신체 관절 위치를 복원할 수 있다. 학습 단계에서만 카메라(MediaPipe Pose Landmarker)를 teacher로 사용해 ground-truth keypoint를 제공하고, **실배포 시에는 Wi-Fi 신호만으로 추론**한다.

전체 시스템은 **PC 서버 없이 임베디드 보드 위에서 완결**되도록 설계했다. 3개의 ESP32 RX 노드가 CSI를 수집하고, coordinator ESP32가 이를 묶어 Zybo Z7-20으로 전달한다. Zynq SoC의 ARM Cortex-A9(PS)은 PetaLinux 위에서 CSI 파싱·윈도잉·INT8 양자화 등 흐름 제어와 전처리를 담당하고, FPGA(PL)는 Vivado HLS로 합성된 CNN 가속기에서 실제 추론 연산을 수행한다. PS와 PL은 AXI 버스로 연결되어 데이터를 주고받으며, 최종 24차원(12 keypoints × x, y) pose vector는 UDP/UART/stdout 중 하나로 GUI에 전달되어 실시간 skeleton으로 시각화된다.

활용 측면에서는 카메라가 부적절하거나 사용할 수 없는 환경 — 어두운 침실에서의 낙상 감지, 욕실에서의 응급 상황 모니터링, 화재로 시야가 차단된 공간에서의 인명 탐지 등 — 에 적용할 수 있는 비접촉·비영상 사람 인식 기술의 임베디드 구현 가능성을 검증한다.

## Overview

3개의 ESP32 RX 노드가 송신 ESP32가 보내는 Wi-Fi 패킷을 받아 OFDM subcarrier별 CSI를 추출하고, UDP로 coordinator ESP32에 전달한다. Coordinator는 모든 RX의 CSI를 한 프레임으로 묶어 USB serial을 통해 Zybo로 보낸다. Zybo PS의 Linux 애플리케이션 `zybo_motion_accel`이 프레임을 수신해 sliding window 입력 텐서(`9 × 128 × 10`)를 구성하고 INT8로 양자화한 뒤, AXI를 통해 PL의 HLS 가속기에 전달한다. PL은 RX별 shared CNN encoder 3개와 MLP regression head를 통과시켜 24차원 pose vector를 출력하고, PS가 이를 후처리하여 외부로 송출한다.

## Features

- **Camera-Free Pose Estimation** — RF 신호만으로 2D pose 추정, 시각 정보 의존 제거
- **PS+PL Co-Design on Zynq-7020** — Linux 기반 I/O·전처리(PS)와 FPGA 기반 가속 추론(PL) 분리
- **Custom HLS Accelerator** — Vivado HLS로 합성된 INT8 / 고정소수점 CNN 가속기
- **Shared-Weight Multi-RX Encoder** — 3개 수신기에 동일 CNN encoder 재사용으로 view-invariant feature 학습
- **Teacher-Student Training** — MediaPipe 12-keypoint를 teacher로 사용, 추론 시 카메라 불필요
- **Multi-Receiver UDP Aggregation** — RX 노드 분산 배치 후 coordinator가 시간 정렬해 단일 프레임 구성

## Hardware

| Component | Role |
|---|---|
| Zybo Z7-20 (Zynq-7020) | 메인 추론 보드, PS(ARM Cortex-A9) + PL(FPGA) |
| ESP32 × 3 (RX nodes) | Wi-Fi CSI 수집, UDP로 coordinator에 전달 |
| ESP32 × 1 (TX / Coordinator) | 송신 + 다중 RX CSI 집계 + Zybo로 serial 전송 |
| USB-Serial 케이블 | Coordinator ESP32 ↔ Zybo |
| Wi-Fi AP (없음) | 별도 AP 불필요, ESP32 간 직접 통신 |

실험 환경: 5.3 m × 3.8 m 실내, RX 3개를 한쪽 벽에 배치, TX 반대편 벽에 배치, 가운데 영역이 sensing zone.

## System Architecture

```
        [TX/Coordinator ESP32]
               | Wi-Fi (broadcast)
   +-----------+-----------+
   |           |           |
[RX1]       [RX2]       [RX3]      ESP32 nodes capture CSI
   |           |           |
   +---+-------+-------+---+
       | UDP   | UDP   | UDP
       v       v       v
       [TX/Coordinator ESP32]      Multi-RX CSI aggregation
               |
               | USB-Serial (CSI cycle frame)
               v
       [Zybo Z7-20]
        |                |
   [ PS (PetaLinux) ] -- AXI -- [ PL (HLS Accelerator) ]
   UART RX                       Shared CNN Enc × 3
   CSI Parse                     Concat (96×8×4)
   Window 9×128×10               Flatten 3072
   INT8 Quantize                 FC 128 → FC 24
        |                                 |
        +---- 24D pose vector <-----------+
                  |
                  | stdout / UDP / UART
                  v
            [PC GUI: real-time skeleton]
```

## Repository Layout

> 소스코드는 정리 중이며, 의도된 디렉터리 구조는 다음과 같다.

```
.
├── esp32/
│   ├── rx_node/                  CSI 수신 펌웨어 (ESP-IDF)
│   └── tx_coordinator/           CSI 집계 + Zybo serial 전송
├── zybo_app/
│   └── zybo_motion_accel/        PetaLinux user-space 애플리케이션
│       ├── main.c
│       ├── csi_parser.c          CSI 프레임 파싱
│       ├── window.c              슬라이딩 윈도우 텐서 구성
│       ├── quantize.c            INT8 양자화
│       └── accel_axi.c           HLS 가속기 AXI 드라이버
├── hls/
│   └── pose_accel/               Vivado HLS 가속기 (C++)
│       ├── pose_accel.cpp        Top-level kernel
│       ├── conv_layer.cpp        CNN encoder
│       ├── mlp_head.cpp          FC regression head
│       └── run_hls.tcl           Synthesis script
├── vivado/
│   └── block_design/             Block design TCL + XDC constraints
├── model/
│   ├── train.py                  Teacher-student CNN 학습 (PyTorch)
│   ├── quantize_calib.py         INT8 calibration
│   └── export_weights.py         가중치를 PL용 C 배열로 export
├── gui/
│   └── skeleton_viewer/          실시간 12-keypoint 시각화
└── Docs/
    └── poster.pdf
```

## CNN Architecture

입력: `9 × 128 × 10` (RX·antenna × subcarrier × time samples)
출력: 24차원 vector (12 keypoints × 2)

| Stage | Layer | Output Shape |
|---|---|---|
| Per-RX Encoder × 3 (shared weights) | Conv → ReLU → Pool | 32 × 8 × 4 |
| Feature Fusion | Concat along channel | 96 × 8 × 4 |
| Flatten | — | 3072 |
| MLP | FC 3072→128 + ReLU + Dropout | 128 |
| Output Head | FC 128→24 | 24 |

학습 전략: Camera로 촬영한 RGB 프레임에 MediaPipe Pose Landmarker를 적용해 12 keypoint를 추출, 동시 수집된 CSI와 쌍을 이루어 supervised regression으로 학습. 추론 시에는 camera 없이 CSI만 입력.

## Build & Flash

### ESP32 RX 노드 (3개)
```bash
cd esp32/rx_node
idf.py set-target esp32
idf.py menuconfig      # Wi-Fi channel, RX node ID 설정
idf.py build flash monitor
```

### ESP32 Coordinator / TX
```bash
cd esp32/tx_coordinator
idf.py set-target esp32
idf.py build flash monitor
```

### HLS 가속기 합성
```bash
cd hls/pose_accel
vitis_hls -f run_hls.tcl       # IP export
# Vivado에서 block design에 IP import → bitstream → XSA export
cd vivado/block_design
vivado -mode batch -source build.tcl
```

### PetaLinux 이미지
```bash
cd zybo_app
petalinux-create -t project --template zynq -n zybo_motion
petalinux-config --get-hw-description=../vivado/build.xsa
petalinux-build
petalinux-package --boot --fsbl --fpga --u-boot --force
```
생성된 `BOOT.BIN`과 `image.ub`를 SD카드에 복사 후 Zybo 부팅.

### 모델 학습 / 양자화 / Export
```bash
cd model
pip install torch torchvision mediapipe numpy
python train.py                  # teacher-student 학습
python quantize_calib.py         # INT8 calibration
python export_weights.py         # hls/pose_accel/ 내 C 헤더 갱신
```

### GUI
```bash
cd gui/skeleton_viewer
python skeleton_viewer.py --source udp --port 5005
```

## Run
```bash
# Zybo 시리얼 콘솔에서
./zybo_motion_accel --output udp --udp-target 192.168.1.100:5005
```

## Communication Protocol

| Link | Direction | Phy | Spec |
|---|---|---|---|
| TX ESP32 → RX ESP32 | TX→RX | Wi-Fi | 802.11 broadcast, fixed channel |
| RX ESP32 → Coordinator | TX→RX | UDP | RX별 CSI 페이로드, RX ID tagging |
| Coordinator → Zybo | TX→RX | USB-Serial | 시간 정렬된 CSI cycle frame |
| PS ↔ PL | bi | AXI4 | Control(Lite) + Stream(Full) |
| Zybo → GUI | TX→RX | UDP / UART | 24-float pose vector |

## Performance

FPGA Implementation Summary (Zynq-7020):

| Metric | Value |
|---|---|
| Clock Frequency | 150.015 MHz |
| WNS (timing slack) | +0.003 ns |
| BRAM utilization | 86.79 % |
| DSP utilization | 32.45 % |
| LUT utilization | 43.26 % |
| FF / Registers | 20.42 % |
| Estimated Power | 2.085 W |

실시간 추론 정확도 (camera 기반 reference 대비):

| Action | Estimation Quality | Reason |
|---|---|---|
| Sitting / Standing | Reliable | 상체·골반 자세 변화가 크고 CSI 특성 뚜렷 |
| Single-leg lift | Reliable | 하반신 움직임이 채널 응답에 크게 반영 |
| Seated trunk flexion | Reliable | 몸통 굴곡으로 인한 강한 CSI 변화 |
| Fine arm pose | Limited | 국소적 변화가 작아 CSI 신호로 분리 어려움 |
| Wrist-level detail | Limited | CSI 공간 해상도 한계 |

Inference backend 비교 (동일 모델):

| Backend | 환경 | 특징 |
|---|---|---|
| PC | x86 + Python | 기준 성능 측정용 |
| PS | ARM Cortex-A9 (Zynq) | Linux 위 순수 C 추론 |
| PL | FPGA HLS accelerator | 최저 지연 · 최저 전력 |

## References

1. F. Wang et al., "Person-in-WiFi: Fine-Grained Person Perception Using WiFi," ICCV 2019.
2. Q. Pu, S. Gupta, S. Gollakota, S. Patel, "Whole-Home Gesture Recognition Using Wireless Signals," ACM MobiCom 2013.
3. M. Zhao et al., "Through-Wall Human Pose Estimation Using Radio Signals," CVPR 2018.
4. R. Du, "An Overview on IEEE 802.11bf: WLAN Sensing," IEEE Communications Surveys & Tutorials, 2025.

