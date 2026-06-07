# pl_accel_v6 PetaLinux Tools 사용법

이 디렉터리는 `HLS/pl_accel_v6`에서 만든 `full_pose_accel` IP를 PetaLinux 보드 또는 PC에서 테스트하기 위한 코드입니다.

현재 IP는 weight를 bitstream 안에 고정하지 않고, DDR에 올린 `pl_accel_v6_weights.bin`을 AXI 명령으로 PL 내부 weight buffer에 로드한 뒤 추론합니다. 따라서 weight 값만 바뀌는 경우 HLS 재합성 없이 weight 파일만 교체해서 사용할 수 있습니다.

## 파일 구성

| 파일 | 용도 |
|---|---|
| `esp_pose_pl_runner_rt.c` | 보드에서 실행하는 메인 프로그램입니다. ESP CSI serial 입력을 받아 PL IP를 실행하고 pose 24개 float 값을 출력합니다. |
| `pl_accel_v6_weights.bin` | PL에 로드할 통합 weight blob입니다. 현재 IP가 기대하는 형식은 version 2, 105748 words, 422992 bytes입니다. |
| `fpga_view_gui.py` | PC에서 SSH로 보드 runner를 실행하고 pose skeleton을 실시간 표시하는 GUI입니다. |
| `fpga_csv_compare_gui.py` | PC CSV 입력을 PC float32, PC int8, PS C, FPGA PL 결과와 비교하는 GUI/CLI 도구입니다. |
| `ps_pose_infer.c` | 보드 또는 Linux PC에서 PS-only C 추론을 수행하는 검증용 프로그램입니다. |
| `export_ps_model.py` | `ps_pose_infer.c`가 읽을 `ps_model.bin`, `ps_input.bin`을 생성합니다. |

## IP와 맞춰야 하는 값

현재 HLS IP의 AXI-Lite register map은 다음과 같습니다.

| 항목 | offset |
|---|---:|
| `input` | `0x10`, `0x14` |
| `weights` | `0x1c`, `0x20` |
| `command` | `0x28` |
| `final_pose` | `0x30`, `0x34` |

`command` 값은 다음처럼 사용합니다.

| 값 | 동작 |
|---:|---|
| `0` | 일반 inference |
| `1` | DDR의 weight blob을 PL 내부 buffer로 로드한 뒤 종료 |

`esp_pose_pl_runner_rt.c`는 시작할 때 weight 파일을 DDR에 복사하고 `command=1`로 한 번 로드합니다. 이후 inference를 실행할 때는 `command=0`을 사용합니다.

## 보드에서 빌드

보드에 이 디렉터리의 `esp_pose_pl_runner_rt.c`와 `pl_accel_v6_weights.bin`을 복사합니다. 예시는 보드 IP가 `192.168.1.15`, 계정이 `root`, 작업 위치가 `/home/root`인 경우입니다.

===
```bash
ssh-keygen -R 192.168.1.15
```

위의 코드를 터미널에서 써서 SSH host key 초기화
===


```bash
scp esp_pose_pl_runner_rt.c pl_accel_v6_weights.bin root@192.168.1.15:/home/root/
ssh root@192.168.1.15
cd /home/root
gcc -O2 -Wall -Wextra -o esp_pose_pl_runner_rt esp_pose_pl_runner_rt.c -lm
```

보드에서 직접 빌드하지 않고 cross compile을 사용할 수도 있지만, 보드에 `gcc`가 있으면 위 방식이 가장 단순합니다.

## Weight만 먼저 로드

bitstream을 새로 올렸거나 weight 파일을 바꾼 뒤 PL 내부 weight buffer만 갱신하려면 다음을 실행합니다.

```bash
cd /home/root
./esp_pose_pl_runner_rt --weights pl_accel_v6_weights.bin --load-weights-only
```

정상 동작하면 `loaded PL weights` 메시지가 출력됩니다. 일반 runner 실행 시에도 시작 단계에서 weight load를 자동으로 수행하므로, 보통은 별도로 실행하지 않아도 됩니다.

weight를 교체하는 절차는 다음과 같습니다.

1. 새 `pl_accel_v6_weights.bin`을 보드의 runner 실행 위치에 복사합니다.
2. 실행 중인 runner가 있으면 중지합니다.
3. `--load-weights-only`로 확인하거나, 일반 runner를 다시 실행합니다.

weight blob의 layout, version, 크기가 바뀌면 HLS 코드와 PetaLinux 코드의 상수도 함께 맞춰야 합니다. 단순히 weight 값만 바뀌고 blob 형식이 같으면 재합성은 필요 없습니다.

## 실시간 PL runner 실행

ESP가 `/dev/ttyACM0`에 연결되어 있고 Vivado address map이 기본값과 같으면 다음처럼 실행합니다.

```bash
cd /home/root
./esp_pose_pl_runner_rt --tty /dev/ttyACM0 --weights pl_accel_v6_weights.bin --print-every 1
```

기본 물리 주소는 다음과 같습니다.

| 옵션 | 기본값 | 의미 |
|---|---:|---|
| `--ctrl-phys` | `0x40000000` | `full_pose_accel` AXI-Lite base 주소 |
| `--input-phys` | `0x1E000000` | ping-pong input0 DDR buffer |
| `--input1-phys` | `0x1E004000` | ping-pong input1 DDR buffer |
| `--weights-phys` | `0x1E020000` | weight blob DDR buffer |
| `--out-phys` | `0x1E090000` | final pose DDR buffer |

Vivado block design에서 IP base address가 다르면 반드시 `--ctrl-phys`를 바꿔야 합니다. DDR buffer 주소도 보드의 reserved memory 설정과 겹치지 않아야 합니다.

자주 쓰는 옵션은 다음과 같습니다.

```bash
# ESP 설정 명령까지 보낸 뒤 실행
./esp_pose_pl_runner_rt --esp-config --timeout-us 1000 --slot-gap-us 500 --channel 6 --second above

# pose 24개 숫자만 출력해서 다른 프로그램에서 파싱하기 쉽게 실행
./esp_pose_pl_runner_rt --results-only --print-every 1

# N번 추론 후 종료
./esp_pose_pl_runner_rt --max-infer 100 --print-every 1

# 로그를 줄이고 pose 출력도 끄기
./esp_pose_pl_runner_rt --quiet
```

## PC에서 실시간 pose GUI 보기

PC에서 `fpga_view_gui.py`를 실행하면 SSH로 보드에 접속해서 `/home/root/esp_pose_pl_runner_rt`를 실행하고, 출력되는 pose를 실시간으로 그립니다.

```bash
python fpga_view_gui.py --host 192.168.1.15 --user root --remote-dir /home/root --runner ./esp_pose_pl_runner_rt --tty /dev/ttyACM0
```

전제 조건은 다음과 같습니다.

- PC에서 보드로 SSH 접속 가능
- 보드 `/home/root`에 `esp_pose_pl_runner_rt` 실행 파일과 `pl_accel_v6_weights.bin` 존재
- ESP serial 장치가 보드에서 `/dev/ttyACM0` 또는 지정한 `--tty` 경로로 보임
- PC Python에 Tkinter 사용 가능

이미 pose CSV line을 다른 프로그램에서 만들고 있다면 SSH 대신 stdin으로 볼 수 있습니다.

```bash
some_pose_generator | python fpga_view_gui.py --stdin
```

## CSV 기반 PC/PS/FPGA 비교

`fpga_csv_compare_gui.py`는 CSV 데이터셋을 기준으로 다음 결과를 비교합니다.

- PC float32 추론
- PC int8 추론
- PS C float32/int8 추론
- FPGA PL 추론

기본 실행 예시는 다음과 같습니다.

```bash
python fpga_csv_compare_gui.py D:\khu\4-1\project\dev\ML\infer_csv\sync_csi_pose_001.csv --host 192.168.1.15 --user root --remote-dir /home/root --runner ./esp_pose_pl_runner_rt --weights D:\khu\4-1\project\dev\petalinux\tools\pl_accel_v6\pl_accel_v6_weights.bin
```

GUI 없이 결과 파일만 만들려면 `--no-gui`를 붙입니다.

```bash
python fpga_csv_compare_gui.py D:\khu\4-1\project\dev\ML\infer_csv --pattern sync_csi_pose*.csv --windows 100 --no-gui
```

결과는 기본적으로 `outputs` 디렉터리에 저장됩니다.

- `*.npz`: pose 배열, timing 배열, 비교용 원본 데이터
- `*.json`: 평균 latency, MAE, windows/sec 등 요약값

PS-only 비교가 필요 없으면 `--skip-ps`를 붙이면 됩니다.

```bash
python fpga_csv_compare_gui.py D:\khu\4-1\project\dev\ML\infer_csv\sync_csi_pose_001.csv --skip-ps
```

## PS-only C 추론 검증

먼저 PC에서 PS 검증용 binary 입력과 모델 파일을 생성합니다.

```bash
python export_ps_model.py --windows 100 --output-dir ps_export
```

생성되는 주요 파일은 다음과 같습니다.

- `ps_export/ps_model.bin`
- `ps_export/ps_input.bin`

보드 또는 Linux PC에서 `ps_pose_infer.c`를 빌드합니다.

```bash
gcc -O3 -o ps_pose_infer ps_pose_infer.c -lm
```

float32 또는 int8 mode로 실행합니다.

```bash
./ps_pose_infer --mode float32 --model ps_model.bin --input ps_input.bin --output poses_float32.bin --timing timing_float32.bin
./ps_pose_infer --mode int8 --model ps_model.bin --input ps_input.bin --output poses_int8.bin --timing timing_int8.bin
```

이 경로는 PL을 쓰지 않고 CPU에서만 같은 모델 흐름을 검증할 때 사용합니다.

## 문제 확인 순서

PL runner가 동작하지 않으면 다음 순서로 확인합니다.

1. `pl_accel_v6_weights.bin` 크기가 `422992` bytes인지 확인합니다.
2. `--load-weights-only`가 성공하는지 확인합니다.
3. `--ctrl-phys`가 Vivado address map의 `full_pose_accel` base 주소와 같은지 확인합니다.
4. DDR buffer 주소들이 reserved memory 범위 안에 있고 서로 겹치지 않는지 확인합니다.
5. `/dev/ttyACM0`가 실제 ESP serial 장치인지 확인합니다.
6. `--verbose`를 붙여 RX/parser/HLS 진행 로그를 확인합니다.

현재 HLS IP는 이전 pose feedback state register를 사용하지 않습니다. runner 출력의 `pose feedback: v6 fast CNN has no HLS previous-pose state register` 메시지는 정상입니다.
