# ESP-RX

ESP-RX는 ESP-IDF 기반의 CSI(Channel State Information) 수신 노드 펌웨어이다. 이 코드를 업로드한 ESP는 `CSI_TX`라는 Wi-Fi SoftAP에 STA 모드로 접속하고, TX 장치가 보내는 제어 신호에 맞춰 CSI를 수집한 뒤 UDP로 TX 쪽에 전송한다.

## 동작 요약

1. 부팅 후 NVS를 초기화하고 Wi-Fi STA 모드로 시작한다.
2. SSID `CSI_TX`에 접속을 시도한다.
   - 기본 채널은 6번이다.
   - 이전에 TX가 채널 변경 명령을 보냈다면 NVS에 저장된 채널 힌트를 먼저 사용한다.
   - 연결이 끊기면 자동으로 재접속을 시도한다.
3. ESP-NOW를 초기화하고 broadcast peer를 등록한다.
4. ESP-NOW로 TX가 보낸 제어 패킷을 수신한다.
   - `ASSIGNMENT`: 이 RX 장치의 슬롯 번호, 전체 RX 수, TX MAC 주소, UDP 전송 간격을 설정한다.
   - `MODE`: CSI 수집 실행/정지 상태와 generation 값을 갱신한다.
   - `CHANNEL`: 새 Wi-Fi 채널을 NVS에 저장하고 재부팅한다.
5. CSI 수신 콜백을 활성화한다.
   - 실행 상태(`running`)일 때만 CSI를 처리한다.
   - CSI 프레임의 송신 MAC이 등록된 TX MAC과 일치할 때만 유효 데이터로 본다.
   - CSI 길이는 최대 512바이트로 제한한다.
6. 수집한 CSI를 UDP 패킷으로 만들어 `192.168.4.1:3333`으로 보낸다.
   - 패킷에는 RX 슬롯 번호, RSSI, CSI 길이, generation, checksum, raw CSI 데이터가 포함된다.
   - RX 슬롯 번호가 0이거나 RX가 1개뿐이면 즉시 전송한다.
   - 여러 RX가 있으면 `slot_index * udp_slot_gap_us`만큼 기다렸다가 전송해 충돌을 줄인다.
7. 수집이 정지된 상태에서는 1초마다 ESP-NOW heartbeat를 broadcast로 보낸다.
   - TX는 이 heartbeat를 통해 RX 노드의 존재, MAC 주소, 슬롯 번호, 실행 상태를 확인할 수 있다.

## 주요 설정값

| 항목 | 값 |
| --- | --- |
| Wi-Fi SSID | `CSI_TX` |
| 기본 채널 | `6` |
| Wi-Fi 대역폭 | HT40 |
| UDP 목적지 | `192.168.4.1:3333` |
| 최대 CSI 길이 | `512` bytes |
| 기본 UDP 슬롯 간격 | `2000` us |
| Heartbeat 주기 | `1000` ms |

## 업로드된 ESP의 역할

이 펌웨어가 올라간 ESP는 독립적으로 데이터를 생성하는 장치가 아니라, TX 장치의 명령을 받아 동작하는 CSI 수신기 역할을 한다. TX가 실행 모드로 전환하고 이 RX에 슬롯을 할당하면, RX는 TX에서 오는 Wi-Fi 신호의 CSI와 RSSI를 캡처해서 TX의 UDP 서버로 전달한다. 여러 RX를 동시에 사용할 경우 각 RX는 할당된 슬롯 시간에 맞춰 UDP를 보내도록 설계되어 있다.

## 빌드

ESP-IDF 환경에서 다음 명령으로 빌드할 수 있다.

```powershell
idf.py build
```

보드에 업로드하고 로그를 보려면 포트를 지정해서 실행한다.

```powershell
idf.py -p COMx flash monitor
```
