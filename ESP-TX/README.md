# ESP-TX

ESP-TX는 ESP-IDF 기반의 CSI(Channel State Information) 송신/수집 코디네이터 펌웨어이다. 이 코드를 업로드한 ESP는 `CSI_TX`라는 SoftAP를 열고, 여러 ESP-RX 노드를 관리하면서 CSI 수집 cycle을 시작한다. RX들이 측정한 CSI는 UDP로 다시 TX에 들어오며, TX는 이를 cycle 단위로 묶어 USB CDC ACM 포트를 통해 PC/FPGA host로 전달한다.

## 동작 요약

1. 부팅 후 NVS에서 채널, timeout, UDP slot gap, 저장된 RX MAC 순서를 읽는다.
2. TinyUSB CDC ACM을 초기화해 PC/FPGA host와 통신할 USB serial 포트를 연다.
3. Wi-Fi SoftAP를 시작한다.
   - SSID는 `CSI_TX`이다.
   - 기본 채널은 6번, 기본 second channel은 `above`이다.
   - Wi-Fi는 11b/g/n, HT40, trigger 송신 rate는 `WIFI_PHY_RATE_MCS3_LGI`로 설정한다.
4. RX가 SoftAP에 접속하거나 끊기면 MAC 주소와 연결 순서를 관리한다.
5. wait mode에서는 연결된 RX 목록과 설정 상태를 1초마다 USB host로 보낸다.
6. USB host에서 `CMD mode run` 또는 `CMD s` 명령이 오면 running mode로 전환한다.
7. running mode 진입 시 live RX 목록을 고정하고 각 RX에 slot 번호를 ESP-NOW 제어 패킷으로 알려준다.
8. 각 cycle마다 raw 802.11 trigger frame을 broadcast로 송신한다.
9. RX들은 trigger를 보고 CSI를 캡처한 뒤 자신에게 할당된 slot 시간에 맞춰 UDP `3333` 포트로 CSI record를 TX에 보낸다.
10. TX는 모든 RX record를 받거나 timeout이 발생하면 cycle 결과를 하나의 USB binary frame으로 묶어 host에 전송하고 다음 cycle을 바로 시작한다.

## 주요 설정값

| 항목 | 값 |
| --- | --- |
| SoftAP SSID | `CSI_TX` |
| 최대 RX 수 | `8` |
| 기본 Wi-Fi 채널 | `6` |
| Wi-Fi 대역폭 | HT40 |
| Trigger TX rate | `WIFI_PHY_RATE_MCS3_LGI` |
| UDP 수신 포트 | `3333` |
| 최대 CSI 길이 | `512` bytes |
| 기본 slot timeout | `2000` us |
| 기본 UDP slot gap | `2000` us |
| 상태 전송 주기 | `1000` ms |
| RX heartbeat live 기준 | `3000` ms |

## TX-RX 통신 프로토콜

TX와 RX 사이에는 Wi-Fi 연결, ESP-NOW 제어, raw 802.11 trigger, UDP CSI record가 함께 사용된다.

### 1. Wi-Fi 연결

TX는 open SoftAP `CSI_TX`를 열고 RX는 STA로 접속한다. TX는 `WIFI_EVENT_AP_STACONNECTED`와 `WIFI_EVENT_AP_STADISCONNECTED` 이벤트로 RX MAC, 연결 상태, 연결 순서를 추적한다.

### 2. ESP-NOW 제어 패킷

TX는 broadcast ESP-NOW로 `link_ctrl_packet_t`를 보낸다. RX는 checksum이 맞는 패킷만 적용한다.

| 필드 | 크기 | 의미 |
| --- | ---: | --- |
| `magic` | 4 | `0x35544343` (`APP_LINK_CTRL_MAGIC`) |
| `version` | 1 | protocol version, 현재 `1` |
| `msg_type` | 1 | `1=ASSIGNMENT`, `2=MODE`, `3=CHANNEL` |
| `slot_index` | 1 | RX에 부여할 slot 번호 |
| `total_nodes` | 1 | 현재 대상 RX 수 |
| `generation` | 4 | running 세대 번호 |
| `timeout_us` | 4 | RX slot timeout 기준값 |
| `udp_slot_gap_us` | 4 | RX 간 UDP 송신 간격 |
| `target_mac` | 6 | assignment 대상 RX MAC, mode/channel은 0일 수 있음 |
| `tx_mac` | 6 | TX SoftAP MAC |
| `wifi_channel` | 1 | 적용할 Wi-Fi channel |
| `second_channel` | 1 | secondary channel 설정 |
| `running` | 1 | running mode 여부 |
| `reserved` | 1 | 예약 |
| `checksum` | 4 | `checksum` 필드 전까지의 checksum32 |

메시지 의미는 다음과 같다.

| `msg_type` | 이름 | 동작 |
| ---: | --- | --- |
| `1` | `ASSIGNMENT` | 특정 RX에 slot 번호, 전체 RX 수, TX MAC, timeout, UDP gap을 알려준다. |
| `2` | `MODE` | RX의 CSI 수집 실행/정지 상태와 generation을 갱신한다. |
| `3` | `CHANNEL` | RX가 새 채널 힌트를 NVS에 저장하고 재부팅하도록 한다. TX도 설정 저장 후 재부팅한다. |

### 3. RX heartbeat

RX는 wait 상태에서 1초마다 ESP-NOW broadcast heartbeat를 보낸다. TX는 이 패킷으로 RX가 아직 live 상태인지 갱신한다.

| 필드 | 크기 | 의미 |
| --- | ---: | --- |
| `magic` | 4 | `0x35554842` (`APP_LINK_HEARTBEAT_MAGIC`) |
| `version` | 1 | protocol version |
| `rx_index` | 1 | RX slot 번호 |
| `running` | 1 | RX의 running 상태 |
| `reserved` | 1 | 예약 |
| `generation` | 4 | RX가 알고 있는 generation |
| `sta_mac` | 6 | RX STA MAC |
| `tx_mac` | 6 | RX가 알고 있는 TX MAC |
| `checksum` | 4 | `checksum` 필드 전까지의 checksum32 |

### 4. Trigger frame

running mode에서 TX는 매 cycle마다 raw 802.11 data frame을 broadcast로 보낸다. RX는 이 Wi-Fi 프레임을 기준으로 CSI callback을 받는다.

Trigger payload에는 다음 값이 포함된다.

| 필드 | 의미 |
| --- | --- |
| `magic` | `0x35524754` (`APP_TRIGGER_DATA_MAGIC`) |
| `version` | protocol version |
| `frame_kind` | 현재 `1` |
| `generation` | 현재 running generation |
| `trigger_seq` | cycle sequence |
| `tx_timestamp_us` | TX가 trigger를 보낸 시각 |
| `active_nodes` | 이번 running session의 RX 수 |

### 5. UDP CSI record

RX는 CSI를 캡처한 뒤 TX의 UDP port `3333`으로 `udp_csi_packet_header_t + raw CSI`를 보낸다.

| 필드 | 크기 | 의미 |
| --- | ---: | --- |
| `magic` | 4 | `0x35554450` (`APP_LINK_UDP_MAGIC`) |
| `version` | 1 | protocol version |
| `rx_index` | 1 | RX slot 번호 |
| `rssi` | 1 | CSI 프레임 RSSI |
| `reserved` | 1 | 예약 |
| `csi_len` | 2 | 뒤따르는 raw CSI 길이, 최대 512 |
| `generation` | 4 | RX가 측정한 generation |
| `checksum` | 4 | header 일부와 raw CSI를 이어 계산한 checksum32 |
| `raw CSI` | 가변 | CSI byte 배열 |

TX는 `magic`, `version`, `csi_len`, 전체 길이, checksum을 검증하고, `generation`과 `rx_index`가 현재 cycle과 맞는 record만 cycle buffer에 반영한다.

## TX-USB Host 통신 프로토콜

USB host와 TX는 TinyUSB CDC ACM 포트 하나를 사용한다. Host에서 TX로 들어오는 명령은 ASCII line 기반이고, TX에서 host로 나가는 데이터는 binary frame 기반이다.

### Host -> TX ASCII 명령

명령은 반드시 `CMD ` prefix로 시작하고 줄바꿈으로 끝난다.

| 명령 | 의미 |
| --- | --- |
| `CMD s` 또는 `CMD toggle` | wait/run 모드를 토글한다. |
| `CMD mode run` | live RX가 있으면 running mode를 시작한다. |
| `CMD mode wait` | running mode를 중지하고 wait mode로 돌아간다. |
| `CMD timeout_us <500..1000000>` | wait mode에서 slot timeout을 변경하고 NVS에 저장한다. |
| `CMD slot_gap_us <100..100000>` | wait mode에서 RX 간 UDP slot gap을 변경하고 NVS에 저장한다. |
| `CMD save_nodes` | 현재 연결 RX 순서를 NVS에 저장한다. 이후 재연결 시 저장 순서를 우선한다. |
| `CMD status` | status frame을 즉시 host로 전송한다. |
| `CMD channel <1..13> <none|above|below>` | wait mode에서 Wi-Fi 채널을 저장하고 RX에 channel 명령을 보낸 뒤 재부팅한다. |

명령 처리 결과는 ACK binary frame으로 돌아간다.

### TX -> Host binary frame

모든 binary frame은 공통 header 뒤에 payload가 붙는다.

| 필드 | 크기 | 의미 |
| --- | ---: | --- |
| `magic` | 4 | `0x35534943` (`APP_SERIAL_MAGIC`) |
| `version` | 1 | protocol version |
| `frame_type` | 1 | `1=STATUS`, `2=CYCLE`, `3=ACK` |
| `payload_len` | 2 | payload 길이 |
| `frame_seq` | 4 | TX가 증가시키는 serial frame sequence |
| `checksum` | 4 | header의 checksum 전 필드와 payload를 이어 계산한 checksum32 |

Frame type은 다음과 같다.

| `frame_type` | 이름 | payload |
| ---: | --- | --- |
| `1` | `STATUS` | `status_payload_header_t` + `status_node_entry_t[]` |
| `2` | `CYCLE` | `cycle_payload_header_t` + RX별 `cycle_slot_header_t`와 CSI bytes |
| `3` | `ACK` | `ack_payload_t` |

### STATUS payload

wait mode에서 1초마다, 또는 `CMD status` 요청 시 전송된다.

`status_payload_header_t`에는 현재 mode, Wi-Fi channel, connected/live/saved RX 수, timeout, UDP gap, generation, 다음 trigger sequence, USB cycle sequence, trigger 송신 수, timeout 수, TX AP MAC이 들어간다.

각 `status_node_entry_t`에는 slot 번호, 저장 순서, 연결 순서, flags, RX MAC, 마지막 확인 시각, 성공/timeout count가 들어간다. `flags`는 다음 bit를 사용한다.

| flag | 의미 |
| --- | --- |
| `BIT0` | connected |
| `BIT1` | saved |
| `BIT2` | live |

### CYCLE payload

running mode에서 한 cycle이 끝날 때마다 전송된다. 모든 RX record를 받거나 cycle timeout이 발생하면 생성된다.

`cycle_payload_header_t`에는 `uart_seq`, `trigger_seq`, trigger 송신 시각, cycle 완료 시각, slot timeout, active RX 수, 실제 수신 RX 수, timeout 발생 여부가 들어간다.

그 뒤에는 active RX 수만큼 `cycle_slot_header_t`가 붙는다. 각 slot header의 `present`가 `1`이면 바로 뒤에 `csi_len`만큼 raw CSI bytes가 이어진다. `present`가 `0`이면 해당 RX는 이번 cycle에서 timeout 또는 미수신으로 처리된다.

### ACK payload

명령 처리 직후 전송된다. `ok`, 현재 mode, channel, timeout, UDP gap, 사람이 읽을 수 있는 message 문자열을 포함한다.

## 빌드

ESP-IDF 5.5 이상과 `espressif/esp_tinyusb` 컴포넌트를 사용한다.

```powershell
idf.py build
```

보드에 업로드하고 로그를 보려면 포트를 지정해서 실행한다.

```powershell
idf.py -p COMx flash monitor
```
