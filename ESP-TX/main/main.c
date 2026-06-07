#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "driver/gptimer.h"
#include "esp_event.h"
#include "esp_idf_version.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"
#include "nvs.h"
#include "nvs_flash.h"
#include "tinyusb.h"
#include "tinyusb_cdc_acm.h"
#include "tinyusb_default_config.h"

#define APP_WIFI_SSID                        "CSI_TX"
#define APP_WIFI_MAX_CONN                    8
#define APP_WIFI_PROTOCOL_BITMAP             (WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N)
#define APP_DEFAULT_WIFI_CHANNEL             6
#define APP_DEFAULT_SECOND_CHANNEL           WIFI_SECOND_CHAN_ABOVE
#define APP_WIFI_BANDWIDTH                   WIFI_BW_HT40
#define APP_TRIGGER_TX_RATE                  WIFI_PHY_RATE_MCS3_LGI

#define APP_UART_LINE_BUF_SIZE               256
#define APP_UART_CMD_PREFIX                  "CMD "
#define APP_USB_CDC_PORT                     TINYUSB_CDC_ACM_0
#define APP_USB_CHUNK_BYTES                  512u
#define APP_USB_FLUSH_TIMEOUT_MS             1u
#define APP_USB_FLUSH_LOG_INTERVAL           100u

#define APP_MAX_RX_NODES                     8
#define APP_MAX_CSI_LEN                      512
#define APP_MAX_UART_FRAME_PAYLOAD           12288
#define APP_ESPNOW_BROADCAST_RETRY_COUNT     2
#define APP_TRACE_HOT_PATH                   0

#define APP_SERIAL_MAGIC                     0x35534943u
#define APP_LINK_CTRL_MAGIC                  0x35544343u
#define APP_LINK_DATA_MAGIC                  0x35544443u
#define APP_LINK_UDP_MAGIC                   0x35554450u
#define APP_LINK_HEARTBEAT_MAGIC             0x35554842u
#define APP_TRIGGER_DATA_MAGIC               0x35524754u
#define APP_PROTOCOL_VERSION                 1u

#define APP_SERIAL_FRAME_STATUS              1u
#define APP_SERIAL_FRAME_CYCLE               2u
#define APP_SERIAL_FRAME_ACK                 3u

#define APP_CTRL_MSG_ASSIGNMENT              1u
#define APP_CTRL_MSG_MODE                    2u
#define APP_CTRL_MSG_CHANNEL                 3u

#define APP_STATUS_FLAG_CONNECTED            BIT0
#define APP_STATUS_FLAG_SAVED                BIT1
#define APP_STATUS_FLAG_LIVE                 BIT2

#define APP_DEFAULT_SLOT_TIMEOUT_US          2000u
#define APP_DEFAULT_UDP_SLOT_GAP_US          2000u
#define APP_STATUS_INTERVAL_MS               1000u
#define APP_HEARTBEAT_STALE_MS               3000u
#define APP_UDP_PORT                         3333

#define APP_EVENT_QUEUE_LEN                  32
#define APP_UART_QUEUE_LEN                   32
#define APP_UART_TX_DRAIN_CHUNKS_AFTER_RX    1u
#define APP_UART_TX_DRAIN_CHUNKS_IDLE        8u
#define APP_UART_BACKPRESSURE_HIGH_WATER     (APP_UART_QUEUE_LEN - 4u)
#define APP_UART_BACKPRESSURE_LOW_WATER      (APP_UART_QUEUE_LEN / 2u)
#define APP_USB_CYCLE_MAX_AGE_MS             200u

static const uint8_t s_broadcast_mac[ESP_NOW_ETH_ALEN] = {
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
};

#define HOT_LOGI(tag, fmt, ...) do { if (APP_TRACE_HOT_PATH) ESP_LOGI(tag, fmt, ##__VA_ARGS__); } while (0)

typedef enum {
    APP_MODE_WAIT = 0,
    APP_MODE_RUNNING = 1,
} app_mode_t;

typedef enum {
    TX_EVT_START = 1,
    TX_EVT_STOP = 2,
    TX_EVT_TIMEOUT = 3,
    TX_EVT_RX_RECORD = 4,
} tx_event_type_t;

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint8_t version;
    uint8_t frame_type;
    uint16_t payload_len;
    uint32_t frame_seq;
    uint32_t checksum;
} serial_frame_header_t;

typedef struct __attribute__((packed)) {
    uint8_t mode;
    uint8_t wifi_channel;
    uint8_t second_channel;
    uint8_t protocol_bitmap;
    uint8_t connected_count;
    uint8_t active_count;
    uint8_t saved_count;
    uint8_t node_entry_count;
    uint32_t timeout_us;
    uint32_t udp_slot_gap_us;
    uint32_t generation;
    uint32_t next_trigger_seq;
    uint32_t uart_seq;
    uint32_t trigger_sent_count;
    uint32_t cycle_timeout_count;
    uint8_t tx_ap_mac[6];
    uint8_t reserved[2];
} status_payload_header_t;

typedef struct __attribute__((packed)) {
    uint8_t slot_index;
    uint8_t saved_order;
    uint8_t connect_order;
    uint8_t flags;
    uint8_t mac[6];
    uint32_t last_seen_ms;
    uint32_t rx_success_count;
    uint32_t rx_timeout_count;
} status_node_entry_t;

typedef struct __attribute__((packed)) {
    uint8_t ok;
    uint8_t mode;
    uint8_t wifi_channel;
    uint8_t second_channel;
    uint32_t timeout_us;
    uint32_t udp_slot_gap_us;
    char message[64];
} ack_payload_t;

typedef struct __attribute__((packed)) {
    uint32_t uart_seq;
    uint32_t trigger_seq;
    uint64_t trigger_tx_us;
    uint64_t cycle_done_us;
    uint32_t slot_timeout_us;
    uint8_t active_node_count;
    uint8_t received_node_count;
    uint8_t timeout_fired;
    uint8_t reserved;
} cycle_payload_header_t;

typedef struct __attribute__((packed)) {
    uint8_t rx_index;
    uint8_t present;
    int8_t rssi;
    uint8_t reserved;
    uint16_t csi_len;
} cycle_slot_header_t;

typedef struct __attribute__((packed)) {
    uint16_t frame_ctrl;
    uint16_t duration;
    uint8_t addr1[ESP_NOW_ETH_ALEN];
    uint8_t addr2[ESP_NOW_ETH_ALEN];
    uint8_t addr3[ESP_NOW_ETH_ALEN];
    uint16_t seq_ctrl;
    uint8_t llc_dsap;
    uint8_t llc_ssap;
    uint8_t llc_control;
    uint8_t oui[3];
    uint8_t protocol_id[2];
    uint32_t magic;
    uint8_t version;
    uint8_t frame_kind;
    uint8_t reserved0[2];
    uint32_t generation;
    uint32_t trigger_seq;
    uint64_t tx_timestamp_us;
    uint8_t active_nodes;
    uint8_t reserved[3];
} trigger_data_frame_t;

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint8_t version;
    uint8_t msg_type;
    uint8_t slot_index;
    uint8_t total_nodes;
    uint32_t generation;
    uint32_t timeout_us;
    uint32_t udp_slot_gap_us;
    uint8_t target_mac[6];
    uint8_t tx_mac[6];
    uint8_t wifi_channel;
    uint8_t second_channel;
    uint8_t running;
    uint8_t reserved;
    uint32_t checksum;
} link_ctrl_packet_t;

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint8_t version;
    uint8_t rx_index;
    int8_t rssi;
    uint8_t reserved;
    uint16_t csi_len;
    uint32_t generation;
    uint32_t checksum;
} udp_csi_packet_header_t;

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint8_t version;
    uint8_t rx_index;
    uint8_t running;
    uint8_t reserved;
    uint32_t generation;
    uint8_t sta_mac[6];
    uint8_t tx_mac[6];
    uint32_t checksum;
} link_heartbeat_packet_t;

typedef struct {
    bool valid;
    uint8_t mac[6];
    uint8_t saved_order;
} saved_node_cfg_t;

typedef struct {
    bool in_use;
    bool connected;
    uint8_t mac[6];
    uint8_t connect_order;
    uint64_t last_seen_us;
    uint32_t rx_success_count;
    uint32_t rx_timeout_count;
    uint8_t slot_index;
} rx_node_t;

typedef struct {
    bool present;
    bool complete;
    int8_t rssi;
    uint16_t csi_len;
    uint8_t csi[APP_MAX_CSI_LEN];
} cycle_node_buffer_t;

typedef struct {
    bool active;
    uint32_t generation;
    uint32_t trigger_seq;
    uint64_t trigger_tx_us;
    uint8_t active_node_count;
    bool timeout_fired;
    cycle_node_buffer_t slots[APP_MAX_RX_NODES];
} cycle_state_t;

typedef struct {
    app_mode_t mode;
    uint8_t wifi_channel;
    wifi_second_chan_t second_channel;
    uint32_t slot_timeout_us;
    uint32_t udp_slot_gap_us;
    uint32_t generation;
    uint32_t next_trigger_seq;
    uint32_t uart_seq;
    uint32_t serial_frame_seq;
    uint32_t trigger_sent_count;
    uint32_t cycle_timeout_count;
    uint8_t tx_ap_mac[6];
    uint8_t running_node_indices[APP_MAX_RX_NODES];
    uint8_t running_node_count;
    bool assignment_dirty;
    rx_node_t nodes[APP_MAX_RX_NODES];
    saved_node_cfg_t saved_nodes[APP_MAX_RX_NODES];
    uint8_t saved_node_count;
    uint8_t connect_counter;
} app_state_t;

typedef struct {
    uint8_t *data;
    size_t len;
    uint8_t frame_type;
    uint32_t enqueue_ms;
} uart_tx_item_t;

typedef struct {
    tx_event_type_t type;
    union {
        struct {
            udp_csi_packet_header_t header;
            uint16_t csi_len;
            uint8_t csi[APP_MAX_CSI_LEN];
        } rx_record;
    } data;
} tx_event_t;

static QueueHandle_t s_evt_queue;
static QueueHandle_t s_uart_tx_queue;
static uart_tx_item_t s_usb_tx_current;
static size_t s_usb_tx_offset;
static bool s_usb_tx_busy;
static uint32_t s_usb_tx_start_ms;
static uint32_t s_usb_tx_last_progress_ms;
static uint32_t s_usb_cycle_drop_count;
static gptimer_handle_t s_timeout_timer;
static app_state_t s_state;
static cycle_state_t s_cycle;
static portMUX_TYPE s_state_lock = portMUX_INITIALIZER_UNLOCKED;
static volatile uint32_t s_timeout_generation;
static volatile uint32_t s_timeout_trigger_seq;
static volatile uint32_t s_evt_queue_isr_drop_count;
static const char *TAG = "CSI_TX";

/* 바이너리 프레임과 제어 패킷에서 공통으로 쓰는 간단한 체크섬이다. */
static uint32_t checksum32(const void *data, size_t len, uint32_t seed)
{
    const uint8_t *bytes = (const uint8_t *)data;
    uint32_t value = seed; // 초기값 적용

    for (size_t i = 0; i < len; ++i) {
        value = (value << 5) - value + bytes[i];
    }
    return value;
}

static uint32_t now_ms(void)
{
    return (uint32_t)(esp_timer_get_time() / 1000LL);
}

static bool enqueue_tx_event(const tx_event_t *evt, const char *source)
{
    UBaseType_t queued;

    if (evt == NULL || s_evt_queue == NULL) {
        return false;
    }

    queued = uxQueueMessagesWaiting(s_evt_queue);
    if (queued >= (APP_EVENT_QUEUE_LEN - 1u)) {
        ESP_LOGW(
            TAG,
            "evt queue near full source=%s queued=%lu/%u type=%u",
            source,
            (unsigned long)queued,
            (unsigned)APP_EVENT_QUEUE_LEN,
            (unsigned)evt->type
        );
    }

    if (xQueueSend(s_evt_queue, evt, 0) != pdTRUE) {
        queued = uxQueueMessagesWaiting(s_evt_queue);
        ESP_LOGW(
            TAG,
            "evt queue overflow source=%s queued=%lu/%u type=%u",
            source,
            (unsigned long)queued,
            (unsigned)APP_EVENT_QUEUE_LEN,
            (unsigned)evt->type
        );
        return false;
    }

    return true;
}

static bool cycle_all_slots_complete(void)
{
    uint8_t i;

    if (!s_cycle.active || s_cycle.active_node_count == 0u) {
        return false;
    }

    for (i = 0; i < s_cycle.active_node_count; ++i) {
        if (!s_cycle.slots[i].complete) {
            return false;
        }
    }

    return true;
}

/* TinyUSB CDC는 endpoint 크기가 작기 때문에 프레임을 잘게 나눠 queue/flush 한다. */
static esp_err_t tinyusb_write_chunk(const uint8_t *data, size_t len, size_t *written_out)
{
    size_t chunk;
    size_t queued;
    esp_err_t flush_err;

    if (data == NULL && len > 0u) {
        return ESP_ERR_INVALID_ARG;
    }
    if (written_out != NULL) {
        *written_out = 0u;
    }
    if (len == 0u) {
        return tinyusb_cdcacm_write_flush(APP_USB_CDC_PORT, APP_USB_FLUSH_TIMEOUT_MS);
    }

    flush_err = tinyusb_cdcacm_write_flush(APP_USB_CDC_PORT, APP_USB_FLUSH_TIMEOUT_MS);

    chunk = len;
    if (chunk > APP_USB_CHUNK_BYTES) {
        chunk = APP_USB_CHUNK_BYTES;
    }

    queued = tinyusb_cdcacm_write_queue(APP_USB_CDC_PORT, data, chunk);
    if (queued == 0u) {
        return flush_err == ESP_OK ? ESP_ERR_TIMEOUT : flush_err;
    }

    flush_err = tinyusb_cdcacm_write_flush(APP_USB_CDC_PORT, APP_USB_FLUSH_TIMEOUT_MS);
    if (written_out != NULL) {
        *written_out = queued;
    }
    if (flush_err != ESP_OK) {
        return flush_err;
    }
    return ESP_OK;
}

/* TinyUSB CDC에서 현재 준비된 수신 바이트만 읽어 온다. 연결 전/직후 실패는 무시한다. */
static int tinyusb_read_available(uint8_t *buffer, size_t buffer_size)
{
    size_t rx_size = 0u;
    esp_err_t err;

    if (buffer == NULL || buffer_size == 0u) {
        return 0;
    }

    err = tinyusb_cdcacm_read(APP_USB_CDC_PORT, buffer, buffer_size, &rx_size);
    if (err != ESP_OK) {
        return 0;
    }
    return (int)rx_size;
}

/* 배열 안에서 MAC을 찾아 해당 노드 슬롯을 반환한다. */
static int find_node_by_mac(const uint8_t mac[6])
{
    int i;

    for (i = 0; i < APP_MAX_RX_NODES; ++i) {
        if (s_state.nodes[i].in_use && memcmp(s_state.nodes[i].mac, mac, 6) == 0) {
            return i;
        }
    }
    return -1;
}

/* 저장된 슬롯 순서를 찾는다. 저장 이력이 없으면 false를 반환한다. */
static bool find_saved_order(const uint8_t mac[6], uint8_t *saved_order_out)
{
    uint8_t i;

    for (i = 0; i < s_state.saved_node_count; ++i) {
        if (s_state.saved_nodes[i].valid &&
            memcmp(s_state.saved_nodes[i].mac, mac, 6) == 0) {
            if (saved_order_out != NULL) {
                *saved_order_out = s_state.saved_nodes[i].saved_order;
            }
            return true;
        }
    }
    return false;
}

/* 연결된 RX를 저장 순서 우선, 없으면 연결 순서 우선으로 정렬한다. */
static uint8_t collect_sorted_connected_nodes(uint8_t out_indices[APP_MAX_RX_NODES])
{
    uint8_t count = 0;
    int i;
    int j;

    for (i = 0; i < APP_MAX_RX_NODES; ++i) {
        if (s_state.nodes[i].in_use && s_state.nodes[i].connected) {
            out_indices[count++] = (uint8_t)i;
        }
    }

    for (i = 1; i < count; ++i) {
        uint8_t current = out_indices[i];
        uint8_t current_saved = 0xffu;
        bool current_has_saved = find_saved_order(s_state.nodes[current].mac, &current_saved);

        j = i;
        while (j > 0) {
            uint8_t prev = out_indices[j - 1];
            uint8_t prev_saved = 0xffu;
            bool prev_has_saved = find_saved_order(s_state.nodes[prev].mac, &prev_saved);
            bool move_prev = false;

            if (current_has_saved && !prev_has_saved) {
                move_prev = true;
            } else if (current_has_saved && prev_has_saved && current_saved < prev_saved) {
                move_prev = true;
            } else if (!current_has_saved && !prev_has_saved &&
                       s_state.nodes[current].connect_order < s_state.nodes[prev].connect_order) {
                move_prev = true;
            }

            if (!move_prev) {
                break;
            }

            out_indices[j] = out_indices[j - 1];
            --j;
        }

        out_indices[j] = current;
    }

    for (i = 0; i < count; ++i) {
        s_state.nodes[out_indices[i]].slot_index = (uint8_t)i;
    }

    return count;
}

static bool node_is_live_locked(uint8_t node_index, uint64_t now_us)
{
    uint64_t last_seen_us;

    if (node_index >= APP_MAX_RX_NODES ||
        !s_state.nodes[node_index].in_use ||
        !s_state.nodes[node_index].connected) {
        return false;
    }

    last_seen_us = s_state.nodes[node_index].last_seen_us;
    if (last_seen_us == 0u) {
        return false;
    }
    if (now_us < last_seen_us) {
        return true;
    }
    return (now_us - last_seen_us) <= ((uint64_t)APP_HEARTBEAT_STALE_MS * 1000ULL);
}

static uint8_t collect_live_connected_nodes(uint8_t out_indices[APP_MAX_RX_NODES])
{
    uint8_t sorted[APP_MAX_RX_NODES];
    uint8_t count;
    uint8_t live_count = 0u;
    uint8_t i;
    uint64_t now_us = (uint64_t)esp_timer_get_time();

    count = collect_sorted_connected_nodes(sorted);
    for (i = 0; i < count; ++i) {
        if (node_is_live_locked(sorted[i], now_us)) {
            out_indices[live_count++] = sorted[i];
        }
    }
    return live_count;
}

static uint8_t running_saved_rx_index(uint8_t compact_index)
{
    uint8_t node_index;
    uint8_t saved_order = 0xffu;

    if (compact_index >= s_state.running_node_count) {
        return compact_index;
    }

    node_index = s_state.running_node_indices[compact_index];
    if (find_saved_order(s_state.nodes[node_index].mac, &saved_order)) {
        return saved_order;
    }
    return compact_index;
}


/* 상태 프레임과 명령 응답 프레임을 위해 UART 바이트 버퍼를 하나 할당한다. */
static uint8_t *alloc_serial_frame(uint8_t frame_type, const void *payload, size_t payload_len, size_t *frame_len_out)
{
    serial_frame_header_t header;
    uint8_t *buffer = NULL;
    size_t frame_len = sizeof(header) + payload_len;

    if (payload_len > APP_MAX_UART_FRAME_PAYLOAD) {
        return NULL;
    }

    buffer = (uint8_t *)malloc(frame_len);
    if (buffer == NULL) {
        return NULL;
    }

    header.magic = APP_SERIAL_MAGIC;
    header.version = APP_PROTOCOL_VERSION;
    header.frame_type = frame_type;
    header.payload_len = (uint16_t)payload_len;
    portENTER_CRITICAL(&s_state_lock);
    header.frame_seq = ++s_state.serial_frame_seq;
    portEXIT_CRITICAL(&s_state_lock);
    header.checksum = 0u;

    memcpy(buffer, &header, sizeof(header));
    if (payload_len > 0u && payload != NULL) {
        memcpy(buffer + sizeof(header), payload, payload_len);
    }

    uint32_t cksum = checksum32(
        buffer,
        sizeof(header) - sizeof(header.checksum),
        0u
    );
    if (payload_len > 0u) {
        cksum = checksum32(buffer + sizeof(header), payload_len, cksum);
    }

    ((serial_frame_header_t *)buffer)->checksum = cksum;

    if (frame_len_out != NULL) {
        *frame_len_out = frame_len;
    }
    return buffer;
}

/* 제어 태스크가 실제 UART 송신을 담당하므로, 다른 태스크는 큐에만 넣는다. */
static void prune_cycle_backlog(void)
{
    uart_tx_item_t kept[APP_UART_QUEUE_LEN];
    UBaseType_t queued = uxQueueMessagesWaiting(s_uart_tx_queue);
    size_t kept_count = 0u;
    uint32_t dropped = 0u;

    if (queued < APP_UART_BACKPRESSURE_HIGH_WATER) {
        return;
    }

    while (xQueueReceive(s_uart_tx_queue, &kept[kept_count], 0) == pdTRUE) {
        if (kept[kept_count].frame_type == APP_SERIAL_FRAME_CYCLE &&
            (kept_count + uxQueueMessagesWaiting(s_uart_tx_queue)) >= APP_UART_BACKPRESSURE_LOW_WATER) {
            free(kept[kept_count].data);
            dropped++;
            continue;
        }
        kept_count++;
        if (kept_count >= APP_UART_QUEUE_LEN) {
            break;
        }
    }

    for (size_t i = 0u; i < kept_count; ++i) {
        if (xQueueSend(s_uart_tx_queue, &kept[i], 0) != pdTRUE) {
            free(kept[i].data);
            dropped++;
        }
    }

    if (dropped > 0u) {
        s_usb_cycle_drop_count += dropped;
        ESP_LOGW(TAG, "dropped old cycle frames=%lu total_dropped=%lu",
                 (unsigned long)dropped, (unsigned long)s_usb_cycle_drop_count);
    }
}

static void enqueue_uart_frame(uint8_t *frame, size_t frame_len)
{
    uart_tx_item_t item;
    UBaseType_t queued;
    uint8_t frame_type = 0u;

    if (frame == NULL || frame_len == 0u) {
        free(frame);
        return;
    }

    if (frame_len >= sizeof(serial_frame_header_t)) {
        frame_type = ((const serial_frame_header_t *)frame)->frame_type;
    }
    if (frame_type == APP_SERIAL_FRAME_CYCLE) {
        prune_cycle_backlog();
    }

    item.data = frame;
    item.len = frame_len;
    item.frame_type = frame_type;
    item.enqueue_ms = now_ms();
    queued = uxQueueMessagesWaiting(s_uart_tx_queue);
    if (queued >= (APP_UART_QUEUE_LEN - 1u)) {
        ESP_LOGW(
            TAG,
            "uart queue near full queued=%lu/%u frame_len=%u",
            (unsigned long)queued,
            (unsigned)APP_UART_QUEUE_LEN,
            (unsigned)frame_len
        );
    }
    if (xQueueSend(s_uart_tx_queue, &item, 0) != pdTRUE) {
        queued = uxQueueMessagesWaiting(s_uart_tx_queue);
        ESP_LOGW(
            TAG,
            "uart queue overflow queued=%lu/%u frame_len=%u",
            (unsigned long)queued,
            (unsigned)APP_UART_QUEUE_LEN,
            (unsigned)frame_len
        );
        free(frame);
    }
}

/* 실제 USB 송신은 제어 태스크 한 곳에서만 수행해 CDC 버퍼 경쟁을 피한다. */
static bool usb_tx_pending(void)
{
    return s_usb_tx_busy || uxQueueMessagesWaiting(s_uart_tx_queue) > 0u;
}

static uint32_t drain_usb_tx_queue(uint32_t max_chunks)
{
    uint32_t chunks = 0u;
    static uint32_t stall_count = 0u;

    while (chunks < max_chunks) {
        size_t written = 0u;
        esp_err_t err;

        if (!s_usb_tx_busy) {
            if (xQueueReceive(s_uart_tx_queue, &s_usb_tx_current, 0) != pdTRUE) {
                break;
            }
            s_usb_tx_offset = 0u;
            s_usb_tx_busy = true;
            s_usb_tx_start_ms = now_ms();
            s_usb_tx_last_progress_ms = s_usb_tx_start_ms;
        }

        if (s_usb_tx_current.frame_type == APP_SERIAL_FRAME_CYCLE &&
            (now_ms() - s_usb_tx_start_ms) > APP_USB_CYCLE_MAX_AGE_MS) {
            ESP_LOGW(
                TAG,
                "drop stale in-flight cycle offset=%u/%u age_ms=%lu",
                (unsigned)s_usb_tx_offset,
                (unsigned)s_usb_tx_current.len,
                (unsigned long)(now_ms() - s_usb_tx_start_ms)
            );
            free(s_usb_tx_current.data);
            memset(&s_usb_tx_current, 0, sizeof(s_usb_tx_current));
            s_usb_tx_offset = 0u;
            s_usb_tx_busy = false;
            s_usb_cycle_drop_count++;
            continue;
        }

        err = tinyusb_write_chunk(
            s_usb_tx_current.data + s_usb_tx_offset,
            s_usb_tx_current.len - s_usb_tx_offset,
            &written
        );
        if (written > 0u) {
            s_usb_tx_offset += written;
            s_usb_tx_last_progress_ms = now_ms();
            chunks++;
        }
        if (err != ESP_OK) {
            stall_count++;
            if ((stall_count % APP_USB_FLUSH_LOG_INTERVAL) == 0u) {
                ESP_LOGW(
                    TAG,
                    "usb tx stalled err=%s offset=%u/%u queued=%lu",
                    esp_err_to_name(err),
                    (unsigned)s_usb_tx_offset,
                    (unsigned)s_usb_tx_current.len,
                    (unsigned long)uxQueueMessagesWaiting(s_uart_tx_queue)
                );
            }
            break;
        }
        stall_count = 0u;
        if (s_usb_tx_offset >= s_usb_tx_current.len) {
            free(s_usb_tx_current.data);
            memset(&s_usb_tx_current, 0, sizeof(s_usb_tx_current));
            s_usb_tx_offset = 0u;
            s_usb_tx_busy = false;
        }
    }
    return chunks;
}

/* ACK 프레임은 설정 명령 직후 바로 보내도록 별도 헬퍼를 둔다. */
static void enqueue_ack(bool ok, const char *message)
{
    ack_payload_t ack;
    uint8_t *frame;
    size_t frame_len = 0u;

    memset(&ack, 0, sizeof(ack));
    portENTER_CRITICAL(&s_state_lock);
    ack.ok = ok ? 1u : 0u;
    ack.mode = (uint8_t)s_state.mode;
    ack.wifi_channel = s_state.wifi_channel;
    ack.second_channel = (uint8_t)s_state.second_channel;
    ack.timeout_us = s_state.slot_timeout_us;
    ack.udp_slot_gap_us = s_state.udp_slot_gap_us;
    portEXIT_CRITICAL(&s_state_lock);

    if (message != NULL) {
        snprintf(ack.message, sizeof(ack.message), "%s", message);
    }

    frame = alloc_serial_frame(APP_SERIAL_FRAME_ACK, &ack, sizeof(ack), &frame_len);
    enqueue_uart_frame(frame, frame_len);
}

/* NVS에서 채널/타임아웃/저장된 MAC 순서를 읽는다. */
static void load_settings_from_nvs(void)
{
    nvs_handle_t nvs = 0;
    uint32_t value32 = 0u;
    size_t blob_len = 0u;

    s_state.wifi_channel = APP_DEFAULT_WIFI_CHANNEL;
    s_state.second_channel = APP_DEFAULT_SECOND_CHANNEL;
    s_state.slot_timeout_us = APP_DEFAULT_SLOT_TIMEOUT_US;
    s_state.udp_slot_gap_us = APP_DEFAULT_UDP_SLOT_GAP_US;
    s_state.saved_node_count = 0u;

    if (nvs_open("csi_tx", NVS_READONLY, &nvs) != ESP_OK) {
        return;
    }

    if (nvs_get_u32(nvs, "timeout_us", &value32) == ESP_OK) {
        s_state.slot_timeout_us = value32;
    }
    if (nvs_get_u32(nvs, "udp_gap_us", &value32) == ESP_OK) {
        s_state.udp_slot_gap_us = value32;
    }
    if (nvs_get_u32(nvs, "channel", &value32) == ESP_OK && value32 >= 1u && value32 <= 13u) {
        s_state.wifi_channel = (uint8_t)value32;
    }
    if (nvs_get_u32(nvs, "second", &value32) == ESP_OK &&
        value32 <= (uint32_t)WIFI_SECOND_CHAN_BELOW) {
        s_state.second_channel = (wifi_second_chan_t)value32;
    }
    if (nvs_get_blob(nvs, "saved_nodes", NULL, &blob_len) == ESP_OK &&
        blob_len <= sizeof(s_state.saved_nodes)) {
        if (nvs_get_blob(nvs, "saved_nodes", s_state.saved_nodes, &blob_len) == ESP_OK) {
            s_state.saved_node_count = (uint8_t)(blob_len / sizeof(saved_node_cfg_t));
        }
    }

    nvs_close(nvs);
}

/* NVS에 현재 설정과 저장된 노드 순서를 기록한다. */
static void save_settings_to_nvs(void)
{
    nvs_handle_t nvs = 0;

    if (nvs_open("csi_tx", NVS_READWRITE, &nvs) != ESP_OK) {
        return;
    }

    nvs_set_u32(nvs, "timeout_us", s_state.slot_timeout_us);
    nvs_set_u32(nvs, "udp_gap_us", s_state.udp_slot_gap_us);
    nvs_set_u32(nvs, "channel", s_state.wifi_channel);
    nvs_set_u32(nvs, "second", (uint32_t)s_state.second_channel);
    nvs_set_blob(
        nvs,
        "saved_nodes",
        s_state.saved_nodes,
        (size_t)s_state.saved_node_count * sizeof(saved_node_cfg_t)
    );
    nvs_commit(nvs);
    nvs_close(nvs);
}

/* 현재 연결 순서를 사용해 저장용 MAC 리스트를 새로 만든다. */
static void snapshot_nodes_to_saved_config(void)
{
    uint8_t sorted[APP_MAX_RX_NODES];
    uint8_t count;
    uint8_t i;

    portENTER_CRITICAL(&s_state_lock);
    count = collect_sorted_connected_nodes(sorted);
    memset(s_state.saved_nodes, 0, sizeof(s_state.saved_nodes));
    for (i = 0; i < count; ++i) {
        s_state.saved_nodes[i].valid = true;
        s_state.saved_nodes[i].saved_order = i;
        memcpy(s_state.saved_nodes[i].mac, s_state.nodes[sorted[i]].mac, 6);
    }
    s_state.saved_node_count = count;
    s_state.assignment_dirty = true;
    portEXIT_CRITICAL(&s_state_lock);

    save_settings_to_nvs();
}

/* TX는 SoftAP로 동작하며, RX의 연결 순서를 wait mode에서 추적한다. */
static void wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data)
{
    (void)arg;

    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_AP_STACONNECTED) {
        wifi_event_ap_staconnected_t *event = (wifi_event_ap_staconnected_t *)event_data;
        int index;
        int i;

        portENTER_CRITICAL(&s_state_lock);
        index = find_node_by_mac(event->mac);
        if (index < 0) {
            for (i = 0; i < APP_MAX_RX_NODES; ++i) {
                if (!s_state.nodes[i].in_use) {
                    index = i;
                    memset(&s_state.nodes[i], 0, sizeof(s_state.nodes[i]));
                    s_state.nodes[i].in_use = true;
                    memcpy(s_state.nodes[i].mac, event->mac, 6);
                    s_state.nodes[i].connect_order = ++s_state.connect_counter;
                    break;
                }
            }
        }
        if (index >= 0) {
            s_state.nodes[index].connected = true;
            s_state.nodes[index].last_seen_us = esp_timer_get_time();
            s_state.assignment_dirty = true;
        }
        portEXIT_CRITICAL(&s_state_lock);
        ESP_LOGI(TAG, "RX connected: " MACSTR " aid=%d slot=%d", MAC2STR(event->mac), event->aid, index);
        return;
    }

    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_AP_STADISCONNECTED) {
        wifi_event_ap_stadisconnected_t *event = (wifi_event_ap_stadisconnected_t *)event_data;
        int index;

        portENTER_CRITICAL(&s_state_lock);
        index = find_node_by_mac(event->mac);
        if (index >= 0) {
            s_state.nodes[index].connected = false;
            s_state.nodes[index].last_seen_us = esp_timer_get_time();
            s_state.assignment_dirty = true;
        }
        portEXIT_CRITICAL(&s_state_lock);
        ESP_LOGW(TAG, "RX disconnected: " MACSTR " aid=%d slot=%d", MAC2STR(event->mac), event->aid, index);
    }
}

/* CSI 수집용 트리거는 HT40/MCS3 raw action frame으로 보낸다. */
static void init_wifi_softap(void)
{
    wifi_init_config_t init_cfg = WIFI_INIT_CONFIG_DEFAULT();
    wifi_config_t ap_cfg = {
        .ap = {
            .channel = APP_DEFAULT_WIFI_CHANNEL,
            .max_connection = APP_WIFI_MAX_CONN,
            .authmode = WIFI_AUTH_OPEN,
        },
    };

    memcpy(ap_cfg.ap.ssid, APP_WIFI_SSID, strlen(APP_WIFI_SSID));
    ap_cfg.ap.ssid_len = strlen(APP_WIFI_SSID);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_ap();
    ESP_ERROR_CHECK(esp_wifi_init(&init_cfg));
    ESP_ERROR_CHECK(
        esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL)
    );
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

    uint8_t ch;
    wifi_second_chan_t second_ch;
    
    portENTER_CRITICAL(&s_state_lock);
    ch = s_state.wifi_channel;
    second_ch = s_state.second_channel;
    portEXIT_CRITICAL(&s_state_lock);

    ap_cfg.ap.channel = ch;

    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap_cfg));
    ESP_ERROR_CHECK(esp_wifi_set_protocol(WIFI_IF_AP, APP_WIFI_PROTOCOL_BITMAP));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_AP, APP_WIFI_BANDWIDTH));
    ESP_ERROR_CHECK(esp_wifi_set_channel(ch, second_ch));

    ESP_ERROR_CHECK(esp_wifi_config_80211_tx_rate(WIFI_IF_AP, APP_TRIGGER_TX_RATE));
    ESP_ERROR_CHECK(esp_wifi_get_mac(WIFI_IF_AP, s_state.tx_ap_mac));
    ESP_LOGI(
        TAG,
        "SoftAP ready ssid=%s ch=%u second=%u mac=" MACSTR,
        APP_WIFI_SSID,
        ch,
        (unsigned)second_ch,
        MAC2STR(s_state.tx_ap_mac)
    );
}

/* ESP-NOW는 RX가 보낸 CSI 조각과 제어 브로드캐스트를 같은 채널에서 다룬다. */
static void init_espnow(void)
{
    esp_now_peer_info_t peer = {0};

    ESP_ERROR_CHECK(esp_now_init());
    memcpy(peer.peer_addr, s_broadcast_mac, ESP_NOW_ETH_ALEN);
    peer.ifidx = WIFI_IF_AP;
    peer.channel = s_state.wifi_channel;
    peer.encrypt = false;
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
}

/* 타임아웃은 gptimer one-shot으로 걸어서 cycle jitter를 줄인다. */
static bool IRAM_ATTR timeout_alarm_cb(
    gptimer_handle_t timer,
    const gptimer_alarm_event_data_t *edata,
    void *user_ctx
)
{
    BaseType_t high_woken = pdFALSE;
    QueueHandle_t queue = (QueueHandle_t)user_ctx;
    tx_event_t evt = {
        .type = TX_EVT_TIMEOUT,
    };

    (void)timer;
    (void)edata;

    if (queue != NULL) {
        if (xQueueSendFromISR(queue, &evt, &high_woken) != pdTRUE) {
            s_evt_queue_isr_drop_count++;
        }
    }
    return high_woken == pdTRUE;
}

/* cycle timeout 타이머를 준비한다. */
static void init_timeout_timer(void)
{
    gptimer_config_t timer_cfg = {
        .clk_src = GPTIMER_CLK_SRC_DEFAULT,
        .direction = GPTIMER_COUNT_UP,
        .resolution_hz = 1000000,
    };
    gptimer_event_callbacks_t callbacks = {
        .on_alarm = timeout_alarm_cb,
    };

    ESP_ERROR_CHECK(gptimer_new_timer(&timer_cfg, &s_timeout_timer));
    ESP_ERROR_CHECK(gptimer_register_event_callbacks(s_timeout_timer, &callbacks, s_evt_queue));
    ESP_ERROR_CHECK(gptimer_enable(s_timeout_timer));
}

/* 매 cycle마다 deadline을 다시 설정한다. */
static void arm_cycle_timeout(uint32_t generation, uint32_t trigger_seq, uint32_t timeout_us)
{
    gptimer_alarm_config_t alarm_cfg = {
        .reload_count = 0,
        .alarm_count = timeout_us,
        .flags.auto_reload_on_alarm = false,
    };

    s_timeout_generation = generation;
    s_timeout_trigger_seq = trigger_seq;
    gptimer_stop(s_timeout_timer);
    gptimer_set_raw_count(s_timeout_timer, 0);
    gptimer_set_alarm_action(s_timeout_timer, &alarm_cfg);
    gptimer_start(s_timeout_timer);
}

/* raw trigger frame 템플릿을 매번 갱신해서 보낸다. */
static esp_err_t send_trigger_frame(
    uint32_t generation,
    uint32_t trigger_seq,
    uint8_t active_nodes,
    uint64_t *tx_us_out
)
{
    trigger_data_frame_t frame;
    uint64_t tx_us = (uint64_t)esp_timer_get_time();

    memset(&frame, 0, sizeof(frame));
    frame.frame_ctrl = 0x0208;
    frame.llc_dsap = 0xAA;
    frame.llc_ssap = 0xAA;
    frame.llc_control = 0x03;
    frame.protocol_id[0] = 0x88;
    frame.protocol_id[1] = 0xB5;
    frame.magic = APP_TRIGGER_DATA_MAGIC;
    frame.version = APP_PROTOCOL_VERSION;
    frame.frame_kind = 1u;
    frame.generation = generation;
    frame.trigger_seq = trigger_seq;
    frame.tx_timestamp_us = tx_us;
    frame.active_nodes = active_nodes;
    memcpy(frame.addr1, s_broadcast_mac, ESP_NOW_ETH_ALEN);
    memcpy(frame.addr2, s_state.tx_ap_mac, ESP_NOW_ETH_ALEN);
    memcpy(frame.addr3, s_state.tx_ap_mac, ESP_NOW_ETH_ALEN);

    if (tx_us_out != NULL) {
        *tx_us_out = tx_us;
    }

    return esp_wifi_80211_tx(WIFI_IF_AP, &frame, sizeof(frame), true);
}

/* assignment/mode/channel 정보는 브로드캐스트 제어 패킷으로 보낸다. */
static void send_ctrl_packet(
    uint8_t msg_type,
    const uint8_t target_mac[6],
    uint8_t slot_index,
    uint8_t total_nodes
)
{
    link_ctrl_packet_t packet;
    int retry;

    memset(&packet, 0, sizeof(packet));
    packet.magic = APP_LINK_CTRL_MAGIC;
    packet.version = APP_PROTOCOL_VERSION;
    packet.msg_type = msg_type;
    packet.slot_index = slot_index;
    packet.total_nodes = total_nodes;

    portENTER_CRITICAL(&s_state_lock);
    packet.generation = s_state.generation;
    packet.timeout_us = s_state.slot_timeout_us;
    packet.udp_slot_gap_us = s_state.udp_slot_gap_us;
    memcpy(packet.tx_mac, s_state.tx_ap_mac, 6);
    packet.wifi_channel = s_state.wifi_channel;
    packet.second_channel = (uint8_t)s_state.second_channel;
    packet.running = (s_state.mode == APP_MODE_RUNNING) ? 1u : 0u;
    portEXIT_CRITICAL(&s_state_lock);

    if (target_mac != NULL) {
        memcpy(packet.target_mac, target_mac, 6);
    }

    packet.checksum = checksum32(&packet, offsetof(link_ctrl_packet_t, checksum), 0);
    ESP_LOGI(
        TAG,
        "ctrl tx type=%u gen=%lu slot=%u total=%u timeout=%lu gap=%lu target=" MACSTR " running=%u",
        (unsigned)msg_type,
        (unsigned long)packet.generation,
        (unsigned)slot_index,
        (unsigned)total_nodes,
        (unsigned long)packet.timeout_us,
        (unsigned long)packet.udp_slot_gap_us,
        MAC2STR(packet.target_mac),
        (unsigned)packet.running
    );
    for (retry = 0; retry < APP_ESPNOW_BROADCAST_RETRY_COUNT; ++retry) {
        esp_now_send(s_broadcast_mac, (const uint8_t *)&packet, sizeof(packet));
        vTaskDelay(pdMS_TO_TICKS(5));
    }
}

/* wait mode에서 연결 변화가 생기면 각 RX에게 현재 slot을 다시 알려준다. */
static void broadcast_assignments_if_needed(void)
{
    uint8_t sorted[APP_MAX_RX_NODES];
    uint8_t count;
    uint8_t i;
    bool dirty = false;
    app_mode_t mode_snapshot;
    uint8_t target_macs[APP_MAX_RX_NODES][6];

    portENTER_CRITICAL(&s_state_lock);
    dirty = s_state.assignment_dirty;
    mode_snapshot = s_state.mode;
    if (dirty && mode_snapshot == APP_MODE_WAIT) {
        count = collect_sorted_connected_nodes(sorted);
        for (i = 0; i < count; ++i) {
            memcpy(target_macs[i], s_state.nodes[sorted[i]].mac, 6);
        }
        s_state.assignment_dirty = false;
    } else {
        count = 0u;
    }
    portEXIT_CRITICAL(&s_state_lock);

    if (!dirty || mode_snapshot != APP_MODE_WAIT) {
        return;
    }

    for (i = 0; i < count; ++i) {
        send_ctrl_packet(APP_CTRL_MSG_ASSIGNMENT, target_macs[i], i, count);
    }
    send_ctrl_packet(APP_CTRL_MSG_MODE, NULL, 0xffu, count);
}

/* 상태 프레임은 wait mode에서 주기적으로 GUI로 보내는 기본 정보다. */
static void enqueue_status_frame(void)
{
    uint8_t sorted[APP_MAX_RX_NODES];
    status_payload_header_t header;
    status_node_entry_t entries[APP_MAX_RX_NODES];
    uint8_t *payload = NULL;
    uint8_t *frame = NULL;
    size_t payload_len;
    size_t frame_len = 0u;
    uint8_t count;
    uint8_t live_count = 0u;
    uint8_t i;
    uint64_t now_us;

    memset(&header, 0, sizeof(header));
    memset(entries, 0, sizeof(entries));

    portENTER_CRITICAL(&s_state_lock);
    now_us = (uint64_t)esp_timer_get_time();
    count = collect_sorted_connected_nodes(sorted);
    for (i = 0; i < count; ++i) {
        if (node_is_live_locked(sorted[i], now_us)) {
            live_count++;
        }
    }
    header.mode = (uint8_t)s_state.mode;
    header.wifi_channel = s_state.wifi_channel;
    header.second_channel = (uint8_t)s_state.second_channel;
    header.protocol_bitmap = (uint8_t)APP_WIFI_PROTOCOL_BITMAP;
    header.connected_count = count;
    header.active_count = (s_state.mode == APP_MODE_RUNNING) ? s_state.running_node_count : live_count;
    header.saved_count = s_state.saved_node_count;
    header.node_entry_count = count;
    header.timeout_us = s_state.slot_timeout_us;
    header.udp_slot_gap_us = s_state.udp_slot_gap_us;
    header.generation = s_state.generation;
    header.next_trigger_seq = s_state.next_trigger_seq;
    header.uart_seq = s_state.uart_seq;
    header.trigger_sent_count = s_state.trigger_sent_count;
    header.cycle_timeout_count = s_state.cycle_timeout_count;
    memcpy(header.tx_ap_mac, s_state.tx_ap_mac, 6);

    for (i = 0; i < count; ++i) {
        uint8_t node_index = sorted[i];
        uint8_t saved_order = 0xffu;
        bool has_saved = find_saved_order(s_state.nodes[node_index].mac, &saved_order);

        entries[i].slot_index = i;
        entries[i].saved_order = has_saved ? saved_order : 0xffu;
        entries[i].connect_order = s_state.nodes[node_index].connect_order;
        entries[i].flags = APP_STATUS_FLAG_CONNECTED;
        if (has_saved) {
            entries[i].flags |= APP_STATUS_FLAG_SAVED;
        }
        if (node_is_live_locked(node_index, now_us)) {
            entries[i].flags |= APP_STATUS_FLAG_LIVE;
        }
        memcpy(entries[i].mac, s_state.nodes[node_index].mac, 6);
        entries[i].last_seen_ms = (uint32_t)(s_state.nodes[node_index].last_seen_us / 1000ULL);
        entries[i].rx_success_count = s_state.nodes[node_index].rx_success_count;
        entries[i].rx_timeout_count = s_state.nodes[node_index].rx_timeout_count;
    }
    portEXIT_CRITICAL(&s_state_lock);

    payload_len = sizeof(header) + ((size_t)count * sizeof(status_node_entry_t));
    payload = (uint8_t *)malloc(payload_len);
    if (payload == NULL) {
        return;
    }

    memcpy(payload, &header, sizeof(header));
    if (count > 0u) {
        memcpy(payload + sizeof(header), entries, (size_t)count * sizeof(status_node_entry_t));
    }

    frame = alloc_serial_frame(APP_SERIAL_FRAME_STATUS, payload, payload_len, &frame_len);
    free(payload);
    enqueue_uart_frame(frame, frame_len);
}

/* cycle이 끝나면 RX별 record를 한 frame으로 묶어 PC로 넘긴다. */
static void finalize_cycle_and_queue_uart(void)
{
    cycle_payload_header_t header;
    cycle_slot_header_t slot_hdr;
    uint8_t *payload;
    uint8_t *frame;
    size_t payload_len = sizeof(header);
    size_t cursor = 0u;
    size_t frame_len = 0u;
    uint8_t i;
    uint8_t received_count = 0u;

    memset(&header, 0, sizeof(header));

    for (i = 0; i < s_cycle.active_node_count; ++i) {
        payload_len += sizeof(slot_hdr);
        if (s_cycle.slots[i].complete) {
            payload_len += s_cycle.slots[i].csi_len;
            received_count++;
        }
    }

    payload = (uint8_t *)malloc(payload_len);
    if (payload == NULL) {
        memset(&s_cycle, 0, sizeof(s_cycle));
        return;
    }

    header.uart_seq = ++s_state.uart_seq;
    header.trigger_seq = s_cycle.trigger_seq;
    header.trigger_tx_us = s_cycle.trigger_tx_us;
    header.cycle_done_us = (uint64_t)esp_timer_get_time();
    header.slot_timeout_us = s_state.slot_timeout_us;
    header.active_node_count = s_cycle.active_node_count;
    header.received_node_count = received_count;
    header.timeout_fired = s_cycle.timeout_fired ? 1u : 0u;

    memcpy(payload + cursor, &header, sizeof(header));
    cursor += sizeof(header);

    for (i = 0; i < s_cycle.active_node_count; ++i) {
        memset(&slot_hdr, 0, sizeof(slot_hdr));
        slot_hdr.rx_index = running_saved_rx_index(i);
        slot_hdr.present = s_cycle.slots[i].complete ? 1u : 0u;
        slot_hdr.rssi = s_cycle.slots[i].complete ? s_cycle.slots[i].rssi : 0;
        slot_hdr.csi_len = s_cycle.slots[i].complete ? s_cycle.slots[i].csi_len : 0u;
        memcpy(payload + cursor, &slot_hdr, sizeof(slot_hdr));
        cursor += sizeof(slot_hdr);

        if (s_cycle.slots[i].complete) {
            memcpy(payload + cursor, s_cycle.slots[i].csi, s_cycle.slots[i].csi_len);
            cursor += s_cycle.slots[i].csi_len;
            s_state.nodes[s_state.running_node_indices[i]].rx_success_count++;
        } else if (i < s_state.running_node_count) {
            s_state.nodes[s_state.running_node_indices[i]].rx_timeout_count++;
        }
    }

    frame = alloc_serial_frame(APP_SERIAL_FRAME_CYCLE, payload, payload_len, &frame_len);
    free(payload);
    enqueue_uart_frame(frame, frame_len);

    HOT_LOGI(
        TAG,
        "cycle done trig=%lu uart=%lu rx=%u/%u timeout=%u",
        (unsigned long)header.trigger_seq,
        (unsigned long)header.uart_seq,
        (unsigned)header.received_node_count,
        (unsigned)header.active_node_count,
        (unsigned)header.timeout_fired
    );

    if (s_cycle.timeout_fired) {
        s_state.cycle_timeout_count++;
    }

    memset(&s_cycle, 0, sizeof(s_cycle));
}

/* 마지막 슬롯 RX가 도착했거나 타임아웃이면 다음 trigger를 바로 시작한다. */
static void start_next_cycle(void)
{
    uint64_t tx_us = 0u;
    uint32_t generation = s_state.generation;
    uint32_t trigger_seq = s_state.next_trigger_seq;
    uint8_t active_count = s_state.running_node_count;
    uint32_t timeout_us = s_state.slot_timeout_us * ((active_count > 0u) ? active_count : 1u);
    int attempt;

    while (s_state.mode == APP_MODE_RUNNING &&
           uxQueueMessagesWaiting(s_uart_tx_queue) >= APP_UART_BACKPRESSURE_HIGH_WATER) {
        vTaskDelay(pdMS_TO_TICKS(1));
    }

    for (attempt = 0; attempt < 3; ++attempt) {
        if (send_trigger_frame(generation, trigger_seq, active_count, &tx_us) == ESP_OK) {
            break;
        }
        vTaskDelay(pdMS_TO_TICKS(2));
    }

    HOT_LOGI(
        TAG,
        "trigger tx gen=%lu trig=%lu active=%u timeout_us=%lu tx_us=%llu",
        (unsigned long)generation,
        (unsigned long)trigger_seq,
        (unsigned)active_count,
        (unsigned long)timeout_us,
        (unsigned long long)tx_us
    );

    memset(&s_cycle, 0, sizeof(s_cycle));
    s_cycle.active = true;
    s_cycle.generation = generation;
    s_cycle.trigger_seq = trigger_seq;
    s_cycle.trigger_tx_us = tx_us;
    s_cycle.active_node_count = active_count;
    s_state.trigger_sent_count++;
    s_state.next_trigger_seq++;

    arm_cycle_timeout(generation, trigger_seq, timeout_us);
}

/* running mode 시작 시 wait mode에서 정한 순서를 snapshot해서 고정한다. */
static bool prepare_running_nodes(void)
{
    uint8_t count;
    uint8_t i;
    uint8_t live_nodes[APP_MAX_RX_NODES];

    count = collect_live_connected_nodes(live_nodes);
    for (i = 0; i < count; ++i) {
        s_state.running_node_indices[i] = live_nodes[i];
    }
    s_state.running_node_count = count;
    if (count == 0u) {
        return false;
    }
    return true;
}

/* coordinator는 running mode의 cycle 수명주기를 전담한다. */
static void coordinator_task(void *arg)
{
    tx_event_t evt;

    (void)arg;

    while (true) {
        if (xQueueReceive(s_evt_queue, &evt, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        if (s_evt_queue_isr_drop_count > 0u) {
            uint32_t dropped = s_evt_queue_isr_drop_count;
            s_evt_queue_isr_drop_count = 0u;
            ESP_LOGW(
                TAG,
                "evt queue overflow from ISR dropped=%lu queued_now=%lu/%u",
                (unsigned long)dropped,
                (unsigned long)uxQueueMessagesWaiting(s_evt_queue),
                (unsigned)APP_EVENT_QUEUE_LEN
            );
        }

        if (evt.type == TX_EVT_START) {
            if (s_state.mode != APP_MODE_RUNNING && prepare_running_nodes()) {
                s_state.mode = APP_MODE_RUNNING;
                s_state.generation++;
                s_state.assignment_dirty = false;
                ESP_LOGI(
                    TAG,
                    "run start generation=%lu nodes=%u",
                    (unsigned long)s_state.generation,
                    (unsigned)s_state.running_node_count
                );

                {
                    uint8_t i;
                    uint8_t count = s_state.running_node_count;
                    uint8_t mac_buf[APP_MAX_RX_NODES][6];

                    for (i = 0; i < count; ++i) {
                        memcpy(mac_buf[i], s_state.nodes[s_state.running_node_indices[i]].mac, 6);
                    }
                    for (i = 0; i < count; ++i) {
                        send_ctrl_packet(APP_CTRL_MSG_ASSIGNMENT, mac_buf[i], i, count);
                    }
                    send_ctrl_packet(APP_CTRL_MSG_MODE, NULL, 0xffu, count);
                }

                vTaskDelay(pdMS_TO_TICKS(40));
                start_next_cycle();
            }
            continue;
        }

        if (evt.type == TX_EVT_STOP) {
            portENTER_CRITICAL(&s_state_lock);
            s_state.mode = APP_MODE_WAIT;
            s_state.running_node_count = 0u;
            memset(&s_cycle, 0, sizeof(s_cycle));
            portEXIT_CRITICAL(&s_state_lock);
            gptimer_stop(s_timeout_timer);
            send_ctrl_packet(APP_CTRL_MSG_MODE, NULL, 0xffu, 0u);
            ESP_LOGI(TAG, "run stop -> wait");
            continue;
        }

        if (evt.type == TX_EVT_TIMEOUT) {
            if (s_state.mode == APP_MODE_RUNNING && s_cycle.active &&
                s_cycle.generation == s_timeout_generation &&
                s_cycle.trigger_seq == s_timeout_trigger_seq) {
                s_cycle.timeout_fired = true;
                ESP_LOGW(
                    TAG,
                    "cycle timeout gen=%lu trig=%lu active=%u",
                    (unsigned long)s_cycle.generation,
                    (unsigned long)s_cycle.trigger_seq,
                    (unsigned)s_cycle.active_node_count
                );
                finalize_cycle_and_queue_uart();
                start_next_cycle();
            }
            continue;
        }

        if (evt.type == TX_EVT_RX_RECORD) {
            udp_csi_packet_header_t *hdr = &evt.data.rx_record.header;
            cycle_node_buffer_t *slot;
            uint32_t current_trigger_seq = s_cycle.trigger_seq;

            if (s_state.mode != APP_MODE_RUNNING ||
                !s_cycle.active ||
                hdr->generation != s_cycle.generation ||
                hdr->rx_index >= s_cycle.active_node_count ||
                hdr->csi_len == 0u ||
                hdr->csi_len > APP_MAX_CSI_LEN ||
                hdr->csi_len != evt.data.rx_record.csi_len) {
                continue;
            }

            slot = &s_cycle.slots[hdr->rx_index];
            slot->present = true;
            slot->complete = true;
            slot->rssi = hdr->rssi;
            slot->csi_len = hdr->csi_len;
            memcpy(slot->csi, evt.data.rx_record.csi, hdr->csi_len);
            HOT_LOGI(
                TAG,
                "udp csi trig=%lu rx=%u rssi=%d csi_len=%u",
                (unsigned long)current_trigger_seq,
                (unsigned)hdr->rx_index,
                (int)hdr->rssi,
                (unsigned)hdr->csi_len
            );

            if (cycle_all_slots_complete()) {
                gptimer_stop(s_timeout_timer);
                finalize_cycle_and_queue_uart();
                start_next_cycle();
            }
            continue;
        }

    }
}

/* RX에서 오는 CSI record fragment만 빠르게 큐로 넘기고 리턴한다. */
#if 0
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
static void espnow_recv_cb(const esp_now_recv_info_t *recv_info, const uint8_t *data, int data_len)
#else
static void espnow_recv_cb(const uint8_t *src_mac, const uint8_t *data, int data_len)
#endif
{
    tx_event_t evt;
    const link_data_fragment_header_t *hdr = (const link_data_fragment_header_t *)data;
    uint32_t current_trigger_seq = s_cycle.trigger_seq;

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
    (void)recv_info;
#else
    (void)src_mac;
#endif

    if (data == NULL || data_len < (int)sizeof(link_data_fragment_header_t) ||
        data_len > (int)(sizeof(link_data_fragment_header_t) + APP_ESPNOW_FRAGMENT_PAYLOAD_MAX)) {
        return;
    }

    if (hdr->magic != APP_LINK_DATA_MAGIC ||
        hdr->version != APP_PROTOCOL_VERSION ||
        hdr->chunk_len > APP_ESPNOW_FRAGMENT_PAYLOAD_MAX ||
        (int)(sizeof(link_data_fragment_header_t) + hdr->chunk_len) != data_len) {
        ESP_LOGW(TAG, "drop fragment invalid header len=%d magic=0x%08" PRIx32, data_len, hdr->magic);
        return;
    }

    uint32_t checksum_calculated = checksum32(
        data,
        offsetof(link_data_fragment_header_t, checksum),
        0u
    );

    if (hdr->chunk_len > 0u) {
        checksum_calculated = checksum32(
            data + sizeof(link_data_fragment_header_t),
            hdr->chunk_len,
            checksum_calculated
        );
    }

    if (hdr->checksum != checksum_calculated) {
        ESP_LOGW(
            TAG,
            "drop fragment checksum trig=%lu rx=%u frag=%u expected=0x%08" PRIx32 " got=0x%08" PRIx32,
            (unsigned long)current_trigger_seq,
            (unsigned)hdr->rx_index,
            (unsigned)hdr->fragment_index,
            checksum_calculated,
            hdr->checksum
        );
        return;
    }

    memset(&evt, 0, sizeof(evt));
    evt.type = TX_EVT_RX_FRAGMENT;
    memcpy(&evt.data.rx_fragment.header, hdr, sizeof(*hdr));
    evt.data.rx_fragment.payload_len = hdr->chunk_len;
    memcpy(evt.data.rx_fragment.payload, data + sizeof(link_data_fragment_header_t), hdr->chunk_len);
    if (!enqueue_tx_event(&evt, "espnow_fragment")) {
        ESP_LOGW(
            TAG,
            "drop fragment enqueue trig=%lu rx=%u frag=%u/%u",
            (unsigned long)current_trigger_seq,
            (unsigned)hdr->rx_index,
            (unsigned)hdr->fragment_index + 1u,
            (unsigned)hdr->fragment_count
        );
        return;
    }
    HOT_LOGI(
        TAG,
        "fragment queued trig=%lu rx=%u frag=%u/%u chunk=%u",
        (unsigned long)current_trigger_seq,
        (unsigned)hdr->rx_index,
        (unsigned)hdr->fragment_index + 1u,
        (unsigned)hdr->fragment_count,
        (unsigned)hdr->chunk_len
    );

    portENTER_CRITICAL(&s_state_lock);
    {
        int node_index = find_node_by_mac(hdr->rx_mac);
        if (node_index >= 0) {
            s_state.nodes[node_index].last_seen_us = esp_timer_get_time();
        }
    }
    portEXIT_CRITICAL(&s_state_lock);
}
#endif

/* 수동 시리얼 터미널과 GUI 둘 다 쓰기 쉽도록 ASCII 명령을 처리한다. */
/* RX wait-mode heartbeat updates live status without affecting CSI data flow. */
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
static void espnow_recv_cb(const esp_now_recv_info_t *recv_info, const uint8_t *data, int data_len)
#else
static void espnow_recv_cb(const uint8_t *src_mac, const uint8_t *data, int data_len)
#endif
{
    const link_heartbeat_packet_t *packet;
    const uint8_t *mac;
    uint32_t checksum_calculated;
    int node_index;

    if (data == NULL || data_len != (int)sizeof(link_heartbeat_packet_t)) {
        return;
    }

    packet = (const link_heartbeat_packet_t *)data;
    if (packet->magic != APP_LINK_HEARTBEAT_MAGIC ||
        packet->version != APP_PROTOCOL_VERSION) {
        return;
    }

    checksum_calculated = checksum32(packet, offsetof(link_heartbeat_packet_t, checksum), 0u);
    if (checksum_calculated != packet->checksum) {
        ESP_LOGW(TAG, "drop heartbeat checksum mismatch");
        return;
    }

    mac = packet->sta_mac;
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
    if (memcmp(mac, "\0\0\0\0\0\0", 6) == 0 && recv_info != NULL) {
        mac = recv_info->src_addr;
    }
#else
    if (memcmp(mac, "\0\0\0\0\0\0", 6) == 0) {
        mac = src_mac;
    }
#endif
    if (mac == NULL) {
        return;
    }

    portENTER_CRITICAL(&s_state_lock);
    node_index = find_node_by_mac(mac);
    if (node_index >= 0 && s_state.nodes[node_index].connected) {
        s_state.nodes[node_index].last_seen_us = esp_timer_get_time();
    }
    portEXIT_CRITICAL(&s_state_lock);

    HOT_LOGI(
        TAG,
        "heartbeat rx mac=" MACSTR " node=%d rx=%u gen=%lu running=%u",
        MAC2STR(mac),
        node_index,
        (unsigned)packet->rx_index,
        (unsigned long)packet->generation,
        (unsigned)packet->running
    );
}

static void udp_rx_task(void *arg)
{
    uint8_t buffer[sizeof(udp_csi_packet_header_t) + APP_MAX_CSI_LEN];
    struct sockaddr_in bind_addr;

    (void)arg;

    memset(&bind_addr, 0, sizeof(bind_addr));
    bind_addr.sin_family = AF_INET;
    bind_addr.sin_port = htons(APP_UDP_PORT);
    bind_addr.sin_addr.s_addr = htonl(INADDR_ANY);

    while (true) {
        int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
        if (sock < 0) {
            ESP_LOGW(TAG, "udp socket create failed errno=%d", errno);
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        if (bind(sock, (const struct sockaddr *)&bind_addr, sizeof(bind_addr)) < 0) {
            ESP_LOGW(TAG, "udp bind failed errno=%d", errno);
            close(sock);
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        while (true) {
            udp_csi_packet_header_t *hdr = (udp_csi_packet_header_t *)buffer;
            tx_event_t evt;
            int recv_len = recvfrom(sock, buffer, sizeof(buffer), 0, NULL, NULL);
            uint32_t checksum_calculated;

            if (recv_len < 0) {
                ESP_LOGW(TAG, "udp recv failed errno=%d", errno);
                close(sock);
                break;
            }

            if (recv_len < (int)sizeof(*hdr) ||
                hdr->magic != APP_LINK_UDP_MAGIC ||
                hdr->version != APP_PROTOCOL_VERSION ||
                hdr->csi_len == 0u ||
                hdr->csi_len > APP_MAX_CSI_LEN ||
                recv_len != (int)(sizeof(*hdr) + hdr->csi_len)) {
                continue;
            }

            checksum_calculated = checksum32(buffer, offsetof(udp_csi_packet_header_t, checksum), 0u);
            checksum_calculated = checksum32(
                buffer + sizeof(*hdr),
                hdr->csi_len,
                checksum_calculated
            );
            if (checksum_calculated != hdr->checksum) {
                ESP_LOGW(
                    TAG,
                    "drop udp checksum rx=%u expected=0x%08" PRIx32 " got=0x%08" PRIx32,
                    (unsigned)hdr->rx_index,
                    checksum_calculated,
                    hdr->checksum
                );
                continue;
            }

            memset(&evt, 0, sizeof(evt));
            evt.type = TX_EVT_RX_RECORD;
            memcpy(&evt.data.rx_record.header, hdr, sizeof(*hdr));
            evt.data.rx_record.csi_len = hdr->csi_len;
            memcpy(evt.data.rx_record.csi, buffer + sizeof(*hdr), hdr->csi_len);
            if (!enqueue_tx_event(&evt, "udp_record")) {
                ESP_LOGW(
                    TAG,
                    "drop udp enqueue rx=%u csi_len=%u",
                    (unsigned)hdr->rx_index,
                    (unsigned)hdr->csi_len
                );
                continue;
            }

            portENTER_CRITICAL(&s_state_lock);
            {
                if (hdr->rx_index < s_state.running_node_count) {
                    uint8_t node_index = s_state.running_node_indices[hdr->rx_index];
                    s_state.nodes[node_index].last_seen_us = esp_timer_get_time();
                }
            }
            portEXIT_CRITICAL(&s_state_lock);
        }
    }
}

static void handle_ascii_command(char *line)
{
    char *ctx = NULL;
    char *cmd;

    if (line == NULL) {
        return;
    }

    while (*line != '\0' && isspace((unsigned char)*line)) {
        ++line;
    }
    if (*line == '\0') {
        return;
    }

    if (strncmp(line, APP_UART_CMD_PREFIX, strlen(APP_UART_CMD_PREFIX)) != 0) {
        return;
    }
    line += strlen(APP_UART_CMD_PREFIX);
    while (*line != '\0' && isspace((unsigned char)*line)) {
        ++line;
    }
    if (*line == '\0') {
        return;
    }

    cmd = strtok_r(line, " \t\r\n", &ctx);
    if (cmd == NULL) {
        return;
    }
    ESP_LOGI(TAG, "cmd rx: %s", cmd);

    if (strcmp(cmd, "s") == 0 || strcmp(cmd, "toggle") == 0) {
        tx_event_t evt = {
            .type = (s_state.mode == APP_MODE_WAIT) ? TX_EVT_START : TX_EVT_STOP,
        };
        const char *ack_message = (evt.type == TX_EVT_START) ? "running requested" : "wait requested";

        if (s_state.mode == APP_MODE_WAIT) {
            uint8_t tmp[APP_MAX_RX_NODES];
            portENTER_CRITICAL(&s_state_lock);
            if (collect_live_connected_nodes(tmp) == 0u) {
                portEXIT_CRITICAL(&s_state_lock);
                enqueue_ack(false, "no active rx");
                return;
            }
            portEXIT_CRITICAL(&s_state_lock);
        }
        if (!enqueue_tx_event(&evt, "cmd_toggle")) {
            enqueue_ack(false, "event queue busy");
        } else {
            enqueue_ack(true, ack_message);
        }
        return;
    }

    if (strcmp(cmd, "mode") == 0) {
        char *arg = strtok_r(NULL, " \t\r\n", &ctx);
        tx_event_t evt;

        if (arg == NULL) {
            enqueue_ack(false, "mode requires wait|run");
            return;
        }

        if (strcmp(arg, "run") == 0) {
            uint8_t tmp[APP_MAX_RX_NODES];
            portENTER_CRITICAL(&s_state_lock);
            if (collect_live_connected_nodes(tmp) == 0u) {
                portEXIT_CRITICAL(&s_state_lock);
                enqueue_ack(false, "no active rx");
                return;
            }
            portEXIT_CRITICAL(&s_state_lock);
            evt.type = TX_EVT_START;
            if (enqueue_tx_event(&evt, "cmd_mode_run")) {
                enqueue_ack(true, "running requested");
            } else {
                enqueue_ack(false, "event queue busy");
            }
            return;
        }

        if (strcmp(arg, "wait") == 0) {
            evt.type = TX_EVT_STOP;
            if (enqueue_tx_event(&evt, "cmd_mode_wait")) {
                enqueue_ack(true, "wait requested");
            } else {
                enqueue_ack(false, "event queue busy");
            }
            return;
        }

        enqueue_ack(false, "mode requires wait|run");
        return;
    }

    if (strcmp(cmd, "timeout_us") == 0) {
        char *arg = strtok_r(NULL, " \t\r\n", &ctx);
        unsigned long value;

        if (arg == NULL) {
            enqueue_ack(false, "timeout_us requires value");
            return;
        }

        portENTER_CRITICAL(&s_state_lock);
        if (s_state.mode != APP_MODE_WAIT) {
            portEXIT_CRITICAL(&s_state_lock);
            enqueue_ack(false, "change timeout in wait mode");
            return;
        }
        portEXIT_CRITICAL(&s_state_lock);

        value = strtoul(arg, NULL, 10);
        if (value < 500u || value > 1000000u) {
            enqueue_ack(false, "timeout range 500..1000000");
            return;
        }

        portENTER_CRITICAL(&s_state_lock);
        s_state.slot_timeout_us = (uint32_t)value;
        s_state.assignment_dirty = true;
        portEXIT_CRITICAL(&s_state_lock);
        save_settings_to_nvs();
        enqueue_ack(true, "timeout updated");
        return;
    }

    if (strcmp(cmd, "slot_gap_us") == 0) {
        char *arg = strtok_r(NULL, " \t\r\n", &ctx);
        unsigned long value;

        if (arg == NULL) {
            enqueue_ack(false, "slot_gap_us requires value");
            return;
        }

        portENTER_CRITICAL(&s_state_lock);
        if (s_state.mode != APP_MODE_WAIT) {
            portEXIT_CRITICAL(&s_state_lock);
            enqueue_ack(false, "change slot gap in wait mode");
            return;
        }
        portEXIT_CRITICAL(&s_state_lock);

        value = strtoul(arg, NULL, 10);
        if (value < 100u || value > 100000u) {
            enqueue_ack(false, "slot gap range 100..100000");
            return;
        }

        portENTER_CRITICAL(&s_state_lock);
        s_state.udp_slot_gap_us = (uint32_t)value;
        s_state.assignment_dirty = true;
        portEXIT_CRITICAL(&s_state_lock);
        save_settings_to_nvs();
        enqueue_ack(true, "slot gap updated");
        return;
    }

    if (strcmp(cmd, "save_nodes") == 0) {
        portENTER_CRITICAL(&s_state_lock);
        if (s_state.mode != APP_MODE_WAIT) {
            portEXIT_CRITICAL(&s_state_lock);
            enqueue_ack(false, "save in wait mode");
            return;
        }
        portEXIT_CRITICAL(&s_state_lock);
        snapshot_nodes_to_saved_config();
        enqueue_ack(true, "node order saved");
        return;
    }

    if (strcmp(cmd, "status") == 0) {
        enqueue_status_frame();
        enqueue_ack(true, "status queued");
        return;
    }

    if (strcmp(cmd, "channel") == 0) {
        char *ch_arg = strtok_r(NULL, " \t\r\n", &ctx);
        char *sc_arg = strtok_r(NULL, " \t\r\n", &ctx);
        unsigned long channel_value;
        wifi_second_chan_t second = WIFI_SECOND_CHAN_NONE;

        if (ch_arg == NULL || sc_arg == NULL) {
            enqueue_ack(false, "channel <1..13> <none|above|below>");
            return;
        }

        portENTER_CRITICAL(&s_state_lock);
        if (s_state.mode != APP_MODE_WAIT) {
            portEXIT_CRITICAL(&s_state_lock);
            enqueue_ack(false, "change channel in wait mode");
            return;
        }
        portEXIT_CRITICAL(&s_state_lock);

        channel_value = strtoul(ch_arg, NULL, 10);
        if (channel_value < 1u || channel_value > 13u) {
            enqueue_ack(false, "channel range 1..13");
            return;
        }

        if (strcmp(sc_arg, "above") == 0) {
            second = WIFI_SECOND_CHAN_ABOVE;
        } else if (strcmp(sc_arg, "below") == 0) {
            second = WIFI_SECOND_CHAN_BELOW;
        } else if (strcmp(sc_arg, "none") == 0) {
            second = WIFI_SECOND_CHAN_NONE;
        } else {
            enqueue_ack(false, "secondary none|above|below");
            return;
        }

        portENTER_CRITICAL(&s_state_lock);
        s_state.wifi_channel = (uint8_t)channel_value;
        s_state.second_channel = second;
        portEXIT_CRITICAL(&s_state_lock);
        save_settings_to_nvs();
        send_ctrl_packet(APP_CTRL_MSG_CHANNEL, NULL, 0xffu, 0u);
        enqueue_ack(true, "channel saved, rebooting");
        while (usb_tx_pending()) {
            (void)drain_usb_tx_queue(APP_UART_TX_DRAIN_CHUNKS_IDLE);
            vTaskDelay(pdMS_TO_TICKS(1));
        }
        vTaskDelay(pdMS_TO_TICKS(200));
        esp_restart();
        return;
    }

    enqueue_ack(false, "unknown command");
}

/* Core 1에서는 UART 입출력과 wait mode 상태 전송만 담당한다. */
static void control_task(void *arg)
{
    uint8_t rx_bytes[64];
    char line_buf[APP_UART_LINE_BUF_SIZE];
    size_t line_len = 0u;
    bool drop_line = false;
    uint32_t last_status_ms = 0u;

    (void)arg;

    while (true) {
        bool has_pending_tx;
        int read_len;

        read_len = tinyusb_read_available(rx_bytes, sizeof(rx_bytes));
        if (read_len > 0) {
            int i;

            for (i = 0; i < read_len; ++i) {
                uint8_t raw = rx_bytes[i];
                char c = (char)raw;

                if (c == '\r' || c == '\n') {
                    if (!drop_line && line_len > 0u) {
                        line_buf[line_len] = '\0';
                        handle_ascii_command(line_buf);
                    }
                    line_len = 0u;
                    drop_line = false;
                    continue;
                }

                /* 바이너리 프레임/에코가 UART RX로 되돌아와도 명령 줄로 해석하지 않도록
                 * 인쇄 가능한 ASCII와 탭만 허용한다. 하나라도 섞이면 개행까지 줄 전체를 버린다. */
                if ((raw < 0x20u || raw > 0x7eu) && c != '\t') {
                    line_len = 0u;
                    drop_line = true;
                    continue;
                }

                if (drop_line) {
                    continue;
                }

                if (line_len + 1u < sizeof(line_buf)) {
                    line_buf[line_len++] = c;
                } else {
                    line_len = 0u;
                    drop_line = true;
                }
            }
        }

        (void)drain_usb_tx_queue(
            read_len > 0 ? APP_UART_TX_DRAIN_CHUNKS_AFTER_RX : APP_UART_TX_DRAIN_CHUNKS_IDLE
        );
        has_pending_tx = usb_tx_pending();

        broadcast_assignments_if_needed();

        portENTER_CRITICAL(&s_state_lock);
        {
            app_mode_t mode_snapshot = s_state.mode;
            portEXIT_CRITICAL(&s_state_lock);

            if (mode_snapshot == APP_MODE_WAIT && (now_ms() - last_status_ms) >= APP_STATUS_INTERVAL_MS) {
                enqueue_status_frame();
                last_status_ms = now_ms();
            }
        }

        if (!has_pending_tx && read_len <= 0 && !usb_tx_pending()) {
            vTaskDelay(pdMS_TO_TICKS(2));
        }
    }
}

/* GUI와의 바이너리 양방향 링크는 TinyUSB CDC ACM 포트 하나로 통일한다. */
static void init_tinyusb(void)
{
    const tinyusb_config_t tusb_cfg = TINYUSB_DEFAULT_CONFIG();
    const tinyusb_config_cdcacm_t cdc_cfg = {
        .cdc_port = APP_USB_CDC_PORT,
        .callback_rx = NULL,
        .callback_rx_wanted_char = NULL,
        .callback_line_state_changed = NULL,
        .callback_line_coding_changed = NULL,
    };

    ESP_ERROR_CHECK(tinyusb_driver_install(&tusb_cfg));
    ESP_ERROR_CHECK(tinyusb_cdcacm_init(&cdc_cfg));
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();

    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    memset(&s_state, 0, sizeof(s_state));
    memset(&s_cycle, 0, sizeof(s_cycle));
    load_settings_from_nvs();

    init_tinyusb();
    esp_log_level_set("*", ESP_LOG_INFO);

    s_evt_queue = xQueueCreate(APP_EVENT_QUEUE_LEN, sizeof(tx_event_t));
    s_uart_tx_queue = xQueueCreate(APP_UART_QUEUE_LEN, sizeof(uart_tx_item_t));
    ESP_ERROR_CHECK(s_evt_queue != NULL ? ESP_OK : ESP_FAIL);
    ESP_ERROR_CHECK(s_uart_tx_queue != NULL ? ESP_OK : ESP_FAIL);

    init_wifi_softap();
    init_espnow();
    ESP_ERROR_CHECK(esp_now_register_recv_cb(espnow_recv_cb));
    init_timeout_timer();

    xTaskCreatePinnedToCore(coordinator_task, "tx_coord", 6144, NULL, 20, NULL, 0);
    xTaskCreatePinnedToCore(control_task, "tx_ctrl", 6144, NULL, 8, NULL, 1);
    xTaskCreatePinnedToCore(udp_rx_task, "tx_udp", 6144, NULL, 18, NULL, 1);

    ESP_LOGI(
        TAG,
        "boot complete timeout_us=%lu gap_us=%lu saved_nodes=%u channel=%u second=%u",
        (unsigned long)s_state.slot_timeout_us,
        (unsigned long)s_state.udp_slot_gap_us,
        (unsigned)s_state.saved_node_count,
        (unsigned)s_state.wifi_channel,
        (unsigned)s_state.second_channel
    );
    enqueue_ack(true, "tx ready");
    enqueue_status_frame();
}
