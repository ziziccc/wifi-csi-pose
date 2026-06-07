#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
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
#include "freertos/task.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"
#include "nvs.h"
#include "nvs_flash.h"

#define APP_WIFI_SSID                        "CSI_TX"
#define APP_WIFI_PROTOCOL_BITMAP             (WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N)
#define APP_DEFAULT_WIFI_CHANNEL             6
#define APP_DEFAULT_SECOND_CHANNEL           WIFI_SECOND_CHAN_ABOVE
#define APP_WIFI_BANDWIDTH                   WIFI_BW_HT40

#define APP_LINK_CTRL_MAGIC                  0x35544343u
#define APP_LINK_UDP_MAGIC                   0x35554450u
#define APP_LINK_HEARTBEAT_MAGIC             0x35554842u
#define APP_PROTOCOL_VERSION                 1u

#define APP_CTRL_MSG_ASSIGNMENT              1u
#define APP_CTRL_MSG_MODE                    2u
#define APP_CTRL_MSG_CHANNEL                 3u

#define APP_MAX_CSI_LEN                      512
#define APP_UDP_PORT                         3333
#define APP_TX_UDP_IP                        "192.168.4.1"
#define APP_DEFAULT_UDP_SLOT_GAP_US          2000u
#define APP_HEARTBEAT_INTERVAL_MS            1000u
#define APP_TRACE_HOT_PATH                   0

static const uint8_t s_broadcast_mac[ESP_NOW_ETH_ALEN] = {
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
};

#define HOT_LOGI(tag, fmt, ...) do { if (APP_TRACE_HOT_PATH) ESP_LOGI(tag, fmt, ##__VA_ARGS__); } while (0)

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
    bool running;
    uint32_t generation;
    uint32_t slot_timeout_us;
    uint32_t udp_slot_gap_us;
    uint8_t assigned_index;
    uint8_t active_node_count;
    uint8_t sta_mac[6];
    uint8_t tx_mac[6];
    uint8_t wifi_channel;
    wifi_second_chan_t second_channel;
    bool pending_valid;
    bool pending_sent;
    int8_t pending_rssi;
    uint16_t pending_csi_len;
    uint8_t pending_csi[APP_MAX_CSI_LEN];
} rx_state_t;

static rx_state_t s_state;
static portMUX_TYPE s_state_lock = portMUX_INITIALIZER_UNLOCKED;
static gptimer_handle_t s_send_timer;
static TaskHandle_t s_send_task;
static int s_udp_sock = -1;
static struct sockaddr_in s_tx_udp_addr;
static const char *TAG = "CSI_RX";

static void arm_send_timer(uint32_t timeout_us);
static bool ensure_udp_socket(void);

/* RX/TX가 공통으로 쓰는 체크섬이다. */
static uint32_t checksum32(const void *data, size_t len, uint32_t seed)
{
    const uint8_t *bytes = (const uint8_t *)data;
    uint32_t value = seed;
    size_t i;

    for (i = 0; i < len; ++i) {
        value = (value << 5) - value + bytes[i];
    }
    return value;
}

/* 트리거/슬롯 상태를 새 cycle 기준으로 초기화한다. */
static void clear_runtime_state_locked(void)
{
    s_state.pending_valid = false;
    s_state.pending_sent = false;
    s_state.pending_rssi = 0;
    s_state.pending_csi_len = 0u;
}

/* 채널 힌트와 second channel 설정을 NVS에 저장한다. */
static void save_channel_hint_to_nvs(uint8_t channel, wifi_second_chan_t second_channel)
{
    nvs_handle_t nvs = 0;

    if (nvs_open("csi_rx", NVS_READWRITE, &nvs) != ESP_OK) {
        return;
    }

    nvs_set_u32(nvs, "channel", channel);
    nvs_set_u32(nvs, "second", (uint32_t)second_channel);
    nvs_commit(nvs);
    nvs_close(nvs);
}

/* 부팅 시 채널 힌트를 읽어두면 TX 채널 변경 후 재연결이 빨라진다. */
static void load_channel_hint_from_nvs(uint8_t *channel_out, wifi_second_chan_t *second_out)
{
    nvs_handle_t nvs = 0;
    uint32_t value = 0u;

    *channel_out = APP_DEFAULT_WIFI_CHANNEL;
    *second_out = APP_DEFAULT_SECOND_CHANNEL;

    if (nvs_open("csi_rx", NVS_READONLY, &nvs) != ESP_OK) {
        return;
    }

    if (nvs_get_u32(nvs, "channel", &value) == ESP_OK && value >= 1u && value <= 13u) {
        *channel_out = (uint8_t)value;
    }
    if (nvs_get_u32(nvs, "second", &value) == ESP_OK &&
        value <= (uint32_t)WIFI_SECOND_CHAN_BELOW) {
        *second_out = (wifi_second_chan_t)value;
    }
    nvs_close(nvs);
}

/* RX는 TX SoftAP가 보일 때까지 STA 모드로 계속 재시도한다. */
static void wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data)
{
    (void)arg;
    (void)event_data;

    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        ESP_LOGI(TAG, "wifi sta start -> connect");
        esp_wifi_connect();
        return;
    }

    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "wifi disconnected -> reconnect");
        esp_wifi_connect();
    }
}

/* RX의 Wi-Fi는 HT40/bgn 조합으로 고정하고, TX AP 채널 힌트를 사용한다. */
static void init_wifi_sta(void)
{
    wifi_init_config_t init_cfg = WIFI_INIT_CONFIG_DEFAULT();
    wifi_config_t sta_cfg = {
        .sta = {
            .scan_method = WIFI_FAST_SCAN,
            .sort_method = WIFI_CONNECT_AP_BY_SIGNAL,
            .threshold.authmode = WIFI_AUTH_OPEN,
            .pmf_cfg = {
                .capable = false,
                .required = false,
            },
        },
    };

    load_channel_hint_from_nvs(&s_state.wifi_channel, &s_state.second_channel);
    memcpy(sta_cfg.sta.ssid, APP_WIFI_SSID, strlen(APP_WIFI_SSID));
    sta_cfg.sta.channel = s_state.wifi_channel;

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    ESP_ERROR_CHECK(esp_wifi_init(&init_cfg));
    ESP_ERROR_CHECK(
        esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL)
    );
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_wifi_set_protocol(WIFI_IF_STA, APP_WIFI_PROTOCOL_BITMAP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta_cfg));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, APP_WIFI_BANDWIDTH));
    ESP_ERROR_CHECK(esp_wifi_get_mac(WIFI_IF_STA, s_state.sta_mac));
    ESP_LOGI(
        TAG,
        "wifi sta ready ssid=%s ch=%u second=%u mac=" MACSTR,
        APP_WIFI_SSID,
        (unsigned)s_state.wifi_channel,
        (unsigned)s_state.second_channel,
        MAC2STR(s_state.sta_mac)
    );
}

/* ESP-NOW broadcast peer를 추가해 TX와 다른 RX 모두에게 패킷을 뿌릴 수 있게 한다. */
static void init_espnow(void)
{
    esp_now_peer_info_t peer = {0};

    ESP_ERROR_CHECK(esp_now_init());
    memcpy(peer.peer_addr, s_broadcast_mac, ESP_NOW_ETH_ALEN);
    peer.ifidx = WIFI_IF_STA;
    peer.channel = 0;
    peer.encrypt = false;
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
}

static bool ensure_udp_socket(void)
{
    if (s_udp_sock >= 0) {
        return true;
    }

    s_udp_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (s_udp_sock < 0) {
        ESP_LOGW(TAG, "udp socket create failed errno=%d", errno);
        return false;
    }

    memset(&s_tx_udp_addr, 0, sizeof(s_tx_udp_addr));
    s_tx_udp_addr.sin_family = AF_INET;
    s_tx_udp_addr.sin_port = htons(APP_UDP_PORT);
    s_tx_udp_addr.sin_addr.s_addr = inet_addr(APP_TX_UDP_IP);
    return true;
}

/* custom data trigger를 promiscuous로 잡아서 CSI callback과 매칭한다. */
#if 0
static bool parse_trigger_frame(const wifi_promiscuous_pkt_t *pkt, trigger_hint_t *hint)
{
    const uint8_t *payload;
    const trigger_data_frame_t *frame;

    if (pkt == NULL || hint == NULL) {
        return false;
    }

    if (pkt->rx_ctrl.sig_len < sizeof(trigger_data_frame_t)) {
        return false;
    }

    payload = pkt->payload;
    frame = (const trigger_data_frame_t *)payload;

    if (((frame->frame_ctrl >> 2) & 0x3u) != 2u) {
        return false;
    }
    if (((frame->frame_ctrl >> 4) & 0x0fu) != 0u) {
        return false;
    }
    if (frame->llc_dsap != 0xAA ||
        frame->llc_ssap != 0xAA ||
        frame->llc_control != 0x03 ||
        frame->protocol_id[0] != 0x88 ||
        frame->protocol_id[1] != 0xB5 ||
        frame->magic != APP_TRIGGER_DATA_MAGIC ||
        frame->version != APP_PROTOCOL_VERSION ||
        frame->frame_kind != 1u) {
        return false;
    }

    memset(hint, 0, sizeof(*hint));
    hint->valid = true;
    hint->consumed = false;
    hint->generation = frame->generation;
    hint->trigger_seq = frame->trigger_seq;
    hint->received_at_us = (uint64_t)esp_timer_get_time();
    hint->active_nodes = frame->active_nodes;
    memcpy(hint->source_mac, frame->addr2, 6);
    ESP_LOGI(
        TAG,
        "trigger seen gen=%lu trig=%lu active=%u from=" MACSTR,
        (unsigned long)hint->generation,
        (unsigned long)hint->trigger_seq,
        (unsigned)hint->active_nodes,
        MAC2STR(hint->source_mac)
    );
    return true;
}

/* 새 trigger를 기준으로 이전 슬롯 RX 패킷 대기 상태도 함께 갱신한다. */
static void store_trigger_hint_locked(const trigger_hint_t *hint)
{
    s_state.latest_trigger = *hint;
    s_state.active_node_count = hint->active_nodes;
    s_state.prev_slot.active = s_state.assigned_index > 0u;
    s_state.prev_slot.complete = false;
    s_state.prev_slot.generation = hint->generation;
    s_state.prev_slot.trigger_seq = hint->trigger_seq;
    s_state.prev_slot.expected_prev_index = (uint8_t)(s_state.assigned_index - 1u);
    s_state.prev_slot.fragment_count = 0u;
    s_state.prev_slot.fragment_mask = 0u;
    s_state.prev_slot.total_len = 0u;
}

/* CSI callback에서 같은 source MAC의 가장 최근 trigger만 가져온다. */
#endif
static bool build_pending_csi_locked(
    int8_t rssi,
    const uint8_t *csi,
    uint16_t csi_len,
    uint8_t *assigned_index_out,
    uint16_t *pending_csi_len_out,
    bool *notify_now_out
)
{
    if (csi == NULL || csi_len == 0u || csi_len > APP_MAX_CSI_LEN) {
        return false;
    }

    s_state.pending_valid = true;
    s_state.pending_sent = false;
    s_state.pending_rssi = rssi;
    s_state.pending_csi_len = csi_len;
    memcpy(s_state.pending_csi, csi, csi_len);

    *assigned_index_out = s_state.assigned_index;
    *pending_csi_len_out = s_state.pending_csi_len;

    if (s_state.assigned_index == 0u || s_state.active_node_count <= 1u) {
        *notify_now_out = true;
    } else {
        *notify_now_out = false;
        arm_send_timer((uint32_t)s_state.assigned_index * s_state.udp_slot_gap_us);
    }
    return true;
}


/* trigger frame만 따로 걸러서 state에 저장한다. */
#if 0
static void rx_promisc_cb(void *buf, wifi_promiscuous_pkt_type_t type)
{
    trigger_hint_t hint;
    bool running = false;
    uint8_t current_assigned_index = 0u;

    if ((type != WIFI_PKT_MGMT && type != WIFI_PKT_DATA) || buf == NULL) {
        return;
    }

    if (!parse_trigger_frame((const wifi_promiscuous_pkt_t *)buf, &hint)) {
        return;
    }

    portENTER_CRITICAL(&s_state_lock);
    running = s_state.running;
    current_assigned_index = s_state.assigned_index;
    if (running) {
        store_trigger_hint_locked(&hint);
    }
    portEXIT_CRITICAL(&s_state_lock);
    HOT_LOGI(
        TAG,
        "trigger stored running=%u assigned=%u gen=%lu trig=%lu active=%u",
        (unsigned)running,
        (unsigned)current_assigned_index,
        (unsigned long)hint.generation,
        (unsigned long)hint.trigger_seq,
        (unsigned)hint.active_nodes
    );
}

/* send 타이머가 울리면 차례가 된 것으로 보고 send task를 깨운다. */
#endif
static bool IRAM_ATTR send_alarm_cb(
    gptimer_handle_t timer,
    const gptimer_alarm_event_data_t *edata,
    void *user_ctx
)
{
    BaseType_t high_woken = pdFALSE;
    TaskHandle_t task = (TaskHandle_t)user_ctx;

    (void)timer;
    (void)edata;

    if (task != NULL) {
        vTaskNotifyGiveFromISR(task, &high_woken);
    }
    return high_woken == pdTRUE;
}

/* slot timeout도 gptimer one-shot으로 맞춘다. */
static void init_send_timer(void)
{
    gptimer_config_t cfg = {
        .clk_src = GPTIMER_CLK_SRC_DEFAULT,
        .direction = GPTIMER_COUNT_UP,
        .resolution_hz = 1000000,
    };
    gptimer_event_callbacks_t callbacks = {
        .on_alarm = send_alarm_cb,
    };

    ESP_ERROR_CHECK(gptimer_new_timer(&cfg, &s_send_timer));
    ESP_ERROR_CHECK(gptimer_register_event_callbacks(s_send_timer, &callbacks, s_send_task));
    ESP_ERROR_CHECK(gptimer_enable(s_send_timer));
}

/* assigned index에 비례한 fallback timeout을 시작한다. */
static void arm_send_timer(uint32_t timeout_us)
{
    gptimer_alarm_config_t alarm_cfg = {
        .reload_count = 0,
        .alarm_count = timeout_us,
        .flags.auto_reload_on_alarm = false,
    };

    gptimer_stop(s_send_timer);
    gptimer_set_raw_count(s_send_timer, 0);
    gptimer_set_alarm_action(s_send_timer, &alarm_cfg);
    gptimer_start(s_send_timer);
}

/* CSI metadata 전체를 GUI에서 풀기 쉽도록 고정 메타 구조로 옮긴다. */
/* CSI callback은 원본 메타 + raw CSI를 node record로 만들고 전송 준비만 한다. */
static void rx_csi_cb(void *ctx, wifi_csi_info_t *info)
{
    uint16_t csi_len;
    bool notify_now = false;
    uint8_t assigned_index = 0u;
    uint16_t pending_csi_len = 0u;
    uint8_t tx_mac[6];

    (void)ctx;

    if (info == NULL || info->buf == NULL || info->len <= 0) {
        ESP_LOGW(TAG, "rx_csi_cb invalid info len=%d", (info != NULL) ? info->len : -1);
        return;
    }

    HOT_LOGI(
        TAG,
        "rx_csi_cb enter len=%d src=" MACSTR " rssi=%d sig_mode=%u mcs=%u cwb=%u",
        info->len,
        MAC2STR(info->mac),
        info->rx_ctrl.rssi,
        (unsigned)info->rx_ctrl.sig_mode,
        (unsigned)info->rx_ctrl.mcs,
        (unsigned)info->rx_ctrl.cwb
    );

    portENTER_CRITICAL(&s_state_lock);
    if (!s_state.running) {
        portEXIT_CRITICAL(&s_state_lock);
        ESP_LOGW(TAG, "rx_csi_cb drop: not running");
        return;
    }
    memcpy(tx_mac, s_state.tx_mac, sizeof(tx_mac));
    if (memcmp(info->mac, tx_mac, sizeof(tx_mac)) != 0) {
        portEXIT_CRITICAL(&s_state_lock);
        HOT_LOGI(
            TAG,
            "rx_csi_cb drop: foreign src=" MACSTR " expected=" MACSTR,
            MAC2STR(info->mac),
            MAC2STR(tx_mac)
        );
        return;
    }
    csi_len = (uint16_t)info->len;
    if (csi_len > APP_MAX_CSI_LEN) {
        csi_len = APP_MAX_CSI_LEN;
    }
    csi_len = (uint16_t)(csi_len & ~1u);
    if (csi_len == 0u) {
        portEXIT_CRITICAL(&s_state_lock);
        ESP_LOGW(
            TAG,
            "rx_csi_cb drop: size reject csi_len=%u",
            (unsigned)csi_len
        );
        return;
    }

    if (!build_pending_csi_locked(
            info->rx_ctrl.rssi,
            (const uint8_t *)info->buf,
            csi_len,
            &assigned_index,
            &pending_csi_len,
            &notify_now)) {
        portEXIT_CRITICAL(&s_state_lock);
        ESP_LOGW(
            TAG,
            "rx_csi_cb drop: build reject csi_len=%u",
            (unsigned)csi_len
        );
        return;
    }
    portEXIT_CRITICAL(&s_state_lock);

    HOT_LOGI(
        TAG,
        "csi ready rx=%u rssi=%d csi_len=%u src=" MACSTR,
        (unsigned)assigned_index,
        (int)info->rx_ctrl.rssi,
        (unsigned)pending_csi_len,
        MAC2STR(info->mac)
    );

    if (notify_now && s_send_task != NULL) {
        HOT_LOGI(TAG, "send notify immediately rx=%u", (unsigned)assigned_index);
        xTaskNotifyGive(s_send_task);
    }
}

/* CSI capture를 위해 promiscuous + CSI callback을 동시에 켠다. */
static void init_csi(void)
{
    wifi_csi_config_t cfg = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = false,
    };

    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(rx_csi_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

/* TX가 보낸 assignment/mode/channel 패킷을 적용한다. */
static void handle_ctrl_packet(const link_ctrl_packet_t *packet)
{
    if (packet->msg_type == APP_CTRL_MSG_ASSIGNMENT) {
        if (memcmp(packet->target_mac, s_state.sta_mac, 6) == 0) {
            portENTER_CRITICAL(&s_state_lock);
            s_state.assigned_index = packet->slot_index;
            s_state.active_node_count = packet->total_nodes;
            s_state.slot_timeout_us = packet->timeout_us;
            s_state.udp_slot_gap_us = packet->udp_slot_gap_us;
            memcpy(s_state.tx_mac, packet->tx_mac, 6);
            portEXIT_CRITICAL(&s_state_lock);
            ESP_LOGI(
                TAG,
                "assignment rx slot=%u total=%u timeout_us=%lu gap_us=%lu tx=" MACSTR,
                (unsigned)packet->slot_index,
                (unsigned)packet->total_nodes,
                (unsigned long)packet->timeout_us,
                (unsigned long)packet->udp_slot_gap_us,
                MAC2STR(packet->tx_mac)
            );
        }
        return;
    }

    if (packet->msg_type == APP_CTRL_MSG_MODE) {
        portENTER_CRITICAL(&s_state_lock);
        s_state.running = packet->running ? true : false;
        s_state.generation = packet->generation;
        s_state.active_node_count = packet->total_nodes;
        s_state.slot_timeout_us = packet->timeout_us;
        s_state.udp_slot_gap_us = packet->udp_slot_gap_us;
        memcpy(s_state.tx_mac, packet->tx_mac, 6);
        clear_runtime_state_locked();
        portEXIT_CRITICAL(&s_state_lock);

        if (!packet->running) {
            gptimer_stop(s_send_timer);
        }
        ESP_LOGI(
            TAG,
            "mode rx running=%u gen=%lu total=%u timeout_us=%lu gap_us=%lu",
            (unsigned)packet->running,
            (unsigned long)packet->generation,
            (unsigned)packet->total_nodes,
            (unsigned long)packet->timeout_us,
            (unsigned long)packet->udp_slot_gap_us
        );
        return;
    }

    if (packet->msg_type == APP_CTRL_MSG_CHANNEL) {
        ESP_LOGW(
            TAG,
            "channel update ch=%u second=%u -> reboot",
            (unsigned)packet->wifi_channel,
            (unsigned)packet->second_channel
        );
        save_channel_hint_to_nvs(packet->wifi_channel, (wifi_second_chan_t)packet->second_channel);
        esp_restart();
    }
}

/* 이전 슬롯 패킷을 다 받으면 바로 자기 CSI를 보내게 한다. */
#if 0
static void handle_prev_slot_fragment(const link_data_fragment_header_t *hdr)
{
    bool notify_now = false;
    bool complete_now = false;
    uint8_t expected_prev_index = 0u;

    portENTER_CRITICAL(&s_state_lock);
    if (!s_state.running ||
        !s_state.prev_slot.active ||
        s_state.prev_slot.complete ||
        hdr->rx_index != s_state.prev_slot.expected_prev_index ||
        hdr->total_len > APP_MAX_NODE_RECORD_LEN ||
        (uint32_t)hdr->chunk_offset + (uint32_t)hdr->chunk_len > (uint32_t)hdr->total_len) {
        portEXIT_CRITICAL(&s_state_lock);
        return;
    }

    s_state.prev_slot.fragment_count = hdr->fragment_count;
    s_state.prev_slot.total_len = hdr->total_len;
    s_state.prev_slot.fragment_mask |= (uint8_t)(1u << hdr->fragment_index);

    if (s_state.prev_slot.fragment_count > 0u &&
        s_state.prev_slot.fragment_mask ==
            (uint8_t)((1u << s_state.prev_slot.fragment_count) - 1u)) {
        s_state.prev_slot.complete = true;
        complete_now = true;
        expected_prev_index = s_state.prev_slot.expected_prev_index;
        if (s_state.pending_valid && !s_state.pending_sent) {
            notify_now = true;
        }
    }
    portEXIT_CRITICAL(&s_state_lock);

    HOT_LOGI(
        TAG,
        "prev fragment rx from slot=%u frag=%u/%u",
        (unsigned)hdr->rx_index,
        (unsigned)hdr->fragment_index + 1u,
        (unsigned)hdr->fragment_count
    );
    if (complete_now) {
        HOT_LOGI(TAG, "prev slot complete expected_prev=%u", (unsigned)expected_prev_index);
    }

    if (notify_now && s_send_task != NULL) {
        xTaskNotifyGive(s_send_task);
    }
}
#endif

/* ESP-NOW 수신 콜백은 제어 패킷과 이전 슬롯 데이터 감지만 담당한다. */
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
static void espnow_recv_cb(const esp_now_recv_info_t *recv_info, const uint8_t *data, int data_len)
#else
static void espnow_recv_cb(const uint8_t *src_mac, const uint8_t *data, int data_len)
#endif
{
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
    (void)recv_info;
#else
    (void)src_mac;
#endif

    if (data == NULL || data_len < 8) {
        return;
    }

    if (*(const uint32_t *)data == APP_LINK_CTRL_MAGIC &&
        data_len == (int)sizeof(link_ctrl_packet_t)) {
        const link_ctrl_packet_t *packet = (const link_ctrl_packet_t *)data;
        if (packet->checksum == checksum32(packet, offsetof(link_ctrl_packet_t, checksum), 0u)) {
            handle_ctrl_packet(packet);
        } else {
            ESP_LOGW(TAG, "drop ctrl checksum mismatch");
        }
        return;
    }

}

/* 실제 ESP-NOW 송신은 send task에서만 해서 callback 경합을 줄인다. */
static void send_task(void *arg)
{
    uint8_t packet_buf[sizeof(udp_csi_packet_header_t) + APP_MAX_CSI_LEN];
    uint8_t csi_copy[APP_MAX_CSI_LEN];

    (void)arg;

    while (true) {
        udp_csi_packet_header_t *hdr = (udp_csi_packet_header_t *)packet_buf;
        uint16_t csi_len;
        uint8_t rx_index;
        int8_t rssi;
        uint32_t generation;

        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        portENTER_CRITICAL(&s_state_lock);
        if (!s_state.running || !s_state.pending_valid || s_state.pending_sent) {
            portEXIT_CRITICAL(&s_state_lock);
            continue;
        }

        csi_len = s_state.pending_csi_len;
        rx_index = s_state.assigned_index;
        rssi = s_state.pending_rssi;
        generation = s_state.generation;
        memcpy(csi_copy, s_state.pending_csi, csi_len);
        s_state.pending_sent = true;
        gptimer_stop(s_send_timer);
        portEXIT_CRITICAL(&s_state_lock);

        if (!ensure_udp_socket()) {
            continue;
        }

        memset(hdr, 0, sizeof(*hdr));
        hdr->magic = APP_LINK_UDP_MAGIC;
        hdr->version = APP_PROTOCOL_VERSION;
        hdr->rx_index = rx_index;
        hdr->rssi = rssi;
        hdr->csi_len = csi_len;
        hdr->generation = generation;
        memcpy(packet_buf + sizeof(*hdr), csi_copy, csi_len);
        hdr->checksum = checksum32(packet_buf, offsetof(udp_csi_packet_header_t, checksum), 0u);
        hdr->checksum = checksum32(packet_buf + sizeof(*hdr), csi_len, hdr->checksum);

        HOT_LOGI(
            TAG,
            "udp send rx=%u rssi=%d csi_len=%u",
            (unsigned)rx_index,
            (int)rssi,
            (unsigned)csi_len
        );

        if (sendto(
                s_udp_sock,
                packet_buf,
                sizeof(*hdr) + csi_len,
                0,
                (const struct sockaddr *)&s_tx_udp_addr,
                sizeof(s_tx_udp_addr)) < 0) {
            ESP_LOGW(TAG, "udp send failed errno=%d", errno);
            close(s_udp_sock);
            s_udp_sock = -1;
        }
    }
}

static void heartbeat_task(void *arg)
{
    (void)arg;

    while (true) {
        link_heartbeat_packet_t packet;
        bool should_send;

        vTaskDelay(pdMS_TO_TICKS(APP_HEARTBEAT_INTERVAL_MS));

        memset(&packet, 0, sizeof(packet));
        portENTER_CRITICAL(&s_state_lock);
        should_send = !s_state.running;
        packet.magic = APP_LINK_HEARTBEAT_MAGIC;
        packet.version = APP_PROTOCOL_VERSION;
        packet.rx_index = s_state.assigned_index;
        packet.running = s_state.running ? 1u : 0u;
        packet.generation = s_state.generation;
        memcpy(packet.sta_mac, s_state.sta_mac, 6);
        memcpy(packet.tx_mac, s_state.tx_mac, 6);
        portEXIT_CRITICAL(&s_state_lock);

        if (!should_send) {
            continue;
        }

        packet.checksum = checksum32(&packet, offsetof(link_heartbeat_packet_t, checksum), 0u);
        (void)esp_now_send(s_broadcast_mac, (const uint8_t *)&packet, sizeof(packet));
        HOT_LOGI(TAG, "heartbeat tx rx=%u gen=%lu", (unsigned)packet.rx_index, (unsigned long)packet.generation);
    }
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
    s_state.udp_slot_gap_us = APP_DEFAULT_UDP_SLOT_GAP_US;
    esp_log_level_set("*", ESP_LOG_INFO);

    init_wifi_sta();
    xTaskCreatePinnedToCore(send_task, "rx_send", 6144, NULL, 20, &s_send_task, 1);
    init_send_timer();
    init_espnow();
    ESP_ERROR_CHECK(esp_now_register_recv_cb(espnow_recv_cb));
    xTaskCreatePinnedToCore(heartbeat_task, "rx_heartbeat", 3072, NULL, 5, NULL, 1);
    init_csi();
    ESP_LOGI(TAG, "boot complete");
}
