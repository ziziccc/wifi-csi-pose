#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#define SERIAL_MAGIC 0x35534943u
#define FRAME_CYCLE 2u
#define MAX_FRAME_PAYLOAD 12288u
#define RX_BUFFER_SIZE (256u * 1024u)
#define READ_CHUNK_SIZE 16384u

#define DEFAULT_TTY "/dev/ttyACM0"
#define DEFAULT_BAUDRATE B115200

#define DEFAULT_CTRL_PHYS 0x40000000u
#define DEFAULT_INPUT_PHYS 0x1E000000u
#define DEFAULT_INPUT1_PHYS 0x1E004000u
#define DEFAULT_WEIGHTS_PHYS 0x1E020000u
#define DEFAULT_OUT_PHYS 0x1E090000u
#define DEFAULT_WEIGHT_BIN "pl_accel_v6_weights.bin"
#define DEFAULT_TIMEOUT_US 1000
#define DEFAULT_SLOT_GAP_US 500
#define DEFAULT_WIFI_CHANNEL 6
#define DEFAULT_SECOND_CHANNEL "above"
#define DEFAULT_ESP_PAYLOAD 246
#define DEFAULT_ESP_PPS 300
#define DEFAULT_PRINT_EVERY 10

#define CTRL_MAP_SIZE 0x10000u
#define INPUT_CHANNELS 9u
#define NODE_COUNT 3u
#define INPUT_H 128u
#define INPUT_W 10u
#define INPUT_SIZE (INPUT_CHANNELS * INPUT_H * INPUT_W)
#define POSE_DIM 24u
#define WINDOW_STRIDE 10u
#define MAX_FILL_GAP 3u
#define MAX_CSI_PAIRS 512u
#define PL_WEIGHT_MAGIC 0x36574C50u
#define PL_WEIGHT_VERSION 2u
#define PL_WEIGHT_WORDS 105748u
#define PL_WEIGHT_BYTES (PL_WEIGHT_WORDS * 4u)
#define PL_WEIGHT_OFF_TOTAL_WORDS 2u
#define PL_WEIGHT_OFF_INPUT_SCALE_BITS 3u

#define AP_CTRL 0x00u
#define ADDR_INPUT_R 0x10u
#define ADDR_WEIGHTS 0x1Cu
#define ADDR_COMMAND 0x28u
#define ADDR_FINAL_POSE 0x30u
#define HLS_COMMAND_INFER 0u
#define HLS_COMMAND_LOAD_WEIGHTS 1u

#define INPUT_BUFFER_COUNT 2u
#define HLS_TIMEOUT_MS 1000

typedef struct {
    const char *tty_path;
    const char *weight_path;
    uint32_t ctrl_phys;
    uint32_t input_phys;
    uint32_t input1_phys;
    uint32_t weights_phys;
    uint32_t out_phys;
    int timeout_us;
    int slot_gap_us;
    int wifi_channel;
    const char *second_channel;
    int esp_payload;
    int esp_pps;
    bool send_esp_config;
    bool legacy_payload_pps;
    int max_infer;
    int status_interval;
    int print_every;
    bool verbose;
    bool quiet;
    bool results_only;
    bool load_weights_only;
} options_t;

typedef struct {
    uint8_t *ptr;
    size_t len;
    size_t cap;
} rx_buffer_t;

typedef struct {
    int rx_index;
    int present;
    int rssi;
    const int8_t *csi;
    uint16_t csi_len;
} cycle_record_t;

typedef struct {
    uint32_t trigger_seq;
    uint8_t active_nodes;
    uint8_t received_nodes;
    cycle_record_t records[NODE_COUNT];
} cycle_t;

typedef struct {
    float window[INPUT_W][INPUT_CHANNELS][INPUT_H];
    float raw_base[INPUT_W][NODE_COUNT][INPUT_H];
    float raw_mask[INPUT_W][NODE_COUNT];
    float stream_last_valid[NODE_COUNT][INPUT_H];
    float prime_last_valid[NODE_COUNT][INPUT_H];
    unsigned stream_gap[NODE_COUNT];
    unsigned prime_gap[NODE_COUNT];
    bool stream_has_last_valid[NODE_COUNT];
    bool prime_has_last_valid[NODE_COUNT];
    unsigned count;
    unsigned stride_count;
    unsigned window_present_count;
    unsigned window_drop_count;
    unsigned window_max_gap;
} feature_state_t;

typedef struct {
    int8_t *ptr;
    uint32_t phys;
    bool ready;
    bool in_use;
    uint32_t trigger_seq;
    uint64_t seq;
    struct timespec usb_rx_ts;
    struct timespec data_start_ts;
    struct timespec data_ready_ts;
    struct timespec pl_in_ts;
    struct timespec pl_out_ts;
    bool dropped_old_before_publish;
    double esp_gap_ms;
    unsigned esp_present_count;
    unsigned esp_drop_count;
    unsigned esp_max_gap;
    uint8_t esp_active_nodes;
    uint8_t esp_received_nodes;
} input_slot_t;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t can_produce;
    pthread_cond_t can_consume;
    input_slot_t slots[INPUT_BUFFER_COUNT];
    uint64_t next_seq;
} hls_input_queue_t;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t has_data;
    rx_buffer_t rx;
    struct timespec last_rx_ts;
} shared_rx_t;

typedef struct {
    pthread_mutex_t mutex;
    uint64_t cycles;
    uint64_t ready_inputs;
    int inferences;
    unsigned window_count;
} runtime_stats_t;

typedef struct {
    int tty_fd;
    shared_rx_t *shared_rx;
} usb_thread_args_t;

typedef struct {
    shared_rx_t *shared_rx;
    hls_input_queue_t *queue;
    runtime_stats_t *stats;
} parser_thread_args_t;


static volatile sig_atomic_t g_stop = 0;
static bool g_verbose = false;
static struct timespec g_time_origin;
static float g_input_scale = 1.0f;

static void on_signal(int sig)
{
    (void)sig;
    g_stop = 1;
}

static uint16_t rd16(const uint8_t *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t rd64(const uint8_t *p)
{
    uint64_t lo = rd32(p);
    uint64_t hi = rd32(p + 4);
    return lo | (hi << 32);
}

static uint32_t checksum32(const uint8_t *a, size_t a_len, const uint8_t *b, size_t b_len)
{
    uint32_t value = 0;
    for (size_t i = 0; i < a_len; ++i) {
        value = ((value << 5) - value + a[i]) & 0xFFFFFFFFu;
    }
    for (size_t i = 0; i < b_len; ++i) {
        value = ((value << 5) - value + b[i]) & 0xFFFFFFFFu;
    }
    return value;
}

static int setup_tty_raw(int fd)
{
    struct termios tio;
    if (tcgetattr(fd, &tio) != 0) {
        perror("tcgetattr");
        return -1;
    }
    cfmakeraw(&tio);
    cfsetispeed(&tio, DEFAULT_BAUDRATE);
    cfsetospeed(&tio, DEFAULT_BAUDRATE);
    tio.c_cflag |= CLOCAL | CREAD;
    tio.c_cflag &= ~CRTSCTS;
    tio.c_cc[VMIN] = 0;
    tio.c_cc[VTIME] = 1;
    if (tcsetattr(fd, TCSANOW, &tio) != 0) {
        perror("tcsetattr");
        return -1;
    }
    tcflush(fd, TCIOFLUSH);
    return 0;
}

static int write_all(int fd, const char *s)
{
    size_t off = 0;
    size_t len = strlen(s);
    while (off < len) {
        ssize_t n = write(fd, s + off, len - off);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        off += (size_t)n;
    }
    return 0;
}

static int send_esp_line(int fd, const char *line)
{
    char command[128];
    int n = snprintf(command, sizeof(command), "CMD %s\n", line);
    if (n <= 0 || (size_t)n >= sizeof(command)) {
        fprintf(stderr, "ESP command too long: %s\n", line);
        return -1;
    }
    if (write_all(fd, command) != 0) {
        perror("write ESP command");
        return -1;
    }
    return 0;
}

static int configure_esp_stream(int fd, const options_t *opt)
{
    char line[128];

    if (send_esp_line(fd, "mode wait") != 0) {
        return -1;
    }
    usleep(20000);

    if (opt->send_esp_config) {
        snprintf(line, sizeof(line), "timeout_us %d", opt->timeout_us);
        if (send_esp_line(fd, line) != 0) {
            return -1;
        }
        usleep(10000);

        snprintf(line, sizeof(line), "slot_gap_us %d", opt->slot_gap_us);
        if (send_esp_line(fd, line) != 0) {
            return -1;
        }
        usleep(10000);

        snprintf(line, sizeof(line), "channel %d %s", opt->wifi_channel, opt->second_channel);
        if (send_esp_line(fd, line) != 0) {
            return -1;
        }
        usleep(10000);

        if (opt->legacy_payload_pps) {
            snprintf(line, sizeof(line), "payload %d", opt->esp_payload);
            if (send_esp_line(fd, line) != 0) {
                return -1;
            }
            usleep(10000);

            snprintf(line, sizeof(line), "pps %d", opt->esp_pps);
            if (send_esp_line(fd, line) != 0) {
                return -1;
            }
            usleep(10000);
        }

        if (send_esp_line(fd, "status") != 0) {
            return -1;
        }
        usleep(10000);
    }

    if (send_esp_line(fd, "mode run") != 0) {
        return -1;
    }
    return 0;
}

static void *map_phys(int mem_fd, uint32_t phys, size_t size)
{
    long page_size = sysconf(_SC_PAGESIZE);
    uint32_t page_mask = (uint32_t)page_size - 1u;
    off_t base = (off_t)(phys & ~page_mask);
    size_t offset = (size_t)(phys & page_mask);
    size_t map_size = offset + size;
    void *mapped = mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, base);
    if (mapped == MAP_FAILED) {
        return MAP_FAILED;
    }
    return (uint8_t *)mapped + offset;
}

static inline void reg_write(volatile uint32_t *base, uint32_t off, uint32_t val)
{
    base[off / 4u] = val;
}

static inline uint32_t reg_read(volatile uint32_t *base, uint32_t off)
{
    return base[off / 4u];
}

static void reg_write_u64(volatile uint32_t *base, uint32_t off, uint64_t val)
{
    reg_write(base, off, (uint32_t)val);
    reg_write(base, off + 4u, (uint32_t)(val >> 32));
}

static float u32_to_float(uint32_t bits)
{
    union {
        uint32_t u;
        float f;
    } value;
    value.u = bits;
    return value.f;
}

static int load_file(const char *path, void *dst, size_t expected_size)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror(path);
        return -1;
    }
    uint8_t *out = (uint8_t *)dst;
    size_t off = 0;
    while (off < expected_size) {
        ssize_t n = read(fd, out + off, expected_size - off);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            perror("read weight");
            close(fd);
            return -1;
        }
        if (n == 0) {
            fprintf(stderr, "weight file too short: expected %zu bytes, got %zu\n", expected_size, off);
            close(fd);
            return -1;
        }
        off += (size_t)n;
    }
    close(fd);
    return 0;
}

static int validate_loaded_weights(const uint32_t *weights, float *input_scale)
{
    if (weights[0] != PL_WEIGHT_MAGIC) {
        fprintf(stderr, "bad PL weight magic: 0x%08x expected 0x%08x\n", weights[0], PL_WEIGHT_MAGIC);
        return -1;
    }
    if (weights[1] != PL_WEIGHT_VERSION) {
        fprintf(stderr, "bad PL weight version: %u expected %u\n", weights[1], PL_WEIGHT_VERSION);
        return -1;
    }
    if (weights[PL_WEIGHT_OFF_TOTAL_WORDS] != PL_WEIGHT_WORDS) {
        fprintf(stderr,
                "bad PL weight word count: %u expected %u\n",
                weights[PL_WEIGHT_OFF_TOTAL_WORDS],
                PL_WEIGHT_WORDS);
        return -1;
    }
    float scale = u32_to_float(weights[PL_WEIGHT_OFF_INPUT_SCALE_BITS]);
    if (!(scale > 0.0f) || !isfinite(scale)) {
        fprintf(stderr, "bad PL input scale: %f\n", scale);
        return -1;
    }
    *input_scale = scale;
    return 0;
}

static int hls_wait_done(volatile uint32_t *ctrl, int timeout_ms)
{
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    while (!g_stop) {
        uint32_t ap = reg_read(ctrl, AP_CTRL);
        if (ap & 0x02u) {
            return 0;
        }
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        long elapsed_ms = (now.tv_sec - start.tv_sec) * 1000L + (now.tv_nsec - start.tv_nsec) / 1000000L;
        if (elapsed_ms > timeout_ms) {
            fprintf(stderr, "HLS timeout, AP_CTRL=0x%08x\n", ap);
            return -1;
        }
        usleep(100);
    }
    return -1;
}

static int hls_load_weights(
    volatile uint32_t *ctrl,
    uint32_t input_phys,
    uint32_t weights_phys,
    uint32_t out_phys,
    int timeout_ms)
{
    reg_write_u64(ctrl, ADDR_INPUT_R, input_phys);
    reg_write_u64(ctrl, ADDR_WEIGHTS, weights_phys);
    reg_write(ctrl, ADDR_COMMAND, HLS_COMMAND_LOAD_WEIGHTS);
    reg_write_u64(ctrl, ADDR_FINAL_POSE, out_phys);
    reg_write(ctrl, AP_CTRL, 0x01u);
    if (hls_wait_done(ctrl, timeout_ms) != 0) {
        return -1;
    }
    reg_write(ctrl, ADDR_COMMAND, HLS_COMMAND_INFER);
    return 0;
}

static int hls_run(
    volatile uint32_t *ctrl,
    uint32_t input_phys,
    uint32_t weights_phys,
    uint32_t out_phys,
    int timeout_ms)
{
    reg_write_u64(ctrl, ADDR_INPUT_R, input_phys);
    reg_write_u64(ctrl, ADDR_WEIGHTS, weights_phys);
    reg_write(ctrl, ADDR_COMMAND, HLS_COMMAND_INFER);
    reg_write_u64(ctrl, ADDR_FINAL_POSE, out_phys);
    reg_write(ctrl, AP_CTRL, 0x01u);
    return hls_wait_done(ctrl, timeout_ms);
}

static int parse_hex_u32(const char *s, uint32_t *out)
{
    char *end = NULL;
    unsigned long v = strtoul(s, &end, 0);
    if (!end || *end != '\0' || v > 0xFFFFFFFFul) {
        return -1;
    }
    *out = (uint32_t)v;
    return 0;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [options]\n"
            "  --tty PATH              ESP serial device, default " DEFAULT_TTY "\n"
            "  --weights PATH          unified PL weight blob, default " DEFAULT_WEIGHT_BIN "\n"
            "  --ctrl-phys ADDR        full_pose_accel AXI-Lite base, default 0x%08x\n"
            "  --input-phys ADDR       ping-pong input0 DDR buffer, default 0x%08x\n"
            "  --input1-phys ADDR      ping-pong input1 DDR buffer, default 0x%08x\n"
            "  --weights-phys ADDR     unified weight DDR buffer, default 0x%08x\n"
            "  --load-weights-only     load DDR weights and issue PL update command, then exit\n"
            "  --out-phys ADDR         final_pose DDR buffer, default 0x%08x\n"
            "  --timeout-us N          ESP CSI timeout_us command, default %d\n"
            "  --slot-gap-us N         ESP CSI slot_gap_us command, default %d\n"
            "  --channel N             ESP Wi-Fi channel command, default %d\n"
            "  --second above|below|none  ESP second channel, default " DEFAULT_SECOND_CHANNEL "\n"
            "  --legacy-payload-pps    also send old payload/pps commands\n"
            "  --payload N             old ESP payload command value, default %d\n"
            "  --pps N                 old ESP packet rate command value, default %d\n"
            "  --esp-config            send ESP timeout/channel commands before CMD mode run\n"
            "  --no-esp-config         only send CMD mode wait/run, default\n"
            "  --max-infer N           stop after N inferences, default unlimited\n"
            "  --status-interval N     print periodic status every N seconds after inference, default 1; 0 disables\n"
            "  --verbose               print RX/parser/HLS progress logs\n"
            "  --print-every N         print one full pose every N inferences, default %d; 0 disables pose print\n"
            "  --results-only          print only pose vectors, suppress startup/status/verbose logs\n"
            "  --quiet                 disable per-inference pose print\n",
            prog,
            DEFAULT_CTRL_PHYS,
            DEFAULT_INPUT_PHYS,
            DEFAULT_INPUT1_PHYS,
            DEFAULT_WEIGHTS_PHYS,
            DEFAULT_OUT_PHYS,
            DEFAULT_TIMEOUT_US,
            DEFAULT_SLOT_GAP_US,
            DEFAULT_WIFI_CHANNEL,
            DEFAULT_ESP_PAYLOAD,
            DEFAULT_ESP_PPS,
            DEFAULT_PRINT_EVERY);
}

static int parse_args(int argc, char **argv, options_t *opt)
{
    *opt = (options_t){
        .tty_path = DEFAULT_TTY,
        .weight_path = DEFAULT_WEIGHT_BIN,
        .ctrl_phys = DEFAULT_CTRL_PHYS,
        .input_phys = DEFAULT_INPUT_PHYS,
        .input1_phys = DEFAULT_INPUT1_PHYS,
        .weights_phys = DEFAULT_WEIGHTS_PHYS,
        .out_phys = DEFAULT_OUT_PHYS,
        .timeout_us = DEFAULT_TIMEOUT_US,
        .slot_gap_us = DEFAULT_SLOT_GAP_US,
        .wifi_channel = DEFAULT_WIFI_CHANNEL,
        .second_channel = DEFAULT_SECOND_CHANNEL,
        .esp_payload = DEFAULT_ESP_PAYLOAD,
        .esp_pps = DEFAULT_ESP_PPS,
        .send_esp_config = false,
        .legacy_payload_pps = false,
        .max_infer = 0,
        .status_interval = 1,
        .print_every = DEFAULT_PRINT_EVERY,
        .verbose = false,
        .quiet = false,
        .results_only = false,
        .load_weights_only = false,
    };
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        const char *val = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            exit(0);
        } else if (strcmp(arg, "--tty") == 0 && val) {
            opt->tty_path = val;
            ++i;
        } else if (strcmp(arg, "--weights") == 0 && val) {
            opt->weight_path = val;
            ++i;
        } else if (strcmp(arg, "--ctrl-phys") == 0 && val) {
            if (parse_hex_u32(val, &opt->ctrl_phys) != 0) return -1;
            ++i;
        } else if (strcmp(arg, "--input-phys") == 0 && val) {
            if (parse_hex_u32(val, &opt->input_phys) != 0) return -1;
            ++i;
        } else if (strcmp(arg, "--input1-phys") == 0 && val) {
            if (parse_hex_u32(val, &opt->input1_phys) != 0) return -1;
            ++i;
        } else if ((strcmp(arg, "--weights-phys") == 0 || strcmp(arg, "--fc1-phys") == 0) && val) {
            if (parse_hex_u32(val, &opt->weights_phys) != 0) return -1;
            ++i;
        } else if (strcmp(arg, "--out-phys") == 0 && val) {
            if (parse_hex_u32(val, &opt->out_phys) != 0) return -1;
            ++i;
        } else if (strcmp(arg, "--timeout-us") == 0 && val) {
            opt->timeout_us = atoi(val);
            ++i;
        } else if (strcmp(arg, "--slot-gap-us") == 0 && val) {
            opt->slot_gap_us = atoi(val);
            ++i;
        } else if (strcmp(arg, "--channel") == 0 && val) {
            opt->wifi_channel = atoi(val);
            ++i;
        } else if (strcmp(arg, "--second") == 0 && val) {
            opt->second_channel = val;
            ++i;
        } else if (strcmp(arg, "--legacy-payload-pps") == 0) {
            opt->legacy_payload_pps = true;
        } else if (strcmp(arg, "--esp-config") == 0) {
            opt->send_esp_config = true;
        } else if (strcmp(arg, "--payload") == 0 && val) {
            opt->esp_payload = atoi(val);
            ++i;
        } else if (strcmp(arg, "--pps") == 0 && val) {
            opt->esp_pps = atoi(val);
            ++i;
        } else if (strcmp(arg, "--no-esp-config") == 0) {
            opt->send_esp_config = false;
        } else if (strcmp(arg, "--max-infer") == 0 && val) {
            opt->max_infer = atoi(val);
            ++i;
        } else if (strcmp(arg, "--status-interval") == 0 && val) {
            opt->status_interval = atoi(val);
            if (opt->status_interval < 0) opt->status_interval = 0;
            ++i;
        } else if (strcmp(arg, "--verbose") == 0) {
            opt->verbose = true;
        } else if (strcmp(arg, "--print-every") == 0 && val) {
            opt->print_every = atoi(val);
            if (opt->print_every < 0) opt->print_every = 0;
            ++i;
        } else if (strcmp(arg, "--results-only") == 0) {
            opt->results_only = true;
            opt->verbose = false;
            opt->status_interval = 0;
        } else if (strcmp(arg, "--load-weights-only") == 0) {
            opt->load_weights_only = true;
        } else if (strcmp(arg, "--quiet") == 0) {
            opt->quiet = true;
            opt->print_every = 0;
        } else {
            fprintf(stderr, "unknown or incomplete option: %s\n", arg);
            return -1;
        }
    }
    return 0;
}

static void rx_append(rx_buffer_t *rx, const uint8_t *data, size_t len)
{
    if (rx->len + len > rx->cap) {
        size_t keep = rx->len < 3u ? rx->len : 3u;
        if (keep) {
            memmove(rx->ptr, rx->ptr + rx->len - keep, keep);
        }
        rx->len = keep;
    }
    if (rx->len + len <= rx->cap) {
        memcpy(rx->ptr + rx->len, data, len);
        rx->len += len;
    }
}

static ssize_t find_magic(const uint8_t *buf, size_t len)
{
    for (size_t i = 0; i + 4u <= len; ++i) {
        if (rd32(buf + i) == SERIAL_MAGIC) {
            return (ssize_t)i;
        }
    }
    return -1;
}

static int parse_cycle_payload(const uint8_t *payload, size_t len, cycle_t *cycle)
{
    if (len < 32u) {
        return -1;
    }
    memset(cycle, 0, sizeof(*cycle));
    cycle->trigger_seq = rd32(payload + 4);
    (void)rd64(payload + 8);
    (void)rd64(payload + 16);
    cycle->active_nodes = payload[28];
    cycle->received_nodes = payload[29];
    size_t cursor = 32u;
    unsigned stored = 0;
    for (unsigned slot = 0; slot < cycle->active_nodes; ++slot) {
        if (cursor + 6u > len) {
            return -1;
        }
        int rx_index = payload[cursor + 0];
        int present = payload[cursor + 1];
        int rssi = (int8_t)payload[cursor + 2];
        uint16_t csi_len = rd16(payload + cursor + 4);
        cursor += 6u;
        if (cursor + csi_len > len) {
            return -1;
        }
        if (present && rx_index >= 0 && rx_index < (int)NODE_COUNT && stored < NODE_COUNT) {
            cycle->records[stored++] = (cycle_record_t){
                .rx_index = rx_index,
                .present = present,
                .rssi = rssi,
                .csi = (const int8_t *)(payload + cursor),
                .csi_len = csi_len,
            };
        }
        cursor += csi_len;
    }
    return (int)stored;
}

static int pop_cycle(rx_buffer_t *rx, cycle_t *cycle)
{
    const size_t header_size = 16u;
    while (rx->len >= header_size) {
        ssize_t pos = find_magic(rx->ptr, rx->len);
        if (pos < 0) {
            if (rx->len > 3u) {
                memmove(rx->ptr, rx->ptr + rx->len - 3u, 3u);
                rx->len = 3u;
            }
            return 0;
        }
        if (pos > 0) {
            memmove(rx->ptr, rx->ptr + pos, rx->len - (size_t)pos);
            rx->len -= (size_t)pos;
        }
        if (rx->len < header_size) {
            return 0;
        }
        uint8_t version = rx->ptr[4];
        uint8_t frame_type = rx->ptr[5];
        uint16_t payload_len = rd16(rx->ptr + 6);
        uint32_t checksum = rd32(rx->ptr + 12);
        if (version != 1u || payload_len > MAX_FRAME_PAYLOAD) {
            memmove(rx->ptr, rx->ptr + 1, rx->len - 1u);
            rx->len--;
            continue;
        }
        size_t total_len = header_size + payload_len;
        if (rx->len < total_len) {
            return 0;
        }
        uint32_t calc = checksum32(rx->ptr, 12u, rx->ptr + header_size, payload_len);
        if (calc != checksum) {
            memmove(rx->ptr, rx->ptr + 1, rx->len - 1u);
            rx->len--;
            continue;
        }
        if (frame_type == FRAME_CYCLE) {
            int records = parse_cycle_payload(rx->ptr + header_size, payload_len, cycle);
            memmove(rx->ptr, rx->ptr + total_len, rx->len - total_len);
            rx->len -= total_len;
            if (records > 0) {
                return 1;
            }
        } else {
            memmove(rx->ptr, rx->ptr + total_len, rx->len - total_len);
            rx->len -= total_len;
        }
    }
    return 0;
}

static void normalize(float *values, size_t len)
{
    if (len == 0) {
        return;
    }
    double sum = 0.0;
    for (size_t i = 0; i < len; ++i) {
        sum += values[i];
    }
    double mean = sum / (double)len;
    double var = 0.0;
    for (size_t i = 0; i < len; ++i) {
        double d = (double)values[i] - mean;
        var += d * d;
    }
    double std = sqrt(var / (double)len);
    double denom = std >= 1e-3 ? std : 1.0;
    for (size_t i = 0; i < len; ++i) {
        double v = ((double)values[i] - mean) / denom;
        if (v > 4.0) v = 4.0;
        if (v < -4.0) v = -4.0;
        values[i] = (float)v;
    }
}

static int parse_csi_feature(const int8_t *csi, uint16_t csi_len, float out[INPUT_H])
{
    size_t pair_count = csi_len / 2u;
    if (pair_count < INPUT_H) {
        return -1;
    }
    if (pair_count > MAX_CSI_PAIRS) {
        pair_count = MAX_CSI_PAIRS;
    }

    float values[MAX_CSI_PAIRS];
    for (size_t i = 0; i < pair_count; ++i) {
        int iv = csi[i * 2u + 0u];
        int qv = csi[i * 2u + 1u];
        values[i] = log1pf((float)(iv * iv + qv * qv));
    }
    normalize(values, pair_count);

    /*
     * Match ML/src/prepare.py:
     * _parse_iq_pairs(..., subcarrier_remap="esp32_htltf_ht40_above_nonstbc")
     * keeps the last 128 HT-LTF pairs, whose raw indices are [0..63, -64..-1],
     * and remaps them onto the signed axis [-64..63].
     */
    const float *raw_chunk = values + pair_count - INPUT_H;
    for (size_t i = 0; i < 64u; ++i) {
        out[64u + i] = raw_chunk[i];
        out[i] = raw_chunk[64u + i];
    }
    normalize(out, INPUT_H);
    return 0;
}

static int quantize_int8(float v)
{
    int q = (int)lrintf(v / g_input_scale);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    return q;
}

static void encode_hls_window(feature_state_t *state, int8_t *hls_input)
{
    float filled_base[INPUT_W][NODE_COUNT][INPUT_H];
    float last_valid[NODE_COUNT][INPUT_H];
    unsigned gap[NODE_COUNT];
    bool has_last_valid[NODE_COUNT] = {0};
    unsigned present_count = 0;
    unsigned drop_count = 0;
    unsigned max_gap = 0;

    memset(filled_base, 0, sizeof(filled_base));
    for (unsigned node = 0; node < NODE_COUNT; ++node) {
        has_last_valid[node] = state->prime_has_last_valid[node];
        gap[node] = state->prime_gap[node];
        for (unsigned sc = 0; sc < INPUT_H; ++sc) {
            last_valid[node][sc] = state->prime_last_valid[node][sc];
        }
    }

    for (unsigned w = 0; w < INPUT_W; ++w) {
        for (unsigned node = 0; node < NODE_COUNT; ++node) {
            bool present = state->raw_mask[w][node] > 0.5f;
            if (present) {
                present_count++;
                for (unsigned sc = 0; sc < INPUT_H; ++sc) {
                    float value = state->raw_base[w][node][sc];
                    filled_base[w][node][sc] = value;
                    last_valid[node][sc] = value;
                }
                has_last_valid[node] = true;
                gap[node] = 0;
                continue;
            }

            drop_count++;
            if (has_last_valid[node] && gap[node] < MAX_FILL_GAP) {
                for (unsigned sc = 0; sc < INPUT_H; ++sc) {
                    filled_base[w][node][sc] = last_valid[node][sc];
                }
            }
            gap[node]++;
            if (gap[node] > max_gap) {
                max_gap = gap[node];
            }
        }
    }

    state->window_present_count = present_count;
    state->window_drop_count = drop_count;
    state->window_max_gap = max_gap;

    for (unsigned w = 0; w < INPUT_W; ++w) {
        for (unsigned node = 0; node < NODE_COUNT; ++node) {
            for (unsigned sc = 0; sc < INPUT_H; ++sc) {
                state->window[w][node][sc] = filled_base[w][node][sc];
                state->window[w][NODE_COUNT + node][sc] =
                    (w == 0u) ? 0.0f : (filled_base[w][node][sc] - filled_base[w - 1u][node][sc]);
                state->window[w][NODE_COUNT * 2u + node][sc] = state->raw_mask[w][node];
            }
        }
    }

    for (unsigned ch = 0; ch < INPUT_CHANNELS; ++ch) {
        for (unsigned h = 0; h < INPUT_H; ++h) {
            for (unsigned w = 0; w < INPUT_W; ++w) {
                hls_input[(ch * INPUT_H + h) * INPUT_W + w] = (int8_t)quantize_int8(state->window[w][ch][h]);
            }
        }
    }
}

static bool feature_push_cycle(feature_state_t *state, const cycle_t *cycle, int8_t *hls_input)
{
    float base[NODE_COUNT][INPUT_H] = {{0}};
    float mask[NODE_COUNT] = {0};

    for (unsigned i = 0; i < NODE_COUNT; ++i) {
        const cycle_record_t *record = &cycle->records[i];
        if (!record->present) {
            continue;
        }
        if (parse_csi_feature(record->csi, record->csi_len, base[record->rx_index]) == 0) {
            mask[record->rx_index] = 1.0f;
        }
    }

    if (state->count >= INPUT_W && state->stride_count == 0u) {
        for (unsigned node = 0; node < NODE_COUNT; ++node) {
            state->prime_has_last_valid[node] = state->stream_has_last_valid[node];
            state->prime_gap[node] = state->stream_gap[node];
            for (unsigned sc = 0; sc < INPUT_H; ++sc) {
                state->prime_last_valid[node][sc] = state->stream_last_valid[node][sc];
            }
        }
    }

    memmove(state->raw_base[0], state->raw_base[1], sizeof(state->raw_base[0]) * (INPUT_W - 1u));
    memmove(state->raw_mask[0], state->raw_mask[1], sizeof(state->raw_mask[0]) * (INPUT_W - 1u));
    unsigned t = INPUT_W - 1u;
    memset(state->raw_base[t], 0, sizeof(state->raw_base[t]));
    memset(state->raw_mask[t], 0, sizeof(state->raw_mask[t]));

    for (unsigned node = 0; node < NODE_COUNT; ++node) {
        for (unsigned sc = 0; sc < INPUT_H; ++sc) {
            float b = base[node][sc];
            state->raw_base[t][node][sc] = b;
        }
        state->raw_mask[t][node] = mask[node];
        if (mask[node] > 0.5f) {
            for (unsigned sc = 0; sc < INPUT_H; ++sc) {
                state->stream_last_valid[node][sc] = base[node][sc];
            }
            state->stream_has_last_valid[node] = true;
            state->stream_gap[node] = 0;
        } else {
            state->stream_gap[node]++;
        }
    }
    bool first_complete_window = false;
    if (state->count < INPUT_W) {
        state->count++;
        first_complete_window = state->count == INPUT_W;
    }
    if (state->count < INPUT_W) {
        return false;
    }

    if (!first_complete_window) {
        state->stride_count++;
        if (state->stride_count < WINDOW_STRIDE) {
            return false;
        }
    }
    state->stride_count = 0;

    encode_hls_window(state, hls_input);
    return true;
}

static void make_abs_timeout_ms(struct timespec *ts, long timeout_ms)
{
    clock_gettime(CLOCK_REALTIME, ts);
    ts->tv_sec += timeout_ms / 1000L;
    ts->tv_nsec += (timeout_ms % 1000L) * 1000000L;
    if (ts->tv_nsec >= 1000000000L) {
        ts->tv_sec += 1;
        ts->tv_nsec -= 1000000000L;
    }
}

static double elapsed_ms(const struct timespec *start, const struct timespec *end)
{
    return (double)(end->tv_sec - start->tv_sec) * 1000.0 +
           (double)(end->tv_nsec - start->tv_nsec) / 1000000.0;
}

static double rel_ms(const struct timespec *ts)
{
    return elapsed_ms(&g_time_origin, ts);
}

static int wait_for_free_slot(hls_input_queue_t *queue, bool *dropped_old)
{
    int slot_index = -1;
    if (dropped_old) {
        *dropped_old = false;
    }
    pthread_mutex_lock(&queue->mutex);
    while (!g_stop) {
        for (unsigned i = 0; i < INPUT_BUFFER_COUNT; ++i) {
            if (!queue->slots[i].ready && !queue->slots[i].in_use) {
                queue->slots[i].in_use = true;
                slot_index = (int)i;
                break;
            }
        }
        if (slot_index >= 0) {
            break;
        }
        uint64_t oldest_seq = UINT64_MAX;
        for (unsigned i = 0; i < INPUT_BUFFER_COUNT; ++i) {
            if (queue->slots[i].ready && !queue->slots[i].in_use && queue->slots[i].seq < oldest_seq) {
                oldest_seq = queue->slots[i].seq;
                slot_index = (int)i;
            }
        }
        if (slot_index >= 0) {
            queue->slots[slot_index].ready = false;
            queue->slots[slot_index].in_use = true;
            if (dropped_old) {
                *dropped_old = true;
            }
            break;
        }
        struct timespec ts;
        make_abs_timeout_ms(&ts, 100);
        pthread_cond_timedwait(&queue->can_produce, &queue->mutex, &ts);
    }
    pthread_mutex_unlock(&queue->mutex);
    return slot_index;
}

static void publish_ready_slot(hls_input_queue_t *queue,
                               int slot_index,
                               uint32_t trigger_seq,
                               const struct timespec *usb_rx_ts,
                               const struct timespec *data_start_ts,
                               const struct timespec *data_ready_ts,
                               bool dropped_old,
                               double esp_gap_ms,
                               unsigned esp_present_count,
                               unsigned esp_drop_count,
                               unsigned esp_max_gap,
                               uint8_t esp_active_nodes,
                               uint8_t esp_received_nodes)
{
    pthread_mutex_lock(&queue->mutex);
    queue->slots[slot_index].trigger_seq = trigger_seq;
    queue->slots[slot_index].usb_rx_ts = *usb_rx_ts;
    queue->slots[slot_index].data_start_ts = *data_start_ts;
    queue->slots[slot_index].data_ready_ts = *data_ready_ts;
    queue->slots[slot_index].dropped_old_before_publish = dropped_old;
    queue->slots[slot_index].esp_gap_ms = esp_gap_ms;
    queue->slots[slot_index].esp_present_count = esp_present_count;
    queue->slots[slot_index].esp_drop_count = esp_drop_count;
    queue->slots[slot_index].esp_max_gap = esp_max_gap;
    queue->slots[slot_index].esp_active_nodes = esp_active_nodes;
    queue->slots[slot_index].esp_received_nodes = esp_received_nodes;
    queue->slots[slot_index].seq = queue->next_seq++;
    queue->slots[slot_index].ready = true;
    queue->slots[slot_index].in_use = false;
    pthread_cond_signal(&queue->can_consume);
    pthread_mutex_unlock(&queue->mutex);
}

static void release_slot(hls_input_queue_t *queue, int slot_index)
{
    pthread_mutex_lock(&queue->mutex);
    queue->slots[slot_index].ready = false;
    queue->slots[slot_index].in_use = false;
    pthread_cond_signal(&queue->can_produce);
    pthread_mutex_unlock(&queue->mutex);
}

static int wait_for_ready_slot(hls_input_queue_t *queue)
{
    int slot_index = -1;
    pthread_mutex_lock(&queue->mutex);
    while (!g_stop) {
        uint64_t best_seq = UINT64_MAX;
        for (unsigned i = 0; i < INPUT_BUFFER_COUNT; ++i) {
            if (queue->slots[i].ready && !queue->slots[i].in_use && queue->slots[i].seq < best_seq) {
                best_seq = queue->slots[i].seq;
                slot_index = (int)i;
            }
        }
        if (slot_index >= 0) {
            queue->slots[slot_index].ready = false;
            queue->slots[slot_index].in_use = true;
            break;
        }
        struct timespec ts;
        make_abs_timeout_ms(&ts, 100);
        pthread_cond_timedwait(&queue->can_consume, &queue->mutex, &ts);
    }
    pthread_mutex_unlock(&queue->mutex);
    return slot_index;
}

static void stats_set_window(runtime_stats_t *stats, unsigned window_count)
{
    pthread_mutex_lock(&stats->mutex);
    stats->window_count = window_count;
    pthread_mutex_unlock(&stats->mutex);
}

static void stats_add_cycle(runtime_stats_t *stats, unsigned window_count)
{
    pthread_mutex_lock(&stats->mutex);
    stats->cycles++;
    stats->window_count = window_count;
    pthread_mutex_unlock(&stats->mutex);
}

static void stats_add_ready(runtime_stats_t *stats)
{
    pthread_mutex_lock(&stats->mutex);
    stats->ready_inputs++;
    pthread_mutex_unlock(&stats->mutex);
}

static void stats_set_inferences(runtime_stats_t *stats, int inferences)
{
    pthread_mutex_lock(&stats->mutex);
    stats->inferences = inferences;
    pthread_mutex_unlock(&stats->mutex);
}

static void stats_snapshot(runtime_stats_t *stats,
                           uint64_t *cycles,
                           uint64_t *ready_inputs,
                           int *inferences,
                           unsigned *window_count)
{
    pthread_mutex_lock(&stats->mutex);
    *cycles = stats->cycles;
    *ready_inputs = stats->ready_inputs;
    *inferences = stats->inferences;
    *window_count = stats->window_count;
    pthread_mutex_unlock(&stats->mutex);
}

static void request_stop(hls_input_queue_t *queue, shared_rx_t *shared_rx)
{
    g_stop = 1;
    if (queue) {
        pthread_mutex_lock(&queue->mutex);
        pthread_cond_broadcast(&queue->can_produce);
        pthread_cond_broadcast(&queue->can_consume);
        pthread_mutex_unlock(&queue->mutex);
    }
    if (shared_rx) {
        pthread_mutex_lock(&shared_rx->mutex);
        pthread_cond_broadcast(&shared_rx->has_data);
        pthread_mutex_unlock(&shared_rx->mutex);
    }
}

static void *usb_rx_thread(void *arg)
{
    usb_thread_args_t *args = (usb_thread_args_t *)arg;
    uint8_t chunk[READ_CHUNK_SIZE];

    while (!g_stop) {
        ssize_t n = read(args->tty_fd, chunk, sizeof(chunk));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            perror("read tty");
            g_stop = 1;
            break;
        }
        if (n == 0) {
            continue;
        }

        pthread_mutex_lock(&args->shared_rx->mutex);
        rx_append(&args->shared_rx->rx, chunk, (size_t)n);
        clock_gettime(CLOCK_MONOTONIC, &args->shared_rx->last_rx_ts);
        size_t rx_len = args->shared_rx->rx.len;
        pthread_cond_signal(&args->shared_rx->has_data);
        pthread_mutex_unlock(&args->shared_rx->mutex);
        if (g_verbose) {
            printf("rx bytes=%zd buffered=%zu\n", n, rx_len);
            fflush(stdout);
        }
    }

    pthread_mutex_lock(&args->shared_rx->mutex);
    pthread_cond_broadcast(&args->shared_rx->has_data);
    pthread_mutex_unlock(&args->shared_rx->mutex);
    return NULL;
}

static void *parser_thread(void *arg)
{
    parser_thread_args_t *args = (parser_thread_args_t *)arg;
    feature_state_t feature_state;
    int8_t *scratch_input = (int8_t *)malloc(INPUT_SIZE);
    if (!scratch_input) {
        perror("malloc scratch_input");
        request_stop(args->queue, args->shared_rx);
        return NULL;
    }
    memset(&feature_state, 0, sizeof(feature_state));
    for (unsigned node = 0; node < NODE_COUNT; ++node) {
        feature_state.stream_gap[node] = MAX_FILL_GAP;
        feature_state.prime_gap[node] = MAX_FILL_GAP;
    }
    stats_set_window(args->stats, feature_state.count);
    struct timespec previous_input_usb_rx_ts = {0, 0};
    bool have_previous_input = false;

    while (!g_stop) {
        cycle_t cycle;
        bool have_cycle = false;
        struct timespec usb_rx_ts;

        pthread_mutex_lock(&args->shared_rx->mutex);
        while (!g_stop) {
            if (pop_cycle(&args->shared_rx->rx, &cycle) == 1) {
                usb_rx_ts = args->shared_rx->last_rx_ts;
                have_cycle = true;
                break;
            }
            struct timespec ts;
            make_abs_timeout_ms(&ts, 100);
            pthread_cond_timedwait(&args->shared_rx->has_data, &args->shared_rx->mutex, &ts);
        }
        pthread_mutex_unlock(&args->shared_rx->mutex);

        if (g_stop || !have_cycle) {
            break;
        }

        struct timespec data_start_ts, data_ready_ts;
        clock_gettime(CLOCK_MONOTONIC, &data_start_ts);
        bool input_ready = feature_push_cycle(&feature_state, &cycle, scratch_input);
        clock_gettime(CLOCK_MONOTONIC, &data_ready_ts);
        stats_add_cycle(args->stats, feature_state.count);
        if (g_verbose) {
            printf("cycle trigger=%u active=0x%02x received=0x%02x window=%u ready=%s\n",
                   cycle.trigger_seq,
                   cycle.active_nodes,
                   cycle.received_nodes,
                   feature_state.count,
                   input_ready ? "yes" : "no");
            fflush(stdout);
        }

        if (input_ready) {
            bool dropped_old = false;
            int slot_index = wait_for_free_slot(args->queue, &dropped_old);
            if (slot_index < 0) {
                break;
            }
            input_slot_t *slot = &args->queue->slots[slot_index];
            memcpy(slot->ptr, scratch_input, INPUT_SIZE);
            msync(slot->ptr, INPUT_SIZE, MS_SYNC);
            double esp_gap_ms = have_previous_input ? elapsed_ms(&previous_input_usb_rx_ts, &usb_rx_ts) : -1.0;
            publish_ready_slot(args->queue,
                               slot_index,
                               cycle.trigger_seq,
                               &usb_rx_ts,
                               &data_start_ts,
                               &data_ready_ts,
                               dropped_old,
                               esp_gap_ms,
                               feature_state.window_present_count,
                               feature_state.window_drop_count,
                               feature_state.window_max_gap,
                               cycle.active_nodes,
                               cycle.received_nodes);
            previous_input_usb_rx_ts = usb_rx_ts;
            have_previous_input = true;
            stats_add_ready(args->stats);
            if (g_verbose) {
                printf("ready trigger=%u slot=%d seq=%llu dropped_old=%d "
                       "esp_gap_ms=%.3f esp_present=%u esp_drop_count=%u esp_max_gap=%u "
                       "esp_active=%u esp_received=%u "
                       "usb_ms=%.3f data_ready_ms=%.3f data_ms=%.3f\n",
                       cycle.trigger_seq,
                       slot_index,
                       (unsigned long long)slot->seq,
                       dropped_old ? 1 : 0,
                       esp_gap_ms,
                       feature_state.window_present_count,
                       feature_state.window_drop_count,
                       feature_state.window_max_gap,
                       (unsigned)cycle.active_nodes,
                       (unsigned)cycle.received_nodes,
                       rel_ms(&usb_rx_ts),
                       rel_ms(&data_ready_ts),
                       elapsed_ms(&data_start_ts, &data_ready_ts));
                fflush(stdout);
            }
        }
    }

    free(scratch_input);
    request_stop(args->queue, args->shared_rx);
    return NULL;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    clock_gettime(CLOCK_MONOTONIC, &g_time_origin);

    options_t opt;
    if (parse_args(argc, argv, &opt) != 0) {
        usage(argv[0]);
        return 1;
    }
    if (opt.results_only) {
        opt.verbose = false;
        opt.status_interval = 0;
    }
    g_verbose = opt.verbose;

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) {
        perror("open /dev/mem");
        return 1;
    }

    volatile uint32_t *ctrl = (volatile uint32_t *)map_phys(mem_fd, opt.ctrl_phys, CTRL_MAP_SIZE);
    int8_t *input0 = (int8_t *)map_phys(mem_fd, opt.input_phys, INPUT_SIZE);
    int8_t *input1 = (int8_t *)map_phys(mem_fd, opt.input1_phys, INPUT_SIZE);
    uint32_t *weights = (uint32_t *)map_phys(mem_fd, opt.weights_phys, PL_WEIGHT_BYTES);
    float *final_pose = (float *)map_phys(mem_fd, opt.out_phys, POSE_DIM * sizeof(float));
    if (ctrl == MAP_FAILED || input0 == MAP_FAILED || input1 == MAP_FAILED ||
        weights == MAP_FAILED || final_pose == MAP_FAILED) {
        perror("mmap /dev/mem");
        close(mem_fd);
        return 1;
    }

    memset(input0, 0, INPUT_SIZE);
    memset(input1, 0, INPUT_SIZE);
    if (load_file(opt.weight_path, weights, PL_WEIGHT_BYTES) != 0) {
        close(mem_fd);
        return 1;
    }
    if (validate_loaded_weights(weights, &g_input_scale) != 0) {
        close(mem_fd);
        return 1;
    }
    msync(input0, INPUT_SIZE, MS_SYNC);
    msync(input1, INPUT_SIZE, MS_SYNC);
    msync(weights, PL_WEIGHT_BYTES, MS_SYNC);

    if (hls_load_weights(ctrl, opt.input_phys, opt.weights_phys, opt.out_phys, HLS_TIMEOUT_MS) != 0) {
        close(mem_fd);
        return 1;
    }
    if (opt.load_weights_only) {
        if (!opt.results_only) {
            printf("loaded PL weights: %s phys=0x%08x bytes=%u input_scale=%.9g\n",
                   opt.weight_path,
                   opt.weights_phys,
                   PL_WEIGHT_BYTES,
                   g_input_scale);
        }
        close(mem_fd);
        return 0;
    }

    int tty_fd = open(opt.tty_path, O_RDWR | O_NOCTTY);
    if (tty_fd < 0) {
        perror(opt.tty_path);
        close(mem_fd);
        return 1;
    }
    if (setup_tty_raw(tty_fd) != 0) {
        close(tty_fd);
        close(mem_fd);
        return 1;
    }

    shared_rx_t shared_rx;
    memset(&shared_rx, 0, sizeof(shared_rx));
    shared_rx.last_rx_ts = g_time_origin;
    pthread_mutex_init(&shared_rx.mutex, NULL);
    pthread_cond_init(&shared_rx.has_data, NULL);
    shared_rx.rx.ptr = malloc(RX_BUFFER_SIZE);
    shared_rx.rx.len = 0;
    shared_rx.rx.cap = RX_BUFFER_SIZE;
    if (!shared_rx.rx.ptr) {
        perror("malloc rx");
        close(tty_fd);
        close(mem_fd);
        return 1;
    }

    hls_input_queue_t hls_queue;
    memset(&hls_queue, 0, sizeof(hls_queue));
    pthread_mutex_init(&hls_queue.mutex, NULL);
    pthread_cond_init(&hls_queue.can_produce, NULL);
    pthread_cond_init(&hls_queue.can_consume, NULL);
    hls_queue.slots[0] = (input_slot_t){.ptr = input0, .phys = opt.input_phys, .ready = false, .in_use = false, .trigger_seq = 0, .seq = 0};
    hls_queue.slots[1] = (input_slot_t){.ptr = input1, .phys = opt.input1_phys, .ready = false, .in_use = false, .trigger_seq = 0, .seq = 0};
    hls_queue.next_seq = 0;

    runtime_stats_t stats;
    memset(&stats, 0, sizeof(stats));
    pthread_mutex_init(&stats.mutex, NULL);

    if (!opt.results_only) {
        printf("ESP serial: %s\n", opt.tty_path);
        printf("full_pose_accel control: 0x%08x\n", opt.ctrl_phys);
        printf("DDR input0=0x%08x input1=0x%08x weights=0x%08x out=0x%08x\n",
               opt.input_phys, opt.input1_phys, opt.weights_phys, opt.out_phys);
        printf("PL weights loaded: %s bytes=%u input_scale=%.9g\n",
               opt.weight_path,
               PL_WEIGHT_BYTES,
               g_input_scale);
        printf("scheduling: USB RX thread + parser thread + HLS consumer, ping-pong input buffers=%u\n",
               INPUT_BUFFER_COUNT);
        printf("pose feedback: v6 fast CNN has no HLS previous-pose state register\n");
        printf("print_every=%d quiet=%s verbose=%s status_interval=%d\n",
               opt.print_every,
               opt.quiet ? "true" : "false",
               opt.verbose ? "true" : "false",
               opt.status_interval);
        printf("starting ESP stream with CMD lines: timeout_us=%d slot_gap_us=%d channel=%d %s config=%s\n",
               opt.timeout_us,
               opt.slot_gap_us,
               opt.wifi_channel,
               opt.second_channel,
               opt.send_esp_config ? "on" : "off");
        if (opt.legacy_payload_pps) {
            printf("also sending legacy CMD lines: payload=%d pps=%d\n",
                   opt.esp_payload,
                   opt.esp_pps);
        }
    }
    if (configure_esp_stream(tty_fd, &opt) != 0) {
        free(shared_rx.rx.ptr);
        close(tty_fd);
        close(mem_fd);
        return 1;
    }

    pthread_t rx_thread;
    pthread_t parse_thread;
    usb_thread_args_t rx_args = {.tty_fd = tty_fd, .shared_rx = &shared_rx};
    parser_thread_args_t parser_args = {.shared_rx = &shared_rx, .queue = &hls_queue, .stats = &stats};

    if (pthread_create(&rx_thread, NULL, usb_rx_thread, &rx_args) != 0) {
        perror("pthread_create usb_rx_thread");
        request_stop(&hls_queue, &shared_rx);
        send_esp_line(tty_fd, "mode wait");
        free(shared_rx.rx.ptr);
        close(tty_fd);
        close(mem_fd);
        return 1;
    }
    if (pthread_create(&parse_thread, NULL, parser_thread, &parser_args) != 0) {
        perror("pthread_create parser_thread");
        request_stop(&hls_queue, &shared_rx);
        pthread_join(rx_thread, NULL);
        send_esp_line(tty_fd, "mode wait");
        free(shared_rx.rx.ptr);
        close(tty_fd);
        close(mem_fd);
        return 1;
    }
    if (!opt.results_only) {
        printf("threads started; waiting for CSI cycles until window=%u before first HLS run, stride=%u fill=forward_fill max_gap=%u\n",
               INPUT_W,
               WINDOW_STRIDE,
               MAX_FILL_GAP);
    }

    int hls_count = 0;
    int infer_count = 0;
    time_t last_status = time(NULL);
    struct timespec previous_pl_out_ts = {0, 0};
    bool have_previous_pl_out = false;

    while (!g_stop) {
        int slot_index = wait_for_ready_slot(&hls_queue);
        if (slot_index < 0) {
            break;
        }

        input_slot_t *slot = &hls_queue.slots[slot_index];
        if (g_verbose) {
            printf("hls_start trigger=%u hls=%d slot=%d\n",
                   slot->trigger_seq,
                   hls_count + 1,
                   slot_index);
            fflush(stdout);
        }

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        slot->pl_in_ts = t0;
        if (hls_run(ctrl, slot->phys, opt.weights_phys, opt.out_phys, HLS_TIMEOUT_MS) != 0) {
            release_slot(&hls_queue, slot_index);
            request_stop(&hls_queue, &shared_rx);
            break;
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        slot->pl_out_ts = t1;

        msync(final_pose, POSE_DIM * sizeof(float), MS_SYNC);

        hls_count++;

        double latency_ms = elapsed_ms(&t0, &t1);
        double data_ms = elapsed_ms(&slot->data_start_ts, &slot->data_ready_ts);
        double usb_to_data_ms = elapsed_ms(&slot->usb_rx_ts, &slot->data_ready_ts);
        double queue_ms = elapsed_ms(&slot->data_ready_ts, &slot->pl_in_ts);
        double total_ms = elapsed_ms(&slot->usb_rx_ts, &slot->pl_out_ts);
        double pl_gap_ms = have_previous_pl_out ? elapsed_ms(&previous_pl_out_ts, &slot->pl_out_ts) : -1.0;
        if (g_verbose) {
            printf("hls_done trigger=%u hls=%d slot=%d hls_ms=%.3f pl_out_ms=%.3f pose0=%.6f\n",
                   slot->trigger_seq,
                   hls_count,
                   slot_index,
                   latency_ms,
                   rel_ms(&slot->pl_out_ts),
                   final_pose[0]);
            fflush(stdout);
        }

        infer_count++;
        stats_set_inferences(&stats, infer_count);

        if (!opt.quiet && opt.print_every > 0 && (infer_count % opt.print_every) == 0) {
            if (opt.results_only) {
                for (unsigned i = 0; i < POSE_DIM; ++i) {
                    printf("%s%.6f", i == 0 ? "" : ",", final_pose[i]);
                }
                printf("\n");
            } else {
                printf("pose trigger=%u infer=%d hls=%d slot=%d dropped_old=%d "
                       "esp_gap_ms=%.3f esp_present=%u esp_drop_count=%u esp_max_gap=%u "
                       "esp_active=%u esp_received=%u "
                       "usb_ms=%.3f data_ready_ms=%.3f data_ms=%.3f usb_to_data_ms=%.3f "
                       "queue_ms=%.3f pl_in_ms=%.3f pl_out_ms=%.3f hls_ms=%.3f total_ms=%.3f gap_ms=%.3f",
                       slot->trigger_seq,
                       infer_count,
                       hls_count,
                       slot_index,
                       slot->dropped_old_before_publish ? 1 : 0,
                       slot->esp_gap_ms,
                       slot->esp_present_count,
                       slot->esp_drop_count,
                       slot->esp_max_gap,
                       (unsigned)slot->esp_active_nodes,
                       (unsigned)slot->esp_received_nodes,
                       rel_ms(&slot->usb_rx_ts),
                       rel_ms(&slot->data_ready_ts),
                       data_ms,
                       usb_to_data_ms,
                       queue_ms,
                       rel_ms(&slot->pl_in_ts),
                       rel_ms(&slot->pl_out_ts),
                       latency_ms,
                       total_ms,
                       pl_gap_ms);
                for (unsigned i = 0; i < POSE_DIM; ++i) {
                    printf("%s%.6f", i == 0 ? " [" : ",", final_pose[i]);
                }
                printf("]\n");
            }
            fflush(stdout);
        }
        previous_pl_out_ts = slot->pl_out_ts;
        have_previous_pl_out = true;

        release_slot(&hls_queue, slot_index);

        if (opt.max_infer > 0 && infer_count >= opt.max_infer) {
            request_stop(&hls_queue, &shared_rx);
            break;
        }

        time_t now = time(NULL);
        if (opt.status_interval > 0 && now - last_status >= opt.status_interval) {
            uint64_t cycles = 0;
            uint64_t ready_inputs = 0;
            int stats_inferences = 0;
            unsigned window_count = 0;
            stats_snapshot(&stats, &cycles, &ready_inputs, &stats_inferences, &window_count);

            size_t rx_len = 0;
            pthread_mutex_lock(&shared_rx.mutex);
            rx_len = shared_rx.rx.len;
            pthread_mutex_unlock(&shared_rx.mutex);

            printf("status cycles=%llu ready=%llu inferences=%d window=%u rxbuf=%zu AP_CTRL=0x%08x\n",
                   (unsigned long long)cycles,
                   (unsigned long long)ready_inputs,
                   stats_inferences,
                   window_count,
                   rx_len,
                   reg_read(ctrl, AP_CTRL));
            fflush(stdout);
            last_status = now;
        }
    }

    if (!opt.results_only) {
        printf("stopping ESP stream...\n");
    }
    request_stop(&hls_queue, &shared_rx);
    send_esp_line(tty_fd, "mode wait");
    pthread_join(rx_thread, NULL);
    pthread_join(parse_thread, NULL);

    free(shared_rx.rx.ptr);
    close(tty_fd);
    close(mem_fd);
    return 0;
}
