#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NODE_COUNT 3
#define INPUT_C 9
#define INPUT_H 128
#define INPUT_W 10
#define INPUT_SIZE (INPUT_C * INPUT_H * INPUT_W)
#define POSE_DIM 24
#define C1 16
#define C2 32
#define POOL_H 8
#define POOL_W 4
#define RECEIVER_FLAT_DIM (C2 * POOL_H * POOL_W)
#define FLAT_DIM (NODE_COUNT * RECEIVER_FLAT_DIM)
#define HIDDEN_DIM 128
#define INT8_QMIN -127
#define INT8_QMAX 127

typedef enum { MODE_FLOAT32 = 0, MODE_INT8 = 1 } infer_mode_t;

typedef struct {
    int out;
    int in;
    int kh;
    int kw;
    float *w;
    float *b;
    int8_t *qw;
    int32_t *qb;
    int32_t *requant_mult;
    int32_t *requant_shift;
} conv_t;

typedef struct {
    int out;
    int in;
    float *w;
    float *b;
    int8_t *qw;
    int32_t *qb;
    int32_t *requant_mult;
    int32_t *requant_shift;
} linear_t;

typedef struct {
    int node_count;
    float input_scale;
    float output_scale;
    int32_t pool_requant_mult;
    int32_t pool_requant_shift;
    conv_t conv1;
    conv_t conv2;
    linear_t fc1;
    linear_t fc2;
    linear_t fc3;
    int8_t lut_encoder_gelu1[256];
    int8_t lut_encoder_gelu2[256];
    int8_t lut_head_gelu1[256];
    int8_t lut_head_gelu2[256];
} model_t;

static double elapsed_ms(struct timespec a, struct timespec b)
{
    return (double)(b.tv_sec - a.tv_sec) * 1000.0 + (double)(b.tv_nsec - a.tv_nsec) / 1000000.0;
}

static int read_exact(FILE *fp, void *dst, size_t bytes)
{
    return fread(dst, 1u, bytes, fp) == bytes ? 0 : -1;
}

static int write_exact(FILE *fp, const void *src, size_t bytes)
{
    return fwrite(src, 1u, bytes, fp) == bytes ? 0 : -1;
}

static float *alloc_floats(size_t n)
{
    float *p = (float *)calloc(n, sizeof(float));
    if (!p) {
        fprintf(stderr, "calloc float failed: %zu\n", n);
        exit(1);
    }
    return p;
}

static int8_t *alloc_i8(size_t n)
{
    int8_t *p = (int8_t *)calloc(n, sizeof(int8_t));
    if (!p) {
        fprintf(stderr, "calloc int8 failed: %zu\n", n);
        exit(1);
    }
    return p;
}

static int32_t *alloc_i32(size_t n)
{
    int32_t *p = (int32_t *)calloc(n, sizeof(int32_t));
    if (!p) {
        fprintf(stderr, "calloc int32 failed: %zu\n", n);
        exit(1);
    }
    return p;
}

static void init_conv(conv_t *l, int out, int in, int kh, int kw)
{
    l->out = out;
    l->in = in;
    l->kh = kh;
    l->kw = kw;
    l->w = alloc_floats((size_t)out * in * kh * kw);
    l->b = alloc_floats((size_t)out);
    l->qw = alloc_i8((size_t)out * in * kh * kw);
    l->qb = alloc_i32((size_t)out);
    l->requant_mult = alloc_i32((size_t)out);
    l->requant_shift = alloc_i32((size_t)out);
}

static void init_linear(linear_t *l, int out, int in)
{
    l->out = out;
    l->in = in;
    l->w = alloc_floats((size_t)out * in);
    l->b = alloc_floats((size_t)out);
    l->qw = alloc_i8((size_t)out * in);
    l->qb = alloc_i32((size_t)out);
    l->requant_mult = alloc_i32((size_t)out);
    l->requant_shift = alloc_i32((size_t)out);
}

static int8_t sat_i8(long long v)
{
    if (v > INT8_QMAX) return INT8_QMAX;
    if (v < INT8_QMIN) return INT8_QMIN;
    return (int8_t)v;
}

static long long round_shift_signed(long long value, int shift)
{
    if (shift <= 0) {
        return value << (-shift);
    }
    long long offset = 1LL << (shift - 1);
    return value >= 0 ? (value + offset) >> shift : (value - offset) >> shift;
}

static long long floor_div_signed(long long numerator, int denominator)
{
    long long quotient = numerator / denominator;
    long long remainder = numerator % denominator;
    if (remainder != 0 && numerator < 0) {
        --quotient;
    }
    return quotient;
}

static int8_t requant_acc_int(int32_t acc, int32_t multiplier, int32_t shift)
{
    long long product = (long long)acc * (long long)multiplier;
    return sat_i8(round_shift_signed(product, shift));
}

static int8_t quantize_float(float value, float scale)
{
    int q = (int)lrintf(value / scale);
    if (q > INT8_QMAX) q = INT8_QMAX;
    if (q < INT8_QMIN) q = INT8_QMIN;
    return (int8_t)q;
}

static float gelu(float x)
{
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

static void conv2d_float(const conv_t *l, const float *x, int h, int w, float *y)
{
    memset(y, 0, (size_t)l->out * h * w * sizeof(float));
    int ph = l->kh / 2;
    int pw = l->kw / 2;
    for (int oc = 0; oc < l->out; ++oc) {
        for (int oh = 0; oh < h; ++oh) {
            for (int ow = 0; ow < w; ++ow) {
                float acc = l->b[oc];
                for (int ic = 0; ic < l->in; ++ic) {
                    for (int kh = 0; kh < l->kh; ++kh) {
                        int ih = oh + kh - ph;
                        if (ih < 0 || ih >= h) continue;
                        for (int kw = 0; kw < l->kw; ++kw) {
                            int iw = ow + kw - pw;
                            if (iw < 0 || iw >= w) continue;
                            size_t wi = (((size_t)oc * l->in + ic) * l->kh + kh) * l->kw + kw;
                            size_t xi = ((size_t)ic * h + ih) * w + iw;
                            acc += l->w[wi] * x[xi];
                        }
                    }
                }
                y[((size_t)oc * h + oh) * w + ow] = acc;
            }
        }
    }
}

static void linear_float(const linear_t *l, const float *x, float *y)
{
    for (int o = 0; o < l->out; ++o) {
        float acc = l->b[o];
        const float *w = l->w + (size_t)o * l->in;
        for (int i = 0; i < l->in; ++i) {
            acc += w[i] * x[i];
        }
        y[o] = acc;
    }
}

static void adaptive_avg_pool_8x4_float(const float *x, int c, int h, int w, float *y)
{
    for (int ch = 0; ch < c; ++ch) {
        for (int oh = 0; oh < POOL_H; ++oh) {
            int hs = (int)floor((double)(oh * h) / POOL_H);
            int he = (int)ceil((double)((oh + 1) * h) / POOL_H);
            for (int ow = 0; ow < POOL_W; ++ow) {
                int ws = (int)floor((double)(ow * w) / POOL_W);
                int we = (int)ceil((double)((ow + 1) * w) / POOL_W);
                double sum = 0.0;
                int count = 0;
                for (int ih = hs; ih < he; ++ih) {
                    for (int iw = ws; iw < we; ++iw) {
                        sum += x[((size_t)ch * h + ih) * w + iw];
                        count++;
                    }
                }
                y[((size_t)ch * POOL_H + oh) * POOL_W + ow] = (float)(sum / (double)count);
            }
        }
    }
}

static void slice_node_float(const float *input, int node, float *out)
{
    int channels[3] = {node, NODE_COUNT + node, NODE_COUNT * 2 + node};
    for (int c = 0; c < 3; ++c) {
        memcpy(
            out + (size_t)c * INPUT_H * INPUT_W,
            input + (size_t)channels[c] * INPUT_H * INPUT_W,
            INPUT_H * INPUT_W * sizeof(float));
    }
}

static void infer_float(const model_t *m, const float *input, float *out_pose)
{
    float *node_in = alloc_floats(3 * INPUT_H * INPUT_W);
    float *conv1 = alloc_floats(C1 * INPUT_H * INPUT_W);
    float *conv2 = alloc_floats(C2 * INPUT_H * INPUT_W);
    float *pool = alloc_floats(C2 * POOL_H * POOL_W);
    float *flat = alloc_floats(FLAT_DIM);
    float h1[HIDDEN_DIM], h2[HIDDEN_DIM];
    size_t cursor = 0;

    for (int node = 0; node < NODE_COUNT; ++node) {
        slice_node_float(input, node, node_in);
        conv2d_float(&m->conv1, node_in, INPUT_H, INPUT_W, conv1);
        for (int i = 0; i < C1 * INPUT_H * INPUT_W; ++i) conv1[i] = gelu(conv1[i]);
        conv2d_float(&m->conv2, conv1, INPUT_H, INPUT_W, conv2);
        for (int i = 0; i < C2 * INPUT_H * INPUT_W; ++i) conv2[i] = gelu(conv2[i]);
        adaptive_avg_pool_8x4_float(conv2, C2, INPUT_H, INPUT_W, pool);
        memcpy(flat + cursor, pool, C2 * POOL_H * POOL_W * sizeof(float));
        cursor += C2 * POOL_H * POOL_W;
    }

    linear_float(&m->fc1, flat, h1);
    for (int i = 0; i < HIDDEN_DIM; ++i) h1[i] = gelu(h1[i]);
    linear_float(&m->fc2, h1, h2);
    for (int i = 0; i < HIDDEN_DIM; ++i) h2[i] = gelu(h2[i]);
    linear_float(&m->fc3, h2, out_pose);

    free(node_in);
    free(conv1);
    free(conv2);
    free(pool);
    free(flat);
}

static void conv1_receiver_i8(const model_t *m, const int8_t *input, int node, int8_t *output)
{
    const conv_t *l = &m->conv1;
    for (int oc = 0; oc < C1; ++oc) {
        for (int h = 0; h < INPUT_H; ++h) {
            for (int w = 0; w < INPUT_W; ++w) {
                int32_t acc = l->qb[oc];
                for (int ch = 0; ch < 3; ++ch) {
                    for (int kh = 0; kh < 5; ++kh) {
                        for (int kw = 0; kw < 3; ++kw) {
                            int ih = h + kh - 2;
                            int iw = w + kw - 1;
                            if (ih >= 0 && ih < INPUT_H && iw >= 0 && iw < INPUT_W) {
                                size_t xi = ((size_t)(node + ch * NODE_COUNT) * INPUT_H + ih) * INPUT_W + iw;
                                size_t wi = (((size_t)oc * 3 + ch) * 5 + kh) * 3 + kw;
                                acc += (int32_t)input[xi] * (int32_t)l->qw[wi];
                            }
                        }
                    }
                }
                output[((size_t)oc * INPUT_H + h) * INPUT_W + w] =
                    requant_acc_int(acc, l->requant_mult[oc], l->requant_shift[oc]);
            }
        }
    }
}

static void conv2_receiver_i8(const model_t *m, const int8_t *input, int8_t *output)
{
    const conv_t *l = &m->conv2;
    for (int oc = 0; oc < C2; ++oc) {
        for (int h = 0; h < INPUT_H; ++h) {
            for (int w = 0; w < INPUT_W; ++w) {
                int32_t acc = l->qb[oc];
                for (int ic = 0; ic < C1; ++ic) {
                    for (int kh = 0; kh < 3; ++kh) {
                        for (int kw = 0; kw < 3; ++kw) {
                            int ih = h + kh - 1;
                            int iw = w + kw - 1;
                            if (ih >= 0 && ih < INPUT_H && iw >= 0 && iw < INPUT_W) {
                                size_t xi = ((size_t)ic * INPUT_H + ih) * INPUT_W + iw;
                                size_t wi = (((size_t)oc * C1 + ic) * 3 + kh) * 3 + kw;
                                acc += (int32_t)input[xi] * (int32_t)l->qw[wi];
                            }
                        }
                    }
                }
                output[((size_t)oc * INPUT_H + h) * INPUT_W + w] =
                    requant_acc_int(acc, l->requant_mult[oc], l->requant_shift[oc]);
            }
        }
    }
}

static void apply_lut_i8(int8_t *data, size_t n, const int8_t lut[256])
{
    for (size_t i = 0; i < n; ++i) {
        data[i] = lut[(int)data[i] + 128];
    }
}

static int div_floor_bin_start(int out_index, int in_size, int out_size)
{
    return (out_index * in_size) / out_size;
}

static int div_ceil_bin_end(int out_index_plus_1, int in_size, int out_size)
{
    return (out_index_plus_1 * in_size + out_size - 1) / out_size;
}

static void adaptive_pool_flatten_i8(const model_t *m, const int8_t *input, int8_t *flatten_segment)
{
    for (int c = 0; c < C2; ++c) {
        for (int oh = 0; oh < POOL_H; ++oh) {
            for (int ow = 0; ow < POOL_W; ++ow) {
                int hs = div_floor_bin_start(oh, INPUT_H, POOL_H);
                int he = div_ceil_bin_end(oh + 1, INPUT_H, POOL_H);
                int ws = div_floor_bin_start(ow, INPUT_W, POOL_W);
                int we = div_ceil_bin_end(ow + 1, INPUT_W, POOL_W);
                int sum = 0;
                for (int ih = hs; ih < he; ++ih) {
                    for (int iw = ws; iw < we; ++iw) {
                        sum += (int)input[((size_t)c * INPUT_H + ih) * INPUT_W + iw];
                    }
                }
                int divisor = (he - hs) * (we - ws);
                long long scaled = round_shift_signed(
                    (long long)sum * (long long)m->pool_requant_mult,
                    m->pool_requant_shift);
                long long rounded = scaled >= 0 ? scaled + divisor / 2 : scaled - divisor / 2;
                long long pooled = floor_div_signed(rounded, divisor);
                flatten_segment[(c * POOL_H + oh) * POOL_W + ow] = sat_i8(pooled);
            }
        }
    }
}

static void linear_i8(const linear_t *l, const int8_t *x, int8_t *y)
{
    for (int o = 0; o < l->out; ++o) {
        int32_t acc = l->qb[o];
        const int8_t *w = l->qw + (size_t)o * l->in;
        for (int i = 0; i < l->in; ++i) {
            acc += (int32_t)x[i] * (int32_t)w[i];
        }
        y[o] = requant_acc_int(acc, l->requant_mult[o], l->requant_shift[o]);
    }
}

static void infer_int8(const model_t *m, const float *input, float *out_pose)
{
    int8_t *input_q = alloc_i8(INPUT_SIZE);
    int8_t *conv1 = alloc_i8(C1 * INPUT_H * INPUT_W);
    int8_t *conv2 = alloc_i8(C2 * INPUT_H * INPUT_W);
    int8_t *flat = alloc_i8(FLAT_DIM);
    int8_t h1[HIDDEN_DIM], h2[HIDDEN_DIM], pose_q[POSE_DIM];

    for (int i = 0; i < INPUT_SIZE; ++i) {
        input_q[i] = quantize_float(input[i], m->input_scale);
    }

    for (int node = 0; node < NODE_COUNT; ++node) {
        conv1_receiver_i8(m, input_q, node, conv1);
        apply_lut_i8(conv1, C1 * INPUT_H * INPUT_W, m->lut_encoder_gelu1);
        conv2_receiver_i8(m, conv1, conv2);
        apply_lut_i8(conv2, C2 * INPUT_H * INPUT_W, m->lut_encoder_gelu2);
        adaptive_pool_flatten_i8(m, conv2, flat + (size_t)node * RECEIVER_FLAT_DIM);
    }

    linear_i8(&m->fc1, flat, h1);
    apply_lut_i8(h1, HIDDEN_DIM, m->lut_head_gelu1);
    linear_i8(&m->fc2, h1, h2);
    apply_lut_i8(h2, HIDDEN_DIM, m->lut_head_gelu2);
    linear_i8(&m->fc3, h2, pose_q);
    for (int i = 0; i < POSE_DIM; ++i) {
        out_pose[i] = (float)pose_q[i] * m->output_scale;
    }

    free(input_q);
    free(conv1);
    free(conv2);
    free(flat);
}

static int load_model(const char *path, model_t *m)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open model failed: %s: %s\n", path, strerror(errno));
        return -1;
    }
    char magic[8];
    uint32_t node_count, reserved;
    if (read_exact(fp, magic, 8) || memcmp(magic, "PSFST2\0\0", 8) != 0) {
        fclose(fp);
        return -1;
    }
    if (read_exact(fp, &node_count, 4) ||
        read_exact(fp, &reserved, 4) ||
        read_exact(fp, &m->input_scale, 4) ||
        read_exact(fp, &m->output_scale, 4) ||
        read_exact(fp, &m->pool_requant_mult, 4) ||
        read_exact(fp, &m->pool_requant_shift, 4)) {
        fclose(fp);
        return -1;
    }
    m->node_count = (int)node_count;
    if (m->node_count != NODE_COUNT) {
        fprintf(stderr, "unsupported node_count=%d\n", m->node_count);
        fclose(fp);
        return -1;
    }

    init_conv(&m->conv1, C1, 3, 5, 3);
    init_conv(&m->conv2, C2, C1, 3, 3);
    init_linear(&m->fc1, HIDDEN_DIM, FLAT_DIM);
    init_linear(&m->fc2, HIDDEN_DIM, HIDDEN_DIM);
    init_linear(&m->fc3, POSE_DIM, HIDDEN_DIM);

#define RF(ptr, n) do { if (read_exact(fp, (ptr), (size_t)(n) * sizeof(float))) { fclose(fp); return -1; } } while (0)
#define RI8(ptr, n) do { if (read_exact(fp, (ptr), (size_t)(n) * sizeof(int8_t))) { fclose(fp); return -1; } } while (0)
#define RI32(ptr, n) do { if (read_exact(fp, (ptr), (size_t)(n) * sizeof(int32_t))) { fclose(fp); return -1; } } while (0)
#define READ_CONV_FLOAT(l) RF((l).w, (l).out * (l).in * (l).kh * (l).kw); RF((l).b, (l).out)
#define READ_LINEAR_FLOAT(l) RF((l).w, (l).out * (l).in); RF((l).b, (l).out)
#define READ_CONV_Q(l) RI8((l).qw, (l).out * (l).in * (l).kh * (l).kw); RI32((l).qb, (l).out); RI32((l).requant_mult, (l).out); RI32((l).requant_shift, (l).out)
#define READ_LINEAR_Q(l) RI8((l).qw, (l).out * (l).in); RI32((l).qb, (l).out); RI32((l).requant_mult, (l).out); RI32((l).requant_shift, (l).out)

    READ_CONV_FLOAT(m->conv1);
    READ_CONV_FLOAT(m->conv2);
    READ_LINEAR_FLOAT(m->fc1);
    READ_LINEAR_FLOAT(m->fc2);
    READ_LINEAR_FLOAT(m->fc3);
    READ_CONV_Q(m->conv1);
    READ_CONV_Q(m->conv2);
    READ_LINEAR_Q(m->fc1);
    READ_LINEAR_Q(m->fc2);
    READ_LINEAR_Q(m->fc3);
    RI8(m->lut_encoder_gelu1, 256);
    RI8(m->lut_encoder_gelu2, 256);
    RI8(m->lut_head_gelu1, 256);
    RI8(m->lut_head_gelu2, 256);

    fclose(fp);
    return 0;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s --mode float32|int8 --model ps_model.bin --input ps_input.bin --output poses.bin [--timing timing.bin]\n",
            prog);
}

int main(int argc, char **argv)
{
    const char *model_path = NULL, *input_path = NULL, *output_path = NULL, *timing_path = NULL;
    infer_mode_t mode = MODE_FLOAT32;
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i], *val = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (strcmp(arg, "--mode") == 0 && val) {
            if (strcmp(val, "float32") == 0) mode = MODE_FLOAT32;
            else if (strcmp(val, "int8") == 0) mode = MODE_INT8;
            else {
                usage(argv[0]);
                return 2;
            }
            ++i;
        } else if (strcmp(arg, "--model") == 0 && val) {
            model_path = val;
            ++i;
        } else if (strcmp(arg, "--input") == 0 && val) {
            input_path = val;
            ++i;
        } else if (strcmp(arg, "--output") == 0 && val) {
            output_path = val;
            ++i;
        } else if (strcmp(arg, "--timing") == 0 && val) {
            timing_path = val;
            ++i;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!model_path || !input_path || !output_path) {
        usage(argv[0]);
        return 2;
    }

    model_t model;
    memset(&model, 0, sizeof(model));
    if (load_model(model_path, &model) != 0) {
        fprintf(stderr, "failed to load model: %s\n", model_path);
        return 1;
    }

    FILE *in = fopen(input_path, "rb");
    FILE *out = fopen(output_path, "wb");
    FILE *timing = timing_path ? fopen(timing_path, "wb") : NULL;
    if (!in || !out || (timing_path && !timing)) {
        fprintf(stderr, "open input/output failed\n");
        return 1;
    }
    char magic[8];
    uint32_t windows = 0, reserved = 0;
    if (read_exact(in, magic, 8) || memcmp(magic, "PSIN1\0\0\0", 8) != 0 ||
        read_exact(in, &windows, 4) || read_exact(in, &reserved, 4)) {
        fprintf(stderr, "bad input file\n");
        return 1;
    }

    float *input = alloc_floats(INPUT_SIZE);
    float pose[POSE_DIM];
    struct timespec total0, total1;
    clock_gettime(CLOCK_MONOTONIC, &total0);
    for (uint32_t w = 0; w < windows; ++w) {
        int32_t file_id;
        (void)file_id;
        if (read_exact(in, &file_id, 4) || read_exact(in, input, sizeof(float) * INPUT_SIZE)) {
            fprintf(stderr, "truncated input at window %u\n", w);
            return 1;
        }
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (mode == MODE_FLOAT32) {
            infer_float(&model, input, pose);
        } else {
            infer_int8(&model, input, pose);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (write_exact(out, pose, sizeof(pose))) return 1;
        if (timing) {
            float ms = (float)elapsed_ms(t0, t1);
            if (write_exact(timing, &ms, sizeof(ms))) return 1;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &total1);
    printf("ps_%s windows=%u elapsed_ms=%.3f avg_ms=%.6f\n",
           mode == MODE_FLOAT32 ? "float32" : "int8",
           windows,
           elapsed_ms(total0, total1),
           elapsed_ms(total0, total1) / (double)(windows ? windows : 1));

    free(input);
    fclose(in);
    fclose(out);
    if (timing) fclose(timing);
    return 0;
}
