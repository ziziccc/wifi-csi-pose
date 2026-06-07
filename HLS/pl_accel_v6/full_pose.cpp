#include "full_pose.h"

static signed char g_conv1_weight_int8[NODE_COUNT][CONV1_OUT][RECEIVER_CHANNELS][5][3];
static int g_conv1_bias_int32[NODE_COUNT][CONV1_OUT];
static int g_conv1_requant_mult[NODE_COUNT][CONV1_OUT];
static int g_conv1_requant_shift[NODE_COUNT][CONV1_OUT];

static signed char g_conv2_weight_int8[NODE_COUNT][CONV2_OUT][CONV1_OUT][3][3];
static int g_conv2_bias_int32[NODE_COUNT][CONV2_OUT];
static int g_conv2_requant_mult[NODE_COUNT][CONV2_OUT];
static int g_conv2_requant_shift[NODE_COUNT][CONV2_OUT];

static int g_fc1_bias_int32[HIDDEN_DIM];
static int g_fc1_requant_mult[HIDDEN_DIM];
static int g_fc1_requant_shift[HIDDEN_DIM];

static signed char g_fc2_weight_int8[HIDDEN_DIM][HIDDEN_DIM];
static int g_fc2_bias_int32[HIDDEN_DIM];
static int g_fc2_requant_mult[HIDDEN_DIM];
static int g_fc2_requant_shift[HIDDEN_DIM];

static signed char g_fc3_weight_int8[POSE_DIM][HIDDEN_DIM];
static int g_fc3_bias_int32[POSE_DIM];
static int g_fc3_requant_mult[POSE_DIM];
static int g_fc3_requant_shift[POSE_DIM];

static signed char g_lut_encoder_gelu1_int8[NODE_COUNT][256];
static signed char g_lut_encoder_gelu2_int8[NODE_COUNT][256];
static signed char g_lut_head_gelu1_int8[256];
static signed char g_lut_head_gelu2_int8[256];

static int g_pool_requant_mult[NODE_COUNT] = {0, 0, 0};
static int g_pool_requant_shift[NODE_COUNT] = {0, 0, 0};
static float g_head_fc3_out_scale = 1.0f;
static bool g_weights_loaded = false;

static qint8_t sat_int8(long long v) {
#pragma HLS INLINE
    if (v > 127) {
        return 127;
    }
    if (v < -127) {
        return -127;
    }
    return (qint8_t)v;
}

static long long round_shift_signed(long long value, int shift) {
#pragma HLS INLINE
    if (shift <= 0) {
        return value << (-shift);
    }
    const long long offset = 1LL << (shift - 1);
    if (value >= 0) {
        return (value + offset) >> shift;
    }
    return (value - offset) >> shift;
}

static int div48_nonnegative(int value) {
#pragma HLS INLINE
    int div16 = value >> 4;
    int reciprocal_product = (div16 << 7) + (div16 << 5) + (div16 << 3) + (div16 << 1) + div16;
    return reciprocal_product >> 9;
}

static int floor_div48_saturated(long long numerator) {
#pragma HLS INLINE
    if (numerator >= 6096) {
        return 127;
    }
    if (numerator <= -6049) {
        return -127;
    }
    if (numerator >= 0) {
        return div48_nonnegative((int)numerator);
    }
    return -div48_nonnegative((int)(-numerator + 47));
}

static qint8_t requant_acc_int(qint32_t acc, int multiplier, int shift) {
#pragma HLS INLINE
    long long product;
#pragma HLS BIND_OP variable=product op=mul impl=dsp latency=3
    product = (long long)acc * (long long)multiplier;
    return sat_int8(round_shift_signed(product, shift));
}

static qint32_t mul_i8_dsp(qint8_t a, qint8_t b) {
#pragma HLS INLINE
    qint32_t product;
#pragma HLS BIND_OP variable=product op=mul impl=dsp latency=2
    product = (qint32_t)a * (qint32_t)b;
    return product;
}

static qint8_t unpack_i8(packed_weight_t word, int byte_index) {
#pragma HLS INLINE
    ap_uint<8> bits = word.range(byte_index * 8 + 7, byte_index * 8);
    qint8_t value;
    value.range(7, 0) = bits;
    return value;
}

static int word_to_i32(packed_weight_t word) {
#pragma HLS INLINE
    ap_int<32> value;
    value.range(31, 0) = word.range(31, 0);
    return (int)value;
}

static float word_to_float(packed_weight_t word) {
#pragma HLS INLINE
    union {
        unsigned int u;
        float f;
    } value;
    value.u = (unsigned int)word;
    return value.f;
}

static signed char read_packed_i8(const packed_weight_t weights[WEIGHT_WORDS],
                                  int word_offset,
                                  int index) {
#pragma HLS INLINE
    return (signed char)unpack_i8(weights[word_offset + index / 4], index & 3);
}

static void load_i32_array(const packed_weight_t weights[WEIGHT_WORDS],
                           int word_offset,
                           int *dst,
                           int count) {
LOAD_I32_ARRAY:
    for (int i = 0; i < count; ++i) {
#pragma HLS PIPELINE II=1
        dst[i] = word_to_i32(weights[word_offset + i]);
    }
}

static void load_i8_array(const packed_weight_t weights[WEIGHT_WORDS],
                          int word_offset,
                          signed char *dst,
                          int count) {
LOAD_I8_ARRAY:
    for (int i = 0; i < count; ++i) {
#pragma HLS PIPELINE II=1
        dst[i] = read_packed_i8(weights, word_offset, i);
    }
}

static void load_encoder_bank(const packed_weight_t weights[WEIGHT_WORDS], int bank) {
#pragma HLS INLINE off
    g_pool_requant_mult[bank] = word_to_i32(weights[WEIGHT_OFF_POOL_MULT]);
    g_pool_requant_shift[bank] = word_to_i32(weights[WEIGHT_OFF_POOL_SHIFT]);

ENC_LOAD_CONV1_WEIGHT_OC:
    for (int oc = 0; oc < CONV1_OUT; ++oc) {
    ENC_LOAD_CONV1_WEIGHT_CH:
        for (int ch = 0; ch < RECEIVER_CHANNELS; ++ch) {
        ENC_LOAD_CONV1_WEIGHT_KH:
            for (int kh = 0; kh < 5; ++kh) {
            ENC_LOAD_CONV1_WEIGHT_KW:
                for (int kw = 0; kw < 3; ++kw) {
#pragma HLS PIPELINE II=1
                    const int index = (((oc * RECEIVER_CHANNELS + ch) * 5 + kh) * 3 + kw);
                    g_conv1_weight_int8[bank][oc][ch][kh][kw] =
                        read_packed_i8(weights, WEIGHT_OFF_CONV1_WEIGHT, index);
                }
            }
        }
    }

ENC_LOAD_CONV1_BIAS:
    for (int oc = 0; oc < CONV1_OUT; ++oc) {
#pragma HLS PIPELINE II=1
        g_conv1_bias_int32[bank][oc] = word_to_i32(weights[WEIGHT_OFF_CONV1_BIAS + oc]);
    }

ENC_LOAD_CONV1_MULT:
    for (int oc = 0; oc < CONV1_OUT; ++oc) {
#pragma HLS PIPELINE II=1
        g_conv1_requant_mult[bank][oc] = word_to_i32(weights[WEIGHT_OFF_CONV1_MULT + oc]);
    }

ENC_LOAD_CONV1_SHIFT:
    for (int oc = 0; oc < CONV1_OUT; ++oc) {
#pragma HLS PIPELINE II=1
        g_conv1_requant_shift[bank][oc] = word_to_i32(weights[WEIGHT_OFF_CONV1_SHIFT + oc]);
    }

ENC_LOAD_CONV2_WEIGHT_OC:
    for (int oc = 0; oc < CONV2_OUT; ++oc) {
    ENC_LOAD_CONV2_WEIGHT_IC:
        for (int ic = 0; ic < CONV1_OUT; ++ic) {
        ENC_LOAD_CONV2_WEIGHT_KH:
        for (int kh = 0; kh < 3; ++kh) {
            ENC_LOAD_CONV2_WEIGHT_KW:
                for (int kw = 0; kw < 3; ++kw) {
#pragma HLS PIPELINE II=1
                    const int index = (((oc * CONV1_OUT + ic) * 3 + kh) * 3 + kw);
                    g_conv2_weight_int8[bank][oc][ic][kh][kw] =
                        read_packed_i8(weights, WEIGHT_OFF_CONV2_WEIGHT, index);
                }
            }
        }
    }

ENC_LOAD_CONV2_BIAS:
    for (int oc = 0; oc < CONV2_OUT; ++oc) {
#pragma HLS PIPELINE II=1
        g_conv2_bias_int32[bank][oc] = word_to_i32(weights[WEIGHT_OFF_CONV2_BIAS + oc]);
    }

ENC_LOAD_CONV2_MULT:
    for (int oc = 0; oc < CONV2_OUT; ++oc) {
#pragma HLS PIPELINE II=1
        g_conv2_requant_mult[bank][oc] = word_to_i32(weights[WEIGHT_OFF_CONV2_MULT + oc]);
    }

ENC_LOAD_CONV2_SHIFT:
    for (int oc = 0; oc < CONV2_OUT; ++oc) {
#pragma HLS PIPELINE II=1
        g_conv2_requant_shift[bank][oc] = word_to_i32(weights[WEIGHT_OFF_CONV2_SHIFT + oc]);
    }

ENC_LOAD_LUT1:
    for (int i = 0; i < 256; ++i) {
#pragma HLS PIPELINE II=1
        g_lut_encoder_gelu1_int8[bank][i] =
            read_packed_i8(weights, WEIGHT_OFF_LUT_ENCODER_GELU1, i);
    }

ENC_LOAD_LUT2:
    for (int i = 0; i < 256; ++i) {
#pragma HLS PIPELINE II=1
        g_lut_encoder_gelu2_int8[bank][i] =
            read_packed_i8(weights, WEIGHT_OFF_LUT_ENCODER_GELU2, i);
    }
}

static void load_fc2_weight(const packed_weight_t weights[WEIGHT_WORDS]) {
FC2_LOAD_WEIGHT_OUT:
    for (int o = 0; o < HIDDEN_DIM; ++o) {
    FC2_LOAD_WEIGHT_IN:
        for (int i = 0; i < HIDDEN_DIM; ++i) {
#pragma HLS PIPELINE II=1
            g_fc2_weight_int8[o][i] =
                read_packed_i8(weights, WEIGHT_OFF_FC2_WEIGHT, o * HIDDEN_DIM + i);
        }
    }
}

static void load_fc3_weight(const packed_weight_t weights[WEIGHT_WORDS]) {
FC3_LOAD_WEIGHT_OUT:
    for (int o = 0; o < POSE_DIM; ++o) {
    FC3_LOAD_WEIGHT_IN:
        for (int i = 0; i < HIDDEN_DIM; ++i) {
#pragma HLS PIPELINE II=1
            g_fc3_weight_int8[o][i] =
                read_packed_i8(weights, WEIGHT_OFF_FC3_WEIGHT, o * HIDDEN_DIM + i);
        }
    }
}

static void load_weight_blob(const packed_weight_t weights[WEIGHT_WORDS]) {
#pragma HLS INLINE off
    g_head_fc3_out_scale = word_to_float(weights[WEIGHT_OFF_OUTPUT_SCALE_BITS]);

LOAD_ENCODER_BANKS:
    for (int bank = 0; bank < NODE_COUNT; ++bank) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3 avg=3
        load_encoder_bank(weights, bank);
    }

    load_i32_array(weights, WEIGHT_OFF_FC1_BIAS, g_fc1_bias_int32, HIDDEN_DIM);
    load_i32_array(weights, WEIGHT_OFF_FC1_MULT, g_fc1_requant_mult, HIDDEN_DIM);
    load_i32_array(weights, WEIGHT_OFF_FC1_SHIFT, g_fc1_requant_shift, HIDDEN_DIM);

    load_fc2_weight(weights);
    load_i32_array(weights, WEIGHT_OFF_FC2_BIAS, g_fc2_bias_int32, HIDDEN_DIM);
    load_i32_array(weights, WEIGHT_OFF_FC2_MULT, g_fc2_requant_mult, HIDDEN_DIM);
    load_i32_array(weights, WEIGHT_OFF_FC2_SHIFT, g_fc2_requant_shift, HIDDEN_DIM);

    load_fc3_weight(weights);
    load_i32_array(weights, WEIGHT_OFF_FC3_BIAS, g_fc3_bias_int32, POSE_DIM);
    load_i32_array(weights, WEIGHT_OFF_FC3_MULT, g_fc3_requant_mult, POSE_DIM);
    load_i32_array(weights, WEIGHT_OFF_FC3_SHIFT, g_fc3_requant_shift, POSE_DIM);

    load_i8_array(weights, WEIGHT_OFF_LUT_HEAD_GELU1, g_lut_head_gelu1_int8, 256);
    load_i8_array(weights, WEIGHT_OFF_LUT_HEAD_GELU2, g_lut_head_gelu2_int8, 256);

    g_weights_loaded = true;
}

template <int OUT_DIM, int IN_DIM>
static void linear_requant(
    const qint8_t input[IN_DIM],
    qint8_t output[OUT_DIM],
    const signed char weight[OUT_DIM][IN_DIM],
    const int bias[OUT_DIM],
    const int requant_mult[OUT_DIM],
    const int requant_shift[OUT_DIM]) {
#pragma HLS ARRAY_PARTITION variable=weight cyclic factor=2 dim=2
LINEAR_REQUANT_OUT:
    for (int o = 0; o < OUT_DIM; ++o) {
#pragma HLS PIPELINE off
        qint32_t acc0 = 0;
        qint32_t acc1 = 0;
    LINEAR_REQUANT_IN:
        for (int i = 0; i < IN_DIM; i += 2) {
#pragma HLS PIPELINE II=1
            acc0 += mul_i8_dsp(input[i + 0], (qint8_t)weight[o][i + 0]);
            acc1 += mul_i8_dsp(input[i + 1], (qint8_t)weight[o][i + 1]);
        }
        qint32_t acc = (qint32_t)bias[o] + acc0 + acc1;
        output[o] = requant_acc_int(acc, requant_mult[o], requant_shift[o]);
    }
}

template <int NODE>
static void conv1_receiver(
    const qint8_t input[INPUT_CHANNELS * INPUT_H * INPUT_W],
    qint8_t output[CONV1_OUT][INPUT_H][INPUT_W]) {
CONV1_OC:
    for (int oc = 0; oc < CONV1_OUT; ++oc) {
    CONV1_H:
        for (int h = 0; h < INPUT_H; ++h) {
        CONV1_W:
            for (int w = 0; w < INPUT_W; ++w) {
#pragma HLS PIPELINE off
                qint32_t acc = g_conv1_bias_int32[NODE][oc];
            CONV1_CH:
                for (int ch = 0; ch < RECEIVER_CHANNELS; ++ch) {
                CONV1_KH:
                    for (int kh = 0; kh < 5; ++kh) {
                    CONV1_KW:
                        for (int kw = 0; kw < 3; ++kw) {
#pragma HLS PIPELINE II=1
                            int ih = h + kh - 2;
                            int iw = w + kw - 1;
                            if (ih >= 0 && ih < INPUT_H && iw >= 0 && iw < INPUT_W) {
                                int index = ((NODE + ch * NODE_COUNT) * INPUT_H + ih) * INPUT_W + iw;
                                acc += mul_i8_dsp(input[index], (qint8_t)g_conv1_weight_int8[NODE][oc][ch][kh][kw]);
                            }
                        }
                    }
                }
                output[oc][h][w] = requant_acc_int(acc, g_conv1_requant_mult[NODE][oc], g_conv1_requant_shift[NODE][oc]);
            }
        }
    }
}

template <int NODE>
static void gelu_conv1(qint8_t data[CONV1_OUT][INPUT_H][INPUT_W]) {
GELU1_OC:
    for (int c = 0; c < CONV1_OUT; ++c) {
    GELU1_H:
        for (int h = 0; h < INPUT_H; ++h) {
        GELU1_W:
            for (int w = 0; w < INPUT_W; ++w) {
#pragma HLS PIPELINE II=1
                data[c][h][w] = (qint8_t)g_lut_encoder_gelu1_int8[NODE][(int)data[c][h][w] + 128];
            }
        }
    }
}

template <int NODE>
static void conv2_receiver(
    const qint8_t input[CONV1_OUT][INPUT_H][INPUT_W],
    qint8_t output[CONV2_OUT][INPUT_H][INPUT_W]) {
#pragma HLS ARRAY_PARTITION variable=input cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=g_conv2_weight_int8 cyclic factor=8 dim=3
CONV2_OC:
    for (int oc = 0; oc < CONV2_OUT; ++oc) {
    CONV2_H:
        for (int h = 0; h < INPUT_H; ++h) {
        CONV2_W:
            for (int w = 0; w < INPUT_W; ++w) {
#pragma HLS PIPELINE off
                qint32_t acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma HLS ARRAY_PARTITION variable=acc complete
            CONV2_IC:
                for (int ic = 0; ic < CONV1_OUT; ic += 8) {
                CONV2_KH:
                    for (int kh = 0; kh < 3; ++kh) {
                    CONV2_KW:
                        for (int kw = 0; kw < 3; ++kw) {
#pragma HLS PIPELINE II=1
                            int ih = h + kh - 1;
                            int iw = w + kw - 1;
                            if (ih >= 0 && ih < INPUT_H && iw >= 0 && iw < INPUT_W) {
                            CONV2_LANE:
                                for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
                                    acc[lane] += mul_i8_dsp(input[ic + lane][ih][iw], (qint8_t)g_conv2_weight_int8[NODE][oc][ic + lane][kh][kw]);
                                }
                            }
                        }
                    }
                }
                qint32_t total = (qint32_t)g_conv2_bias_int32[NODE][oc];
            CONV2_ACC_SUM:
                for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
                    total += acc[lane];
                }
                output[oc][h][w] = requant_acc_int(total, g_conv2_requant_mult[NODE][oc], g_conv2_requant_shift[NODE][oc]);
            }
        }
    }
}

template <int NODE>
static void gelu_conv2(qint8_t data[CONV2_OUT][INPUT_H][INPUT_W]) {
GELU2_OC:
    for (int c = 0; c < CONV2_OUT; ++c) {
    GELU2_H:
        for (int h = 0; h < INPUT_H; ++h) {
        GELU2_W:
            for (int w = 0; w < INPUT_W; ++w) {
#pragma HLS PIPELINE II=1
                data[c][h][w] = (qint8_t)g_lut_encoder_gelu2_int8[NODE][(int)data[c][h][w] + 128];
            }
        }
    }
}

static int div_floor_bin_start(int out_index, int in_size, int out_size) {
#pragma HLS INLINE
    return (out_index * in_size) / out_size;
}

static int div_ceil_bin_end(int out_index_plus_1, int in_size, int out_size) {
#pragma HLS INLINE
    return (out_index_plus_1 * in_size + out_size - 1) / out_size;
}

template <int NODE>
static void adaptive_pool_flatten_receiver(
    const qint8_t input[CONV2_OUT][INPUT_H][INPUT_W],
    qint8_t flatten_segment[RECEIVER_FLATTEN_DIM]) {
POOL_C:
    for (int c = 0; c < CONV2_OUT; ++c) {
    POOL_H_LOOP:
        for (int oh = 0; oh < POOL_H; ++oh) {
        POOL_W_LOOP:
            for (int ow = 0; ow < POOL_W; ++ow) {
#pragma HLS PIPELINE off
                int hs = div_floor_bin_start(oh, INPUT_H, POOL_H);
                int he = div_ceil_bin_end(oh + 1, INPUT_H, POOL_H);
                int ws = div_floor_bin_start(ow, INPUT_W, POOL_W);
                int we = div_ceil_bin_end(ow + 1, INPUT_W, POOL_W);
                int sum = 0;
            POOL_IH:
                for (int ih = hs; ih < he; ++ih) {
#pragma HLS loop_tripcount min=16 max=16 avg=16
                POOL_IW:
                    for (int iw = ws; iw < we; ++iw) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=2 max=3 avg=3
                        sum += (int)input[c][ih][iw];
                    }
                }
                // 128x10 -> 8x4 adaptive pooling always uses 16x3 bins.
                long long scaled = round_shift_signed((long long)sum * (long long)g_pool_requant_mult[NODE], g_pool_requant_shift[NODE]);
                long long rounded = scaled >= 0 ? scaled + 24 : scaled - 24;
                int pooled = floor_div48_saturated(rounded);
                int index = (c * POOL_H + oh) * POOL_W + ow;
                flatten_segment[index] = sat_int8(pooled);
            }
        }
    }
}

template <int NODE>
static void encode_receiver_path(
    const qint8_t input[INPUT_CHANNELS * INPUT_H * INPUT_W],
    qint8_t flatten_segment[RECEIVER_FLATTEN_DIM]) {
#pragma HLS INLINE off
    qint8_t conv1[CONV1_OUT][INPUT_H][INPUT_W];
    qint8_t conv2[CONV2_OUT][INPUT_H][INPUT_W];

    conv1_receiver<NODE>(input, conv1);
    gelu_conv1<NODE>(conv1);
    conv2_receiver<NODE>(conv1, conv2);
    gelu_conv2<NODE>(conv2);
    adaptive_pool_flatten_receiver<NODE>(conv2, flatten_segment);
}

static void load_input_copies(
    const qint8_t input[INPUT_CHANNELS * INPUT_H * INPUT_W],
    qint8_t input0[INPUT_CHANNELS * INPUT_H * INPUT_W],
    qint8_t input1[INPUT_CHANNELS * INPUT_H * INPUT_W],
    qint8_t input2[INPUT_CHANNELS * INPUT_H * INPUT_W]) {
#pragma HLS INLINE off
LOAD_INPUT_COPIES:
    for (int i = 0; i < INPUT_CHANNELS * INPUT_H * INPUT_W; ++i) {
#pragma HLS PIPELINE II=1
        qint8_t value = input[i];
        input0[i] = value;
        input1[i] = value;
        input2[i] = value;
    }
}

static void encode_receivers_parallel_direct(
    const qint8_t input0[INPUT_CHANNELS * INPUT_H * INPUT_W],
    const qint8_t input1[INPUT_CHANNELS * INPUT_H * INPUT_W],
    const qint8_t input2[INPUT_CHANNELS * INPUT_H * INPUT_W],
    qint8_t flatten_banks[NODE_COUNT][RECEIVER_FLATTEN_DIM]) {
#pragma HLS dataflow disable_start_propagation
    encode_receiver_path<0>(input0, flatten_banks[0]);
    encode_receiver_path<1>(input1, flatten_banks[1]);
    encode_receiver_path<2>(input2, flatten_banks[2]);
}

static void encode_all_receivers_direct(
    const qint8_t input[INPUT_CHANNELS * INPUT_H * INPUT_W],
    qint8_t flatten_banks[NODE_COUNT][RECEIVER_FLATTEN_DIM]) {
    qint8_t input0[INPUT_CHANNELS * INPUT_H * INPUT_W];
    qint8_t input1[INPUT_CHANNELS * INPUT_H * INPUT_W];
    qint8_t input2[INPUT_CHANNELS * INPUT_H * INPUT_W];

    load_input_copies(input, input0, input1, input2);
    encode_receivers_parallel_direct(input0, input1, input2, flatten_banks);
}

static qint8_t read_banked_flatten(
    const qint8_t input[NODE_COUNT][RECEIVER_FLATTEN_DIM],
    int index) {
#pragma HLS INLINE
    int node = index / RECEIVER_FLATTEN_DIM;
    int offset = index - node * RECEIVER_FLATTEN_DIM;
    return input[node][offset];
}

static void fc1_requant_4way_banked(
    const qint8_t input[NODE_COUNT][RECEIVER_FLATTEN_DIM],
    const packed_weight_t weights[WEIGHT_WORDS],
    qint8_t output[HIDDEN_DIM]) {
FC1_OUT:
    for (int o = 0; o < HIDDEN_DIM; ++o) {
        qint32_t acc0 = 0;
        qint32_t acc1 = 0;
        qint32_t acc2 = 0;
        qint32_t acc3 = 0;
        int base = o * (FLATTEN_DIM / FC1_PACK);
    FC1_IN:
        for (int i = 0; i < FLATTEN_DIM; i += 4) {
#pragma HLS PIPELINE II=1
            packed_weight_t packed = weights[WEIGHT_OFF_FC1_WEIGHT + base + i / FC1_PACK];
            acc0 += mul_i8_dsp(read_banked_flatten(input, i + 0), unpack_i8(packed, 0));
            acc1 += mul_i8_dsp(read_banked_flatten(input, i + 1), unpack_i8(packed, 1));
            acc2 += mul_i8_dsp(read_banked_flatten(input, i + 2), unpack_i8(packed, 2));
            acc3 += mul_i8_dsp(read_banked_flatten(input, i + 3), unpack_i8(packed, 3));
        }
        qint32_t acc = (qint32_t)g_fc1_bias_int32[o] + acc0 + acc1 + acc2 + acc3;
        output[o] = requant_acc_int(acc, g_fc1_requant_mult[o], g_fc1_requant_shift[o]);
    }
}

static void apply_lut_128(const qint8_t input[HIDDEN_DIM],
                          qint8_t output[HIDDEN_DIM],
                          const signed char lut[256]) {
LUT128_LOOP:
    for (int i = 0; i < HIDDEN_DIM; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = (qint8_t)lut[(int)input[i] + 128];
    }
}

static void apply_head_gelu1(const qint8_t input[HIDDEN_DIM],
                             qint8_t output[HIDDEN_DIM]) {
#pragma HLS INLINE off
    apply_lut_128(input, output, g_lut_head_gelu1_int8);
}

static void apply_head_gelu2(const qint8_t input[HIDDEN_DIM],
                             qint8_t output[HIDDEN_DIM]) {
#pragma HLS INLINE off
    apply_lut_128(input, output, g_lut_head_gelu2_int8);
}

static void fc2_requant(const qint8_t input[HIDDEN_DIM],
                        qint8_t output[HIDDEN_DIM]) {
#pragma HLS INLINE off
    linear_requant<128, 128>(
        input,
        output,
        g_fc2_weight_int8,
        g_fc2_bias_int32,
        g_fc2_requant_mult,
        g_fc2_requant_shift);
}

static void fc3_requant(const qint8_t input[HIDDEN_DIM],
                        qint8_t output[POSE_DIM]) {
#pragma HLS INLINE off
    linear_requant<24, 128>(
        input,
        output,
        g_fc3_weight_int8,
        g_fc3_bias_int32,
        g_fc3_requant_mult,
        g_fc3_requant_shift);
}

static void write_final_pose(const qint8_t pose_q[POSE_DIM],
                             float final_pose[POSE_DIM]) {
FINAL_LOOP:
    for (int i = 0; i < POSE_DIM; ++i) {
#pragma HLS PIPELINE II=1
        final_pose[i] = (float)pose_q[i] * g_head_fc3_out_scale;
    }
}

void full_pose_accel(
    const qint8_t input[INPUT_CHANNELS * INPUT_H * INPUT_W],
    const packed_weight_t weights[WEIGHT_WORDS],
    int command,
    float final_pose[POSE_DIM]) {
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=11520
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem1 depth=105748
#pragma HLS INTERFACE m_axi port=final_pose offset=slave bundle=gmem2 depth=24
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=command bundle=control
#pragma HLS INTERFACE s_axilite port=final_pose bundle=control
#pragma HLS ARRAY_PARTITION variable=g_conv1_weight_int8 complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_conv1_bias_int32 complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_conv1_requant_mult complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_conv1_requant_shift complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_conv2_weight_int8 complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_conv2_bias_int32 complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_conv2_requant_mult complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_conv2_requant_shift complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_lut_encoder_gelu1_int8 complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_lut_encoder_gelu2_int8 complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_pool_requant_mult complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_pool_requant_shift complete dim=1

    if (command != 0 || !g_weights_loaded) {
        load_weight_blob(weights);
        if (command != 0) {
            return;
        }
    }

    qint8_t flatten_banks[NODE_COUNT][RECEIVER_FLATTEN_DIM];
    qint8_t head_fc1[HIDDEN_DIM];
    qint8_t head_gelu1[HIDDEN_DIM];
    qint8_t head_fc2[HIDDEN_DIM];
    qint8_t head_gelu2[HIDDEN_DIM];
    qint8_t pose_q[POSE_DIM];

#pragma HLS ARRAY_PARTITION variable=flatten_banks complete dim=1
#pragma HLS ARRAY_PARTITION variable=head_fc1 cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=head_gelu1 cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=head_fc2 cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=head_gelu2 cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=pose_q complete

    encode_all_receivers_direct(input, flatten_banks);

    fc1_requant_4way_banked(flatten_banks, weights, head_fc1);
    apply_head_gelu1(head_fc1, head_gelu1);
    fc2_requant(head_gelu1, head_fc2);
    apply_head_gelu2(head_fc2, head_gelu2);
    fc3_requant(head_gelu2, pose_q);
    write_final_pose(pose_q, final_pose);
}
