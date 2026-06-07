#pragma once

#include <ap_int.h>

static const int INPUT_CHANNELS = 9;
static const int INPUT_H = 128;
static const int INPUT_W = 10;
static const int NODE_COUNT = 3;
static const int RECEIVER_CHANNELS = 3;
static const int CONV1_OUT = 16;
static const int CONV2_OUT = 32;
static const int POOL_H = 8;
static const int POOL_W = 4;
static const int RECEIVER_FLATTEN_DIM = CONV2_OUT * POOL_H * POOL_W;
static const int FLATTEN_DIM = NODE_COUNT * RECEIVER_FLATTEN_DIM;
static const int HIDDEN_DIM = 128;
static const int POSE_DIM = 24;

typedef ap_int<8> qint8_t;
typedef ap_int<32> qint32_t;
typedef ap_uint<32> packed_weight_t;

static const int FC1_PACK = 4;
static const int FC1_WEIGHT_WORDS = HIDDEN_DIM * FLATTEN_DIM / FC1_PACK;
static const int WEIGHT_MAGIC = 0x36574C50;
static const int WEIGHT_VERSION = 2;
static const int WEIGHT_WORDS = 105748;

static const int WEIGHT_OFF_MAGIC = 0;
static const int WEIGHT_OFF_VERSION = 1;
static const int WEIGHT_OFF_TOTAL_WORDS = 2;
static const int WEIGHT_OFF_INPUT_SCALE_BITS = 3;
static const int WEIGHT_OFF_OUTPUT_SCALE_BITS = 4;
static const int WEIGHT_OFF_POOL_MULT = 5;
static const int WEIGHT_OFF_POOL_SHIFT = 6;
static const int WEIGHT_OFF_CONV1_WEIGHT = 8;
static const int WEIGHT_OFF_CONV1_BIAS = 188;
static const int WEIGHT_OFF_CONV1_MULT = 204;
static const int WEIGHT_OFF_CONV1_SHIFT = 220;
static const int WEIGHT_OFF_CONV2_WEIGHT = 236;
static const int WEIGHT_OFF_CONV2_BIAS = 1388;
static const int WEIGHT_OFF_CONV2_MULT = 1420;
static const int WEIGHT_OFF_CONV2_SHIFT = 1452;
static const int WEIGHT_OFF_FC1_WEIGHT = 1484;
static const int WEIGHT_OFF_FC1_BIAS = 99788;
static const int WEIGHT_OFF_FC1_MULT = 99916;
static const int WEIGHT_OFF_FC1_SHIFT = 100044;
static const int WEIGHT_OFF_FC2_WEIGHT = 100172;
static const int WEIGHT_OFF_FC2_BIAS = 104268;
static const int WEIGHT_OFF_FC2_MULT = 104396;
static const int WEIGHT_OFF_FC2_SHIFT = 104524;
static const int WEIGHT_OFF_FC3_WEIGHT = 104652;
static const int WEIGHT_OFF_FC3_BIAS = 105420;
static const int WEIGHT_OFF_FC3_MULT = 105444;
static const int WEIGHT_OFF_FC3_SHIFT = 105468;
static const int WEIGHT_OFF_LUT_ENCODER_GELU1 = 105492;
static const int WEIGHT_OFF_LUT_ENCODER_GELU2 = 105556;
static const int WEIGHT_OFF_LUT_HEAD_GELU1 = 105620;
static const int WEIGHT_OFF_LUT_HEAD_GELU2 = 105684;

void full_pose_accel(
    const qint8_t input[INPUT_CHANNELS * INPUT_H * INPUT_W],
    const packed_weight_t weights[WEIGHT_WORDS],
    int command,
    float final_pose[POSE_DIM]);
