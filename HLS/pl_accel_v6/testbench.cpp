#include <cmath>
#include <cstdio>

#include "full_pose.h"
#include "test_vectors.h"

int main() {
    static qint8_t input[INPUT_CHANNELS * INPUT_H * INPUT_W];
    static packed_weight_t weights[WEIGHT_WORDS];
    static float output[POSE_DIM];

    for (int i = 0; i < INPUT_CHANNELS * INPUT_H * INPUT_W; ++i) {
        input[i] = (qint8_t)test_input[i];
    }
    for (int i = 0; i < WEIGHT_WORDS; ++i) {
        weights[i] = (packed_weight_t)test_weights[i];
    }

    full_pose_accel(input, weights, 1, output);
    full_pose_accel(input, weights, 0, output);

    float checksum = 0.0f;
    float max_abs_err = 0.0f;
    int invalid = 0;
    for (int i = 0; i < POSE_DIM; ++i) {
        checksum += output[i];
        float err = std::fabs(output[i] - test_expected_pose[i]);
        if (err > max_abs_err) {
            max_abs_err = err;
        }
        if (!std::isfinite(output[i]) || err > 1.0e-5f) {
            ++invalid;
        }
    }

    std::printf("int8 fast cnn checksum=%0.9f max_abs_err=%0.9f invalid=%d\n",
                checksum,
                max_abs_err,
                invalid);
    for (int i = 0; i < POSE_DIM; ++i) {
        std::printf("pose[%02d]=%0.9f expected=%0.9f\n", i, output[i], test_expected_pose[i]);
    }

    return invalid == 0 ? 0 : 1;
}
