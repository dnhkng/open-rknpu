/* SPDX-License-Identifier: MIT
 * Board runner for the mel-CNN spoken-digit example (examples/mel-kws).
 *
 * The whole model runs on the NPU: Conv/Relu/MaxPool/Conv/Relu/MaxPool/Conv emitted by
 * `open_rknpu.walk`. This runner streams every test utterance through one container,
 * checks the INT8 output bytes against the host reference and scores the utterance by
 * averaging the 8x8 logit map per class (global average pooling).
 *
 *   ./mel-kws-run prefix.bin inputs.u8 labels.u8 expected.i8 [rounds] [actual.i8]
 *
 * inputs.u8   N * 3072 bytes, NHWC UINT8 (input scale 1/255, zero point 0)
 * labels.u8   N bytes, digit 0..9
 * expected.i8 N * 640 bytes, NHWC INT8 reference output
 */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "open_rknpu.h"

#define CLASSES 10u

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static void *read_file(const char *path, size_t bytes) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    void *data = malloc(bytes);
    if (!data || fread(data, 1, bytes, file) != bytes) {
        fprintf(stderr, "read %s: short file\n", path);
        exit(1);
    }
    fclose(file);
    return data;
}

static int classify(const int8_t *output, uint32_t positions) {
    /* output is NHWC: `positions` spatial cells x CLASSES */
    int32_t score[CLASSES];
    memset(score, 0, sizeof(score));
    for (uint32_t position = 0; position < positions; position++)
        for (uint32_t label = 0; label < CLASSES; label++)
            score[label] += output[position * CLASSES + label];
    int best = 0;
    for (uint32_t label = 1; label < CLASSES; label++)
        if (score[label] > score[best]) best = (int)label;
    return best;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s prefix.bin inputs.u8 labels.u8 expected.i8 [rounds]\n", argv[0]);
        return 2;
    }
    int rounds = argc > 5 ? atoi(argv[5]) : 5;
    ornpu_model *model = NULL;
    ornpu_info info;
    if (ornpu_inspect(argv[1], &info) < 0 || ornpu_open(argv[1], &model) < 0) {
        fprintf(stderr, "open %s failed\n", argv[1]);
        return 1;
    }
    /* sample count comes from the label file size */
    FILE *labels_file = fopen(argv[3], "rb");
    if (!labels_file) {
        fprintf(stderr, "open %s: %s\n", argv[3], strerror(errno));
        return 1;
    }
    fseek(labels_file, 0, SEEK_END);
    long count = ftell(labels_file);
    fclose(labels_file);
    if (count <= 0) {
        fprintf(stderr, "empty label file\n");
        return 1;
    }
    /* The API buffer sizes follow the tensor table (no padding in API buffers). */
    uint32_t input_bytes = info.batch * info.height * info.width * info.input_channels;
    uint32_t output_bytes = info.batch * info.output_height * info.output_width * info.output_channels;
    uint32_t positions = info.output_height * info.output_width;
    uint8_t *inputs = read_file(argv[2], (size_t)count * input_bytes);
    int8_t *expected = read_file(argv[4], (size_t)count * output_bytes);
    uint8_t *labels = read_file(argv[3], (size_t)count);
    int8_t *output = malloc(output_bytes);
    int8_t *actual = malloc((size_t)count * output_bytes);

    printf("container: %ux%ux%u -> %ux%ux%u, %u tasks, %u engine run(s), submission %s\n",
           info.height, info.width, info.input_channels, info.output_height, info.output_width,
           info.output_channels, info.task_count, info.engine_runs,
           info.submission_serial ? "serial" : "batched");
    printf("samples: %ld, rounds: %d\n", count, rounds);

    long correct = 0, mismatched_bytes = 0, bad_samples = 0;
    long per_digit_correct[CLASSES] = {0}, per_digit_total[CLASSES] = {0};
    double first_sweep_us = 0, total_us = 0;
    long inferences = 0;
    for (int round = 0; round < rounds; round++) {
        double sweep_start = now_us();
        for (long index = 0; index < count; index++) {
            const uint8_t *input = inputs + (size_t)index * input_bytes;
            int8_t *want = expected + (size_t)index * output_bytes;
            int rc = ornpu_run(model, input, input_bytes, output, output_bytes);
            if (rc < 0) {
                fprintf(stderr, "inference %ld failed (rc=%d)\n", index, rc);
                return 1;
            }
            if (round == 0) {
                memcpy(actual + (size_t)index * output_bytes, output, output_bytes);
                if (memcmp(output, want, output_bytes) != 0) {
                    bad_samples++;
                    for (uint32_t byte = 0; byte < output_bytes; byte++)
                        if (output[byte] != want[byte]) mismatched_bytes++;
                }
                int predicted = classify(output, positions);
                int truth = labels[index];
                per_digit_total[truth]++;
                if (predicted == truth) {
                    correct++;
                    per_digit_correct[truth]++;
                }
            }
            inferences++;
        }
        double sweep_us = now_us() - sweep_start;
        if (round == 0) first_sweep_us = sweep_us;
        total_us += sweep_us;
    }

    printf("PASS: %ld models, %ld inferences, %ld exact bytes\n", count, inferences,
           (long)count * output_bytes - mismatched_bytes);
    if (bad_samples)
        printf("FAIL: %ld samples differ from the reference (%ld bytes)\n", bad_samples, mismatched_bytes);
    printf("accuracy: %ld/%ld (%.2f%%)\n", correct, count, 100.0 * correct / count);
    for (uint32_t label = 0; label < CLASSES; label++)
        if (per_digit_total[label])
            printf("  digit %u: %ld/%ld\n", label, per_digit_correct[label], per_digit_total[label]);
    printf("latency: first sweep %.1f us/inference, mean of %d sweeps %.1f us/inference\n",
           first_sweep_us / count, rounds, total_us / inferences);
    if (argc > 6) {
        FILE *dump = fopen(argv[6], "wb");
        if (dump) {
            fwrite(actual, 1, (size_t)count * output_bytes, dump);
            fclose(dump);
        }
    }

    ornpu_close(model);
    free(inputs);
    free(expected);
    free(labels);
    free(output);
    free(actual);
    return bad_samples ? 1 : 0;
}
