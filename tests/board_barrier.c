/* SPDX-License-Identifier: MIT
 * Fence-free completion probe (docs/plans/pipelining-plan.md S4 residual).
 *
 * The attached kernel has no CONFIG_ROCKCHIP_RKNPU_FENCE, so `JOB_FENCE_OUT` returns
 * -EINVAL and there is no pollable completion fd. Jobs do run in order per core, so a
 * **barrier job** - any small blocking submission after a queued one - completes only
 * after the queued work does:
 *
 *   ornpu_submit_flags(A, JOB_NONBLOCK, NULL);   queue the inference
 *   ornpu_run(barrier, ...);                     blocking: completes after A
 *   ornpu_sync_outputs(A);                       A's outputs are final
 *
 * That gives a single-instance stream with **lag 0** (the output is read as soon as it
 * is produced) at the cost of one small job per inference, instead of the lag-1
 * queue-then-drain pipeline in `board_async.c` that needs a second full inference.
 *
 * usage: board_barrier <model.bin> <input.u8> <iterations> [expected.i8 [barrier.bin]]
 */
#include "open_rknpu.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define JOB_NONBLOCK 0x2u

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static size_t read_file(const char *path, void *data, size_t size) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    size_t got = fread(data, 1, size, file);
    fclose(file);
    return got;
}

static double median(double *values, unsigned count) {
    double *sorted = malloc(count * sizeof(double));
    memcpy(sorted, values, count * sizeof(double));
    qsort(sorted, count, sizeof(double), compare_double);
    double result = sorted[count / 2];
    free(sorted);
    return result;
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 6) {
        fprintf(stderr, "usage: %s <model.bin> <input.u8> <iterations> "
                        "[expected.i8 [barrier.bin]]\n", argv[0]);
        return 2;
    }
    const char *expected_path = argc >= 5 ? argv[4] : NULL;
    const char *barrier_path = argc >= 6 ? argv[5] : NULL;
    unsigned iterations = (unsigned)strtoul(argv[3], NULL, 10);
    if (!iterations || iterations > 100000) return 2;
    ornpu_model *model = NULL;
    ornpu_info info;
    if (ornpu_open(argv[1], &model) || ornpu_get_info(model, &info)) {
        fprintf(stderr, "open failed\n");
        return 1;
    }
    size_t in_size = info.input_bytes, out_size = info.output_bytes;
    uint8_t *input = malloc(in_size);
    int8_t *output = malloc(out_size), *expected = expected_path ? malloc(out_size) : NULL;
    if (!input || !output || read_file(argv[2], input, in_size) != in_size ||
        (expected_path && read_file(expected_path, expected, out_size) != out_size)) {
        fprintf(stderr, "input/expected read failed\n");
        return 1;
    }
    /* warm up the queued model */
    for (unsigned i = 0; i < 8; i++)
        if (ornpu_run(model, input, in_size, output, out_size)) {
            fprintf(stderr, "warmup failed\n");
            return 1;
        }

    ornpu_model *barrier = NULL;
    ornpu_info barrier_info;
    uint8_t *barrier_in = NULL;
    int8_t *barrier_out = NULL;
    if (barrier_path) {
        if (ornpu_open(barrier_path, &barrier) || ornpu_get_info(barrier, &barrier_info)) {
            fprintf(stderr, "barrier open failed\n");
            return 1;
        }
        barrier_in = malloc(barrier_info.input_bytes);
        barrier_out = malloc(barrier_info.output_bytes);
        memset(barrier_in, 0, barrier_info.input_bytes);
        for (unsigned i = 0; i < 8; i++)
            if (ornpu_run(barrier, barrier_in, barrier_info.input_bytes, barrier_out,
                          barrier_info.output_bytes)) {
                fprintf(stderr, "barrier warmup failed\n");
                return 1;
            }
        double *barrier_times = malloc(iterations * sizeof(double));
        for (unsigned i = 0; i < iterations; i++) {
            double start = now_us();
            int rc = ornpu_run(barrier, barrier_in, barrier_info.input_bytes, barrier_out,
                               barrier_info.output_bytes);
            barrier_times[i] = now_us() - start;
            if (rc) { fprintf(stderr, "barrier run failed\n"); return 1; }
        }
        printf("barrier: tasks=%u median=%.1fus\n", barrier_info.task_count,
               median(barrier_times, iterations));
        free(barrier_times);
    }

    double *sync = malloc(iterations * sizeof(double));
    double *queued = malloc(iterations * sizeof(double));
    unsigned bad_sync = 0, bad_queued = 0;
    for (unsigned i = 0; i < iterations; i++) {
        double start = now_us();
        if (ornpu_run(model, input, in_size, output, out_size)) {
            fprintf(stderr, "sync run failed\n");
            return 1;
        }
        sync[i] = now_us() - start;
        if (expected && memcmp(expected, output, out_size)) bad_sync++;
        if (!barrier) continue;
        start = now_us();
        if (ornpu_submit_flags(model, JOB_NONBLOCK, NULL)) {
            fprintf(stderr, "nonblock submit failed\n");
            return 1;
        }
        if (ornpu_run(barrier, barrier_in, barrier_info.input_bytes, barrier_out,
                      barrier_info.output_bytes)) {
            fprintf(stderr, "barrier run failed\n");
            return 1;
        }
        if (ornpu_sync_outputs(model)) {
            fprintf(stderr, "sync outputs failed\n");
            return 1;
        }
        queued[i] = now_us() - start;
        if (expected && memcmp(expected, output, out_size)) bad_queued++;
    }
    double sync_median = median(sync, iterations);
    printf("single-instance: runs=%u tasks=%u %s sync_median=%.1fus mismatches=%u\n",
           iterations, info.task_count, info.submission_serial ? "serial" : "batched",
           sync_median, bad_sync);
    if (barrier) {
        double queued_median = median(queued, iterations);
        printf("barrier-completed: median=%.1fus overhead=%.1fus (%.0f%%) mismatches=%u\n",
               queued_median, queued_median - sync_median,
               100.0 * (queued_median - sync_median) / sync_median, bad_queued);
    }
    ornpu_close(model);
    if (barrier) ornpu_close(barrier);
    free(queued);
    free(sync);
    free(barrier_out);
    free(barrier_in);
    free(expected);
    free(output);
    free(input);
    return (bad_sync || bad_queued) ? 1 : 0;
}
