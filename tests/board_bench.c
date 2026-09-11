/* SPDX-License-Identifier: MIT
 * Latency and stability harness for one model (v5 named-tensor or legacy).
 *
 * Runs the model repeatedly and reports wall time, plus mismatches against an
 * optional expected file and against the first run (which catches
 * nondeterministic/stale results under task overlap).
 *
 * usage: board_bench <model.bin> <input.u8> <iterations> [expected.i8]
 *
 * Inputs are streamed from one file holding every external input concatenated in
 * tensor-index order (the same fixture convention as board_io); every external
 * output is compared against the expected file.
 */
#include "open_rknpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

int main(int argc, char **argv) {
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "usage: %s <model.bin> <input.u8> <iterations> [expected.i8]\n", argv[0]);
        return 2;
    }
    const char *expected_path = argc == 5 ? argv[4] : NULL;
    unsigned iterations = (unsigned)strtoul(argv[3], NULL, 10);
    if (!iterations || iterations > 1000000) return 2;
    ornpu_model *model = NULL;
    ornpu_info info;
    if (ornpu_open(argv[1], &model) || ornpu_get_info(model, &info)) {
        fprintf(stderr, "open failed\n");
        return 1;
    }
    unsigned ni = info.input_tensor_count, no = info.output_tensor_count;
    int named = info.tensor_count != 0;
    if (named) {
        if (!ni || !no || ni > 8 || no > 8) {
            fprintf(stderr, "unsupported tensor counts %u/%u\n", ni, no);
            return 1;
        }
    } else {
        ni = no = 1;
    }
    size_t in_total = 0, out_total = 0;
    ornpu_io *ins = calloc(ni, sizeof(*ins)), *outs = calloc(no, sizeof(*outs));
    for (unsigned t = 0; t < info.tensor_count; t++) {
        ornpu_tensor_info ti;
        if (ornpu_get_tensor(model, t, &ti)) { fprintf(stderr, "tensor %u failed\n", t); return 1; }
        if (ti.role == ORNPU_TENSOR_INPUT) in_total += ti.api_bytes;
        else if (ti.role == ORNPU_TENSOR_OUTPUT) out_total += ti.api_bytes;
    }
    if (!named) { in_total = info.input_bytes; out_total = info.output_bytes; }
    uint8_t *input = malloc(in_total);
    int8_t *output = malloc(out_total), *first = malloc(out_total), *expected = NULL;
    double *times = malloc(iterations * sizeof(double));
    if (!ins || !outs || !input || !output || !first || !times) return 1;
    for (unsigned t = 0; t < info.tensor_count; t++) {
        ornpu_tensor_info ti;
        ornpu_get_tensor(model, t, &ti);
        if (ti.role == ORNPU_TENSOR_INPUT && ti.index < ni)
            ins[ti.index] = (ornpu_io){t, input + ti.api_offset, ti.api_bytes};
        else if (ti.role == ORNPU_TENSOR_OUTPUT && ti.index < no)
            outs[ti.index] = (ornpu_io){t, output + ti.api_offset, ti.api_bytes};
    }
    if (read_file(argv[2], input, in_total) != in_total) {
        fprintf(stderr, "input read failed (want %zu bytes)\n", in_total);
        return 1;
    }
    if (expected_path) {
        expected = malloc(out_total);
        if (!expected || read_file(expected_path, expected, out_total) != out_total) {
            fprintf(stderr, "expected read failed (want %zu bytes)\n", out_total);
            return 1;
        }
    }
    unsigned warmup = iterations / 4 < 8 ? iterations / 4 : 8;
    for (unsigned i = 0; i < warmup; i++) {
        int rc = named ? ornpu_run_io(model, ins, ni, outs, no)
                       : ornpu_run(model, input, in_total, output, out_total);
        if (rc) {
            fprintf(stderr, "warmup run failed: rc=%d tasks=%u %s inputs=%u outputs=%u\n",
                    rc, info.task_count, info.submission_serial ? "serial" : "batched", ni, no);
            return 1;
        }
    }
    int mismatches = 0, unstable = 0;
    unsigned runs = 0;
    for (unsigned i = 0; i < iterations; i++) {
        double start = now_us();
        int rc = named ? ornpu_run_io(model, ins, ni, outs, no)
                       : ornpu_run(model, input, in_total, output, out_total);
        times[runs] = now_us() - start;
        if (rc) {
            fprintf(stderr, "run %u failed: %d\n", i, rc);
            return 1;
        }
        if (i == 0) memcpy(first, output, out_total);
        if (memcmp(first, output, out_total)) unstable++;
        if (expected && memcmp(expected, output, out_total)) mismatches++;
        runs++;
    }
    double *sorted = malloc(runs * sizeof(double));
    memcpy(sorted, times, runs * sizeof(double));
    qsort(sorted, runs, sizeof(double), compare_double);
    double sum = 0;
    for (unsigned i = 0; i < runs; i++) sum += times[i];
    printf("runs=%u tasks=%u %s min=%.1fus median=%.1fus mean=%.1fus max=%.1fus "
           "unstable=%d mismatches=%d engine_runs=%u%s\n",
           runs, info.task_count, info.submission_serial ? "serial" : "batched",
           sorted[0], sorted[runs / 2], sum / runs, sorted[runs - 1], unstable, mismatches,
           info.engine_runs, expected ? "" : " (no expected file)");
    free(sorted);
    free(times);
    free(expected);
    free(first);
    free(output);
    free(input);
    free(outs);
    free(ins);
    ornpu_close(model);
    return (mismatches || unstable) ? 1 : 0;
}
