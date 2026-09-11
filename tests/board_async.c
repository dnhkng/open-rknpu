/* SPDX-License-Identifier: MIT
 * Cross-job pipelining probe (docs/plans/pipelining-plan.md S4).
 *
 * Measures the blocking baseline, then probes the driver's job flags:
 *   NONBLOCK             - does SUBMIT return before the job completes?
 *   NONBLOCK|FENCE_OUT   - does the driver return a pollable fence fd?
 *   |FENCE_IN            - does it accept and wait on that fd for the next job?
 * Outputs are compared against the expected file after the queued job completes.
 *
 * usage: board_async <model.bin> <input.u8> <iterations> [expected.i8]
 */
#include "open_rknpu.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define JOB_NONBLOCK 0x2u
#define JOB_FENCE_IN 0x8u
#define JOB_FENCE_OUT 0x10u

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
    unsigned iterations = (unsigned)strtoul(argv[3], NULL, 10);
    ornpu_model *model = NULL;
    ornpu_info info;
    if (ornpu_open(argv[1], &model) || ornpu_get_info(model, &info)) {
        fprintf(stderr, "open failed\n");
        return 1;
    }
    unsigned ni = info.input_tensor_count, no = info.output_tensor_count;
    if (!info.tensor_count || !ni || !no) {
        fprintf(stderr, "NONBLOCK probe expects a v5 named-tensor model\n");
        return 1;
    }
    size_t in_total = 0, out_total = 0;
    ornpu_io *ins = calloc(ni, sizeof(*ins)), *outs = calloc(no, sizeof(*outs));
    for (unsigned t = 0; t < info.tensor_count; t++) {
        ornpu_tensor_info ti;
        ornpu_get_tensor(model, t, &ti);
        if (ti.role == ORNPU_TENSOR_INPUT) in_total += ti.api_bytes;
        else if (ti.role == ORNPU_TENSOR_OUTPUT) out_total += ti.api_bytes;
    }
    uint8_t *input = malloc(in_total);
    int8_t *output = malloc(out_total), *reference = malloc(out_total), *expected = NULL;
    double *times = malloc((iterations ? iterations : 1) * sizeof(double));
    if (!ins || !outs || !input || !output || !reference || !times) return 1;
    for (unsigned t = 0; t < info.tensor_count; t++) {
        ornpu_tensor_info ti;
        ornpu_get_tensor(model, t, &ti);
        if (ti.role == ORNPU_TENSOR_INPUT && ti.index < ni)
            ins[ti.index] = (ornpu_io){t, input + ti.api_offset, ti.api_bytes};
        else if (ti.role == ORNPU_TENSOR_OUTPUT && ti.index < no)
            outs[ti.index] = (ornpu_io){t, output + ti.api_offset, ti.api_bytes};
    }
    if (read_file(argv[2], input, in_total) != in_total) {
        fprintf(stderr, "input read failed\n");
        return 1;
    }
    if (argc == 5) {
        expected = malloc(out_total);
        if (!expected || read_file(argv[4], expected, out_total) != out_total) {
            fprintf(stderr, "expected read failed\n");
            return 1;
        }
    }
    /* Warm up (also packs the input into the arena for the raw submits below). */
    if (ornpu_run_io(model, ins, ni, outs, no)) { fprintf(stderr, "warmup failed\n"); return 1; }
    memcpy(reference, output, out_total);

    unsigned runs = 0;
    for (unsigned i = 0; i < iterations; i++) {
        double start = now_us();
        if (ornpu_run_io(model, ins, ni, outs, no)) { fprintf(stderr, "sync run failed\n"); return 1; }
        times[runs++] = now_us() - start;
    }
    double *sorted = malloc(runs * sizeof(double));
    memcpy(sorted, times, runs * sizeof(double));
    qsort(sorted, runs, sizeof(double), compare_double);
    printf("sync: runs=%u tasks=%u %s median=%.1fus\n", runs, info.task_count,
           info.submission_serial ? "serial" : "batched", sorted[runs / 2]);

    /* NONBLOCK: does SUBMIT return before completion, and does a second job wait? */
    double start = now_us();
    int rc = ornpu_submit_flags(model, JOB_NONBLOCK, NULL);
    double queued = now_us() - start;
    double second_start = now_us();
    int rc2 = ornpu_submit_flags(model, 0, NULL);
    double drained = now_us() - second_start;
    if (ornpu_sync_outputs(model)) { fprintf(stderr, "sync outputs failed\n"); return 1; }
    int async_ok = memcmp(output, reference, out_total) == 0;
    printf("nonblock: rc=%d return=%.1fus second_rc=%d wait=%.1fus output_matches=%s\n",
           rc, queued, rc2, drained, async_ok ? "yes" : "no");
    if (expected) {
        printf("  expected_match=%s\n", memcmp(output, expected, out_total) == 0 ? "yes" : "no");
    }

    /* Paired, interleaved measurement: one round is two inferences (A then B) run
     * synchronously, the next round queues A without blocking while B runs and is
     * drained by B's blocking submit. Interleaving makes board-load drift hit both
     * modes equally, so the ratio of medians is meaningful. */
    /* Two-instance pipeline: instance A is queued without blocking while instance B
     * runs and is drained by B's blocking submission (jobs run in order, so B's
     * completion implies A's). A's outputs are then verified - a real cross-job
     * pipeline without a completion fence. */
    ornpu_model *second = NULL;
    ornpu_info info2;
    int pipe_ok = 0;
    if (ornpu_open(argv[1], &second) == 0 && ornpu_get_info(second, &info2) == 0
        && info2.input_bytes == info.input_bytes && info2.output_bytes == info.output_bytes) {
        uint8_t *input2 = malloc(in_total);
        int8_t *output2 = malloc(out_total);
        ornpu_io *ins2 = calloc(ni, sizeof(*ins2)), *outs2 = calloc(no, sizeof(*outs2));
        memcpy(input2, input, in_total);
        for (unsigned t = 0; t < info2.tensor_count; t++) {
            ornpu_tensor_info ti;
            ornpu_get_tensor(second, t, &ti);
            if (ti.role == ORNPU_TENSOR_INPUT && ti.index < ni)
                ins2[ti.index] = (ornpu_io){t, input2 + ti.api_offset, ti.api_bytes};
            else if (ti.role == ORNPU_TENSOR_OUTPUT && ti.index < no)
                outs2[ti.index] = (ornpu_io){t, output2 + ti.api_offset, ti.api_bytes};
        }
        unsigned rounds = iterations ? iterations : 16;
        double *sync_pair = malloc(rounds * sizeof(double));
        double *pipe_pair = malloc(rounds * sizeof(double));
        int ok = 1;
        for (unsigned i = 0; i < rounds; i++) {
            double r0 = now_us();
            if (ornpu_run_io(model, ins, ni, outs, no)) { ok = 0; break; }
            if (ornpu_run_io(second, ins2, ni, outs2, no)) { ok = 0; break; }
            sync_pair[i] = now_us() - r0;
            r0 = now_us();
            if (ornpu_submit_flags(model, JOB_NONBLOCK, NULL)) { ok = 0; break; }
            if (ornpu_run_io(second, ins2, ni, outs2, no)) { ok = 0; break; }
            if (ornpu_sync_outputs(model)) { ok = 0; break; }
            pipe_pair[i] = now_us() - r0;
            if (expected && memcmp(output, expected, out_total)) ok = 0;
        }
        double pair_sync = 0, pair_pipe = 0;
        if (ok) {
            qsort(sync_pair, rounds, sizeof(double), compare_double);
            qsort(pipe_pair, rounds, sizeof(double), compare_double);
            pair_sync = sync_pair[rounds / 2];
            pair_pipe = pipe_pair[rounds / 2];
        }
        pipe_ok = ok && memcmp(output2, reference, out_total) == 0;
        printf("two-instance pipeline: rounds=%u median sync=%.0fus pipelined=%.0fus "
               "ratio=%.3f outputs_ok=%s\n", rounds, pair_sync, pair_pipe,
               pair_pipe > 0 ? pair_sync / pair_pipe : 0.0, pipe_ok ? "yes" : "no");
        free(ins2);
        free(outs2);
        free(output2);
        free(input2);
        ornpu_close(second);
    }

    /* FENCE_OUT: a pollable completion fd. */
    int fence = -1;
    start = now_us();
    rc = ornpu_submit_flags(model, JOB_NONBLOCK | JOB_FENCE_OUT, &fence);
    double submit_us = now_us() - start;
    printf("fence_out: rc=%d fd=%d submit=%.1fus\n", rc, fence, submit_us);
    if (rc == 0 && fence >= 0) {
        start = now_us();
        int waited = ornpu_wait_fence(fence, 1000);
        printf("  wait_fence=%d after=%.1fus\n", waited, now_us() - start);
        /* FENCE_IN: the next job waits on the same fd. */
        int next = fence;
        rc2 = ornpu_submit_flags(model, JOB_NONBLOCK | JOB_FENCE_IN | JOB_FENCE_OUT, &next);
        printf("fence_in: rc=%d out_fd=%d\n", rc2, next);
        if (rc2 == 0 && next >= 0) {
            waited = ornpu_wait_fence(next, 1000);
            printf("  wait_fence=%d\n", waited);
            close(next);
        }
        close(fence);
    }
    free(sorted);
    free(times);
    free(expected);
    free(reference);
    free(output);
    free(input);
    free(outs);
    free(ins);
    ornpu_close(model);
    return 0;
}
