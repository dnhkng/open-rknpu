/* SPDX-License-Identifier: MIT
 *
 * Board harness for examples/depthwise_separable.
 *
 * Runs the compiled depthwise-separable classifier prefix and compares every
 * output byte with the integer reference `expected.i8` produced by build.py.
 * The fixtures are consumed in lockstep, one case per inference:
 *
 *   inputs.u8    packed NHWC UINT8 cases, back to back
 *   expected.i8  packed NHWC INT8 outputs, one per input case
 *
 * usage: board prefix.bin inputs.u8 expected.i8
 *
 * The output is one machine-readable line per case and a final SUMMARY line, so
 * a board run can be pasted into the README table. Nothing is written to disk.
 * Only libc and the open_rknpu runtime are used; the container is opened once
 * and every case reuses the same model instance. Timing is a wall-clock
 * CLOCK_MONOTONIC span per inference, including input packing and output
 * readback but excluding model loading and file I/O.
 *
 * Cross-compile (the exact line is in README.md under "Board"):
 *
 *   <vendor>-gcc --sysroot=... -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime \
 *     examples/depthwise_separable/board.c runtime/open_rknpu.c -o board
 */
#define _POSIX_C_SOURCE 200809L
#include "open_rknpu.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double milliseconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec * 1000.0 + (double)now.tv_nsec / 1.0e6;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s prefix.bin inputs.u8 expected.i8\n", argv[0]);
        return 2;
    }
    ornpu_info info;
    int rc = ornpu_inspect(argv[1], &info);
    if (rc) {
        fprintf(stderr, "inspect %s: %d\n", argv[1], rc);
        return 1;
    }
    if (!info.input_bytes || !info.output_bytes) {
        fprintf(stderr, "prefix has an empty input or output buffer\n");
        return 1;
    }
    ornpu_model *model = NULL;
    rc = ornpu_open(argv[1], &model);
    if (rc) {
        fprintf(stderr, "open %s: %d\n", argv[1], rc);
        return 1;
    }
    FILE *inputs = fopen(argv[2], "rb");
    FILE *expected = fopen(argv[3], "rb");
    uint8_t *input = malloc(info.input_bytes);
    int8_t *gold = malloc(info.output_bytes);
    int8_t *output = malloc(info.output_bytes);
    if (!inputs || !expected || !input || !gold || !output) {
        fprintf(stderr, "cannot read the fixtures or allocate the buffers\n");
        if (inputs) fclose(inputs);
        if (expected) fclose(expected);
        free(input);
        free(gold);
        free(output);
        ornpu_close(model);
        return 1;
    }
    printf("container %s: %ux%ux%u -> %ux%ux%u, %u input bytes, %u output bytes, %u tasks\n",
           argv[1], (unsigned)info.height, (unsigned)info.width, (unsigned)info.input_channels,
           (unsigned)info.output_height, (unsigned)info.output_width,
           (unsigned)info.output_channels, (unsigned)info.input_bytes,
           (unsigned)info.output_bytes, (unsigned)info.task_count);
    unsigned cases = 0, failures = 0;
    unsigned long long exact = 0;
    double total = 0.0, worst = 0.0;
    for (;;) {
        size_t got_input = fread(input, 1, info.input_bytes, inputs);
        if (!got_input && feof(inputs)) break;
        size_t got_gold = fread(gold, 1, info.output_bytes, expected);
        if (got_input != info.input_bytes || got_gold != info.output_bytes) {
            fprintf(stderr, "case %u: truncated fixtures\n", cases);
            rc = -EINVAL;
            break;
        }
        double start = milliseconds();
        rc = ornpu_run(model, input, info.input_bytes, output, info.output_bytes);
        double elapsed = milliseconds() - start;
        if (rc) {
            fprintf(stderr, "case %u: ornpu_run returned %d\n", cases, rc);
            break;
        }
        size_t mismatches = 0;
        for (size_t i = 0; i < info.output_bytes; i++)
            if (output[i] != gold[i]) mismatches++;
        exact += info.output_bytes - mismatches;
        if (mismatches) failures++;
        total += elapsed;
        if (elapsed > worst) worst = elapsed;
        printf("case %u bytes=%u mismatches=%zu ms=%.3f\n", cases, (unsigned)info.output_bytes,
               mismatches, elapsed);
        fflush(stdout);
        cases++;
    }
    if (!rc && (ferror(inputs) || ferror(expected) || fgetc(inputs) != EOF
                || fgetc(expected) != EOF)) {
        fprintf(stderr, "the fixture files disagree on the case count\n");
        rc = -EINVAL;
    }
    fclose(inputs);
    fclose(expected);
    free(input);
    free(gold);
    free(output);
    ornpu_close(model);
    printf("SUMMARY prefix=%s cases=%u inferences=%u exact_bytes=%llu mismatches=%u "
           "total_ms=%.3f ms_per_inference=%.4f worst_ms=%.3f result=%s\n",
           argv[1], cases, cases, exact, failures, total,
           cases ? total / (double)cases : 0.0, worst, (rc || failures || !cases) ? "FAIL" : "PASS");
    return (rc || failures || !cases) ? 1 : 0;
}
