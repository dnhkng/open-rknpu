/* SPDX-License-Identifier: MIT
 * S8 job-field probe: are core_mask / subcore_task[] inert on this SoC?
 *
 * The RV1106 driver config has one IRQ; `rknpu_job_alloc` therefore forces
 * `core_mask = RKNPU_CORE0_MASK` and `rknpu_job_subcore_commit_pc` reads
 * `subcore_task[]` only when `num_irqs > 1`. This harness submits the same verified
 * container with several mask/window sets and checks that the output bytes are
 * unchanged and still match the expected file.
 *
 * usage: board_core <model.bin> <input.u8> <iterations> [expected.i8]
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

static int run_case(ornpu_model *model, const uint8_t *input, size_t in_size,
                    int8_t *output, size_t out_size, const int8_t *reference,
                    unsigned iterations, unsigned *mismatches, double *median_us) {
    double *times = malloc(iterations * sizeof(double));
    if (!times) return -1;
    unsigned bad = 0, runs = 0;
    for (unsigned i = 0; i < iterations; i++) {
        double start = now_us();
        int rc = ornpu_run(model, input, in_size, output, out_size);
        times[runs++] = now_us() - start;
        if (rc) { free(times); return rc; }
        if (reference && memcmp(reference, output, out_size)) bad++;
    }
    double *sorted = malloc(runs * sizeof(double));
    memcpy(sorted, times, runs * sizeof(double));
    qsort(sorted, runs, sizeof(double), compare_double);
    *mismatches = bad;
    *median_us = sorted[runs / 2];
    free(sorted);
    free(times);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "usage: %s <model.bin> <input.u8> <iterations> [expected.i8]\n", argv[0]);
        return 2;
    }
    const char *expected_path = argc == 5 ? argv[4] : NULL;
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
    int8_t *output = malloc(out_size), *baseline = malloc(out_size);
    int8_t *expected = expected_path ? malloc(out_size) : NULL;
    if (!input || !output || !baseline ||
        read_file(argv[2], input, in_size) != in_size ||
        (expected_path && read_file(expected_path, expected, out_size) != out_size)) {
        fprintf(stderr, "input/expected read failed\n");
        return 1;
    }
    unsigned mismatches = 0;
    double median = 0;
    if (run_case(model, input, in_size, output, out_size, expected, iterations,
                 &mismatches, &median)) {
        fprintf(stderr, "baseline run failed\n");
        return 1;
    }
    memcpy(baseline, output, out_size);
    printf("core=0x0 task=%u windows=0 runs=%u mismatches=%d changed=0 median=%.1fus\n",
           info.task_count, iterations, (int)mismatches, median);
    const uint32_t windows[10] = {0, 1, 1, 1, 2, 0, 3, 0, 4, 0};
    struct { const char *name; uint32_t core; uint32_t pairs; } cases[] = {
        {"core=0x7", 0x7u, 0},
        {"core=0x7+windows", 0x7u, 5},
        {"core=0x0+windows", 0x0u, 5},
        {"core=0x1+window1", 0x1u, 1},
    };
    int failed = mismatches != 0;
    for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        if (ornpu_set_submit_core(model, cases[c].core, windows, cases[c].pairs)) {
            fprintf(stderr, "set core failed\n");
            return 1;
        }
        if (run_case(model, input, in_size, output, out_size, expected, iterations,
                     &mismatches, &median)) {
            fprintf(stderr, "%s run failed\n", cases[c].name);
            return 1;
        }
        int changed = memcmp(baseline, output, out_size) != 0;
        printf("%s task=%u runs=%u mismatches=%d changed=%d median=%.1fus\n",
               cases[c].name, info.task_count, iterations, (int)mismatches, changed, median);
        failed += mismatches != 0 || changed;
    }
    ornpu_close(model);
    return failed ? 1 : 0;
}
