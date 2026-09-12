/* SPDX-License-Identifier: MIT
 * Timing harness for `ornpu_run_timed` (checklist F9).
 *
 * The board has no userspace cycle counter and the driver exposes none, so the supported
 * measurement is the runtime's own wall-clock breakdown of one inference: pack (marshalling
 * the caller's input into the mapped arena), submit (the SUBMIT ioctl, where the engine runs
 * and waits), readback (cache sync plus unpacking the output), and total.
 *
 * Output is one machine-readable line per run and a summary, so the numbers can be pasted
 * into docs/performance.md:
 *
 *   run 3 pack=1200 submit=26000 readback=900 total=28400
 *   SUMMARY runs=16 min_submit_ns=... median_submit_ns=... mean_submit_ns=...
 *           min_total_ns=... exact_bytes=... mismatches=0 result=PASS
 *
 * usage: board_timed <model.bin> <input.u8> <expected.i8> [runs]
 */
#include "open_rknpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int compare_u64(const void *a, const void *b) {
    unsigned long long x = *(const unsigned long long *)a, y = *(const unsigned long long *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static size_t read_file(const char *path, unsigned char *data, size_t size) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    size_t got = fread(data, 1, size, file);
    fclose(file);
    return got;
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "usage: %s <model.bin> <input.u8> <expected.i8> [runs]\n", argv[0]);
        return 2;
    }
    unsigned runs = argc == 5 ? (unsigned)strtoul(argv[4], NULL, 10) : 16;
    if (!runs || runs > 1000000) return 2;

    ornpu_model *model = NULL;
    ornpu_info info;
    if (ornpu_open(argv[1], &model) || ornpu_get_info(model, &info)) {
        fprintf(stderr, "open failed\n");
        return 1;
    }
    if (info.tensor_count && (info.input_tensor_count != 1 || info.output_tensor_count != 1)) {
        fprintf(stderr, "board_timed needs exactly one external input and output\n");
        ornpu_close(model);
        return 1;
    }
    unsigned char *input = malloc(info.input_bytes ? info.input_bytes : 1);
    signed char *output = malloc(info.output_bytes ? info.output_bytes : 1);
    signed char *expected = malloc(info.output_bytes ? info.output_bytes : 1);
    if (!input || !output || !expected) { ornpu_close(model); return 1; }
    if (read_file(argv[2], input, info.input_bytes) != info.input_bytes) {
        fprintf(stderr, "input must be exactly %u bytes\n", info.input_bytes);
        ornpu_close(model);
        return 1;
    }
    int have_expected = read_file(argv[3], (unsigned char *)expected, info.output_bytes) == info.output_bytes;

    /* One untimed invocation warms the adb/open path and the caches; the driver's first
     * submission after open is always the slowest (docs/performance.md). */
    if (ornpu_run_timed(model, input, info.input_bytes, output, info.output_bytes, NULL)) {
        fprintf(stderr, "warm-up run failed\n");
        ornpu_close(model);
        return 1;
    }

    unsigned long long *submit = malloc(sizeof(*submit) * runs);
    unsigned long long *total = malloc(sizeof(*total) * runs);
    if (!submit || !total) { ornpu_close(model); return 1; }
    unsigned long long mismatches = 0, exact = 0;
    for (unsigned i = 0; i < runs; i++) {
        struct ornpu_timing timing;
        memset(&timing, 0, sizeof(timing));
        if (ornpu_run_timed(model, input, info.input_bytes, output, info.output_bytes, &timing)) {
            fprintf(stderr, "run %u failed\n", i);
            ornpu_close(model);
            return 1;
        }
        submit[i] = timing.submit_ns;
        total[i] = timing.total_ns;
        if (have_expected) {
            if (memcmp(output, expected, info.output_bytes) == 0) exact += info.output_bytes;
            else mismatches++;
        }
        printf("run %u pack=%llu submit=%llu readback=%llu total=%llu\n", i,
               (unsigned long long)timing.pack_ns, (unsigned long long)timing.submit_ns,
               (unsigned long long)timing.readback_ns, (unsigned long long)timing.total_ns);
    }
    qsort(submit, runs, sizeof(*submit), compare_u64);
    qsort(total, runs, sizeof(*total), compare_u64);
    unsigned long long sum = 0;
    for (unsigned i = 0; i < runs; i++) sum += submit[i];
    printf("SUMMARY runs=%u min_submit_ns=%llu median_submit_ns=%llu mean_submit_ns=%llu "
           "min_total_ns=%llu exact_bytes=%llu mismatches=%llu result=%s\n",
           runs, submit[0], submit[runs / 2], sum / runs, total[0], exact, mismatches,
           mismatches == 0 ? "PASS" : "FAIL");
    int rc = mismatches == 0 ? 0 : 1;
    free(submit); free(total); free(input); free(output); free(expected);
    ornpu_close(model);
    return rc;
}
