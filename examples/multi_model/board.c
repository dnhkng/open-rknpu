/* SPDX-License-Identifier: MIT
 *
 * E13 board harness: two containers, two open handles, one process.
 *
 * usage: board_multi <model_a.bin> <model_b.bin> \
 *                    <input_a.u8> <input_b.u8> <expected_a.i8> <expected_b.i8> \
 *                    <alternations>
 *
 * ornpu_open is called exactly once per model and ornpu_close exactly once per
 * model at the end: each handle keeps its own dma arena and its own copy of the
 * packed weights for the whole alternation. That is the supported pattern the
 * example teaches; rebuilding the handles per inference would re-create the dma
 * allocations and re-upload the weights every time, and a missing ornpu_close
 * would leak them.
 *
 * Each input file holds `cases` packed NHWC UINT8 inputs back to back and each
 * expected file holds one packed INT8 output per case; alternation i uses case
 * i % cases. Every inference prints one machine-readable line
 *
 *   model=A bytes=4096 exact=1
 *
 * and the run ends with the three numbers the README's results table needs
 * (inferences, exact bytes, ms per inference) per model and in total:
 *
 *   summary model=A inferences=4 exact_bytes=16384 ms_per_inference=1.234
 *   summary total inferences=8 exact_bytes=18432 ms_per_inference=2.221
 *
 * It never opens the runtime for any model twice, so a successful run also
 * proves the two independent arenas coexist on one board session.
 */
#include "open_rknpu.h"
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MODELS 2

struct fixture {
    const char *name;      /* "A" or "B", the label the output lines carry */
    const char *model_path;
    const char *input_path;
    const char *expected_path;
    ornpu_model *model;
    uint8_t *inputs;       /* cases * input_case bytes */
    int8_t *expected;      /* cases * output_case bytes */
    int8_t *output;        /* one packed output */
    size_t input_case;
    size_t output_case;
    unsigned cases;
    unsigned inferences;
    unsigned mismatches;
    double milliseconds;
};

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

/* Read a whole file into a fresh buffer; 0 on success, -errno on failure. */
static int read_file(const char *path, void **data, size_t *size) {
    FILE *file = fopen(path, "rb");
    if (!file) return -errno;
    if (fseek(file, 0, SEEK_END)) { fclose(file); return -EIO; }
    long end = ftell(file);
    if (end < 0) { fclose(file); return -EIO; }
    if (fseek(file, 0, SEEK_SET)) { fclose(file); return -EIO; }
    *size = (size_t)end;
    uint8_t *buffer = malloc(*size ? *size : 1);
    if (!buffer) { fclose(file); return -ENOMEM; }
    if (*size && fread(buffer, 1, *size, file) != *size) {
        free(buffer);
        fclose(file);
        return -EIO;
    }
    fclose(file);
    *data = buffer;
    return 0;
}

/* Release every device and heap resource one fixture owns (safe on a partial open). */
static void release(struct fixture *f) {
    if (f->model) ornpu_close(f->model);
    free(f->inputs);
    free(f->expected);
    free(f->output);
    f->model = NULL;
    f->inputs = NULL;
    f->expected = NULL;
    f->output = NULL;
}

/* Open one v5 container, size its single external input and output, load its fixtures. */
static int prepare(struct fixture *f, unsigned alternations) {
    ornpu_info info;
    ornpu_tensor_info tensor;
    void *input_data = NULL, *expected_data = NULL;
    size_t input_size = 0, expected_size = 0;
    int rc = ornpu_open(f->model_path, &f->model);
    if (rc) { fprintf(stderr, "%s: open %s: %d\n", f->name, f->model_path, rc); return rc; }
    if ((rc = ornpu_get_info(f->model, &info))) goto fail;
    if (!info.tensor_count || info.input_tensor_count != 1 || info.output_tensor_count != 1) {
        fprintf(stderr, "%s: expected one named input and output\n", f->name);
        rc = -EINVAL;
        goto fail;
    }
    if ((rc = ornpu_get_input_tensor(f->model, 0, &tensor))) goto fail;
    f->input_case = tensor.api_bytes;
    for (unsigned t = 0; t < info.tensor_count && !f->output_case; t++) {
        if (ornpu_get_tensor(f->model, t, &tensor)) { rc = -EINVAL; goto fail; }
        if (tensor.role == ORNPU_TENSOR_OUTPUT) f->output_case = tensor.api_bytes;
    }
    if (!f->input_case || !f->output_case) { rc = -EINVAL; goto fail; }

    if ((rc = read_file(f->input_path, &input_data, &input_size))) goto fail;
    if ((rc = read_file(f->expected_path, &expected_data, &expected_size))) goto fail;
    if (input_size % f->input_case || expected_size % f->output_case ||
        input_size / f->input_case != expected_size / f->output_case ||
        input_size < f->input_case) {
        fprintf(stderr, "%s: fixture sizes do not match the container\n", f->name);
        rc = -EINVAL;
        goto fail;
    }
    f->inputs = input_data;
    f->expected = expected_data;
    input_data = expected_data = NULL;
    f->cases = (unsigned)(input_size / f->input_case);
    f->output = malloc(f->output_case);
    if (!f->output) { rc = -ENOMEM; goto fail; }
    if (alternations % f->cases) {
        fprintf(stderr, "%s: note: %u alternations cycle %u cases (the last one repeats)\n",
                f->name, alternations, f->cases);
    }
    printf("model=%s container opened: %u tasks, input %zu B, output %zu B, %u cases\n",
           f->name, info.task_count, f->input_case, f->output_case, f->cases);
    return 0;
fail:
    free(input_data);
    free(expected_data);
    release(f);
    return rc;
}

/* One inference on the already-open handle: run, compare, print the run line. */
static void infer(struct fixture *f, unsigned run) {
    unsigned index = run % f->cases;
    double started = now_ms();
    int rc = ornpu_run(f->model, f->inputs + (size_t)index * f->input_case, f->input_case,
                       f->output, f->output_case);
    f->milliseconds += now_ms() - started;
    if (rc) {
        printf("model=%s bytes=%zu exact=0 error=%d\n", f->name, f->output_case, rc);
        f->mismatches++;
        return;
    }
    const int8_t *expected = f->expected + (size_t)index * f->output_case;
    unsigned exact = memcmp(f->output, expected, f->output_case) == 0;
    if (!exact) f->mismatches++;
    printf("model=%s bytes=%zu exact=%u\n", f->name, f->output_case, exact);
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc != 8) {
        fprintf(stderr, "usage: %s <model_a.bin> <model_b.bin> <input_a.u8> <input_b.u8> "
                        "<expected_a.i8> <expected_b.i8> <alternations>\n", argv[0]);
        return 2;
    }
    unsigned alternations = (unsigned)strtoul(argv[7], NULL, 10);
    if (!alternations || alternations > 1000000) return 2;
    struct fixture fixtures[MODELS] = {
        {.name = "A", .model_path = argv[1], .input_path = argv[3], .expected_path = argv[5]},
        {.name = "B", .model_path = argv[2], .input_path = argv[4], .expected_path = argv[6]},
    };
    for (unsigned i = 0; i < MODELS; i++) {
        if (prepare(&fixtures[i], alternations)) {
            for (unsigned j = 0; j <= i; j++) release(&fixtures[j]);
            return 1;
        }
    }
    for (unsigned run = 0; run < alternations; run++) {
        struct fixture *f = &fixtures[run % MODELS];
        infer(f, run / MODELS);
        f->inferences++;
    }
    unsigned failed = 0, total_inferences = 0, total_mismatches = 0;
    size_t total_exact_bytes = 0;
    double total_milliseconds = 0.0;
    for (unsigned i = 0; i < MODELS; i++) {
        struct fixture *f = &fixtures[i];
        size_t exact_bytes = (size_t)(f->inferences - f->mismatches) * f->output_case;
        printf("summary model=%s inferences=%u exact_bytes=%zu ms_per_inference=%.3f\n",
               f->name, f->inferences, exact_bytes,
               f->inferences ? f->milliseconds / f->inferences : 0.0);
        failed += f->mismatches != 0;
        total_inferences += f->inferences;
        total_mismatches += f->mismatches;
        total_exact_bytes += exact_bytes;
        total_milliseconds += f->milliseconds;
        release(f);
    }
    printf("summary total inferences=%u exact_bytes=%zu ms_per_inference=%.3f\n",
           total_inferences, total_exact_bytes,
           total_inferences ? total_milliseconds / total_inferences : 0.0);
    if (failed || total_mismatches)
        printf("FAIL: %u/%u inferences mismatched (one open per model, no reloads)\n",
               total_mismatches, total_inferences);
    else
        printf("PASS: %u models, %u inferences, %zu exact bytes, one open per model\n",
               MODELS, total_inferences, total_exact_bytes);
    return failed ? 1 : 0;
}
