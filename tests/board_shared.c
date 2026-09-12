/* SPDX-License-Identifier: MIT
 * Zero-copy input from a dma-buf (checklist F8).
 *
 * The NPU driver imports an existing dma-buf when CREATE's `handle` is the fd and bit 7 of
 * `flags` is set (`research/probe_dmabuf.c`). `ornpu_open_shared` uses that to allocate the
 * model's arena from the Rockchip CMA heap and hand the caller the fd; this harness plays
 * the part of a producer: it maps the fd, writes the packed input bytes at the offset the
 * runtime reports, and runs the model without the runtime copying anything in.
 *
 * The proof is byte equality with the suite's recorded `expectedNNN.i8`:
 *
 *   usage: board_shared <model.bin> <input.u8> <expected.i8> [runs]
 *   arena fd=4 offset=8192 bytes=192
 *   run 0 exact=192
 *   SUMMARY runs=2 arena_fd=4 input_offset=8192 input_bytes=192 exact_bytes=384 mismatches=0 result=PASS
 */
#include "open_rknpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

static size_t read_file(const char *path, void *data, size_t size) {
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
    unsigned runs = argc == 5 ? (unsigned)strtoul(argv[4], NULL, 10) : 2;
    if (!runs || runs > 1000) return 2;

    ornpu_model *model = NULL;
    int arena_fd = -1;
    int rc = ornpu_open_shared(argv[1], &model, &arena_fd);
    if (rc) { fprintf(stderr, "open_shared failed: %d\n", rc); return 1; }
    ornpu_info info;
    if (ornpu_get_info(model, &info)) { fprintf(stderr, "get_info failed\n"); ornpu_close(model); return 1; }
    struct ornpu_input_view view;
    rc = ornpu_input_view(model, 0, &view);
    if (rc) { fprintf(stderr, "input_view failed: %d (is the input layout packed?)\n", rc); ornpu_close(model); return 1; }
    uint64_t offset = view.offset;
    size_t bytes = (size_t)view.batch * view.height * view.width * view.channels;
    if (bytes != info.input_bytes) {
        fprintf(stderr, "the view covers %zu API bytes, info says %u\n", bytes, info.input_bytes);
        ornpu_close(model); return 1;
    }
    fprintf(stderr, "arena fd=%d offset=%llu row_stride=%u %ux%ux%ux%u\n", arena_fd,
            (unsigned long long)offset, view.row_stride, view.batch, view.height, view.width,
            view.channels);

    /* The producer maps the same dma-buf the NPU reads. */
    unsigned char *arena = mmap(NULL, info.arena_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, arena_fd, 0);
    if (arena == MAP_FAILED) { fprintf(stderr, "mmap of the shared arena failed\n"); ornpu_close(model); return 1; }
    unsigned char *input = malloc(bytes ? bytes : 1);
    signed char *output = malloc(info.output_bytes ? info.output_bytes : 1);
    signed char *expected = malloc(info.output_bytes ? info.output_bytes : 1);
    if (!input || !output || !expected) { ornpu_close(model); return 1; }
    if (read_file(argv[2], input, bytes) != bytes) {
        fprintf(stderr, "input must be exactly %zu bytes\n", bytes);
        ornpu_close(model); return 1;
    }
    int have_expected = read_file(argv[3], expected, info.output_bytes) == info.output_bytes;

    unsigned long long exact = 0, mismatches = 0;
    for (unsigned i = 0; i < runs; i++) {
        /* The "producer" writes straight into the engine's memory, row by row in the
         * arena's layout: no ornpu_run copy. A V4L2/ISP producer with the same stride
         * would write the rows itself; the runtime fills only the stride padding. */
        for (uint32_t b = 0; b < view.batch; b++)
            for (uint32_t row = 0; row < view.height; row++)
                memcpy(arena + offset + ((uint64_t)b * view.height + row) * view.row_stride * view.channels,
                       input + ((uint64_t)b * view.height + row) * view.width * view.channels,
                       (size_t)view.width * view.channels);
        rc = ornpu_run_prefilled(model, output, info.output_bytes);
        if (rc) { fprintf(stderr, "run %u failed: %d\n", i, rc); ornpu_close(model); return 1; }
        if (have_expected) {
            if (memcmp(output, expected, info.output_bytes) == 0) exact += info.output_bytes;
            else {
                mismatches++;
                size_t at = 0;
                while (at < info.output_bytes && output[at] == expected[at]) at++;
                printf("mismatch run %u first_diff=%zu expected=%d got=%d\n",
                       i, at, at < info.output_bytes ? expected[at] : 0,
                       at < info.output_bytes ? output[at] : 0);
            }
        }
        printf("run %u exact=%u\n", i, info.output_bytes);
    }
    printf("SUMMARY runs=%u arena_fd=%d input_offset=%llu input_bytes=%zu exact_bytes=%llu "
           "mismatches=%llu result=%s\n", runs, arena_fd, (unsigned long long)offset, bytes,
           exact, mismatches, mismatches == 0 ? "PASS" : "FAIL");
    int status = mismatches == 0 ? 0 : 1;
    free(input); free(output); free(expected);
    munmap(arena, info.arena_bytes);
    ornpu_close(model);
    return status;
}
