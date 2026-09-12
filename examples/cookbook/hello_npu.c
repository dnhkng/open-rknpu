/* SPDX-License-Identifier: MIT
 *
 * The smallest open-rknpu program: inspect a compiled container, open it, run
 * one packed input through the NPU, print the output codes and close it.
 *
 * A host build only compiles: the device calls are behind `ORNPU_BOARD`, so
 * `gcc -O2 -Wall -Wextra -Werror -Iruntime -c hello_npu.c` succeeds without the
 * vendor toolchain. Running the model needs the RV1103 board and the libc-only
 * runtime linked in:
 *
 *   research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
 *     --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
 *     -O2 -std=gnu99 -Wall -Wextra -Werror -DORNPU_BOARD -Iruntime \
 *     examples/cookbook/hello_npu.c runtime/open_rknpu.c -o hello_npu
 *
 * usage: hello_npu <container.bin> [input.u8]
 * The input file is the packed NHWC UINT8 buffer `info.input_bytes` long; when
 * it is omitted the buffer is zero-filled.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "open_rknpu.h"

#define HELLO_PRINTED_CODES 16

#ifdef ORNPU_BOARD

static int read_exact(const char *path, void *data, size_t size) {
    FILE *file = fopen(path, "rb");
    if (!file) return -1;
    size_t got = fread(data, 1, size, file);
    int extra = fgetc(file);
    fclose(file);
    return (got == size && extra == EOF) ? 0 : -1;
}

static int hello(const char *model_path, const char *input_path) {
    ornpu_info info;
    int rc = ornpu_inspect(model_path, &info);       /* header only: no device access */
    if (rc) {
        fprintf(stderr, "ornpu_inspect(%s) failed: %d\n", model_path, rc);
        return 1;
    }
    printf("container: %ux%u input, %ux%u output, %u input bytes, %u output bytes\n",
           (unsigned)info.height, (unsigned)info.width,
           (unsigned)info.output_height, (unsigned)info.output_width,
           (unsigned)info.input_bytes, (unsigned)info.output_bytes);
    printf("task_count=%u serial=%u profile tasks run as %s\n",
           (unsigned)info.task_count, (unsigned)info.submission_serial,
           info.submission_serial ? "separate submissions" : "linked job(s)");

    ornpu_model *model = NULL;
    rc = ornpu_open(model_path, &model);
    if (rc) {
        fprintf(stderr, "ornpu_open failed: %d\n", rc);
        return 1;
    }
    uint8_t *input = (uint8_t *)calloc(1, info.input_bytes ? info.input_bytes : 1);
    int8_t *output = (int8_t *)malloc(info.output_bytes ? info.output_bytes : 1);
    if (!input || !output) {
        fprintf(stderr, "out of memory\n");
        free(input);
        free(output);
        ornpu_close(model);
        return 1;
    }
    if (input_path && read_exact(input_path, input, info.input_bytes)) {
        fprintf(stderr, "input %s must contain exactly %u bytes\n", input_path, (unsigned)info.input_bytes);
        free(input);
        free(output);
        ornpu_close(model);
        return 1;
    }
    rc = ornpu_run(model, input, info.input_bytes, output, info.output_bytes);
    if (rc) {
        fprintf(stderr, "ornpu_run failed: %d\n", rc);
        free(input);
        free(output);
        ornpu_close(model);
        return 1;
    }
    unsigned shown = info.output_bytes < HELLO_PRINTED_CODES ? info.output_bytes : HELLO_PRINTED_CODES;
    printf("output codes [0..%u):", shown);
    for (unsigned i = 0; i < shown; i++) printf(" %d", (int)output[i]);
    printf("\n");

    free(input);
    free(output);
    ornpu_close(model);
    return 0;
}

#else /* !ORNPU_BOARD */

static int hello(const char *model_path, const char *input_path) {
    (void)model_path;
    (void)input_path;
    fprintf(stderr,
            "hello_npu: this is a host build, so the device calls were compiled out.\n"
            "Rebuild with -DORNPU_BOARD and link runtime/open_rknpu.c to run it on the board:\n"
            "  <arm-linux-uclibcgnueabihf-gcc> -O2 -std=gnu99 -DORNPU_BOARD -Iruntime \\\n"
            "    examples/cookbook/hello_npu.c runtime/open_rknpu.c -o hello_npu\n");
    return 0;
}

#endif /* ORNPU_BOARD */

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s <container.bin> [input.u8]\n", argv[0]);
        return 2;
    }
    return hello(argv[1], argc == 3 ? argv[2] : NULL);
}
