/* SPDX-License-Identifier: MIT
 *
 * E5: camera/V4L2 -> NPU. A libc-only board program that captures one frame from a
 * V4L2 node (or reads a recorded frame), converts it to the model's packed NHWC UINT8
 * RGB input with exactly the rule `examples/camera/convert.py` documents, runs the
 * container and compares the output byte for byte with `--expected`.
 *
 * Modes
 * -----
 *   --frame FILE --format yuyv|nv12|rgb --width W --height H
 *       Read a recorded capture buffer, convert, run the NPU. With --expected FILE the
 *       output is compared byte for byte and the machine-readable line
 *       `SUMMARY ... exact_bytes=.. mismatches=.. result=PASS` is printed.
 *   --emit-input FILE
 *       Convert only (no device, no NPU) and write the packed NHWC bytes, so the host
 *       test can compare this program's conversion against convert.py.
 *   --device /dev/videoN --format yuyv|nv12 --width W --height H
 *       Negotiate the format, capture one frame with mmap buffers, then run the NPU.
 *       The capture node is opened *before* the NPU: a busy node (the ISP's
 *       rkisp_mainpath while rkipc streams it) then fails with its own errno instead of
 *       a misleading NPU error. When the input layout is packed and the CMA heap exists
 *       the run goes through the zero-copy path (ornpu_open_shared / ornpu_input_view /
 *       ornpu_run_prefilled); otherwise it falls back to ornpu_run's copying path.
 *
 * A frame the device never delivers is reported after a bounded wait instead of blocking
 * forever: `rkcif_scale_ch2` never produced a frame on the reference board, and a capture
 * probe of a free node must time out rather than hang.
 *
 * Cross-compile (the exact line is in README.md under "Board"):
 *
 *   research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
 *     --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
 *     -O2 -std=gnu99 -Wall -Wextra -Werror -D_GNU_SOURCE -Iruntime \
 *     examples/camera/camera_npu.c runtime/open_rknpu.c -o examples/camera/build/camera_npu
 */
#define _POSIX_C_SOURCE 200809L
#include "open_rknpu.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <unistd.h>

#define CAPTURE_TIMEOUT_MS 5000
#define MMAP_BUFFERS 2

enum format_id { FORMAT_YUYV, FORMAT_NV12, FORMAT_RGB };
enum run_path { PATH_COPY, PATH_SHARED };

/* --------------------------------------------------------------------------- */
/* Errors: every failure names the call, the path and the errno.               */
/* --------------------------------------------------------------------------- */
static const char *errno_name(int error) {
    if (error == ENOTTY) return "ENOTTY";
    if (error == EBUSY) return "EBUSY";
    if (error == ENOENT) return "ENOENT";
    if (error == EACCES) return "EACCES";
    if (error == EPERM) return "EPERM";
    if (error == EINVAL) return "EINVAL";
    if (error == ENODEV) return "ENODEV";
    if (error == EAGAIN) return "EAGAIN";
    if (error == ETIMEDOUT) return "ETIMEDOUT";
    if (error == ENOMEM) return "ENOMEM";
    if (error == ENOSPC) return "ENOSPC";
    if (error == EMFILE) return "EMFILE";
    if (error == ENOSYS) return "ENOSYS";
    if (error == ERANGE) return "ERANGE";
    return "errno";
}

/* Print one machine-readable error and return 1, so callers can `return fail(...)`. */
static int fail(const char *path, const char *what, int error) {
    fprintf(stderr, "camera_npu: %s: %s failed: %s (errno=%d %s)\n",
            path, what, strerror(error), error, errno_name(error));
    return 1;
}

/* --------------------------------------------------------------------------- */
/* Argument parsing                                                            */
/* --------------------------------------------------------------------------- */
struct options {
    const char *model;
    const char *frame;
    const char *device;
    const char *expected;
    const char *emit_input;
    enum format_id format;
    int have_format;
    unsigned width, height;
    int have_geometry;
    unsigned runs;
};

static void usage(FILE *stream) {
    fprintf(stream,
            "usage: camera_npu --model model.bin [--frame FILE --format yuyv|nv12|rgb "
            "--width W --height H]\n"
            "                   [--expected FILE] [--emit-input FILE]\n"
            "                   [--device /dev/videoN] [--runs N]\n"
            "\n"
            "  --frame FILE       read a recorded capture buffer\n"
            "  --device NODE      capture one frame from a V4L2 node (needs --format)\n"
            "  --emit-input FILE  convert only and write the packed NHWC bytes (no device)\n"
            "  --expected FILE    compare the NPU output against this INT8 reference\n"
            "  --runs N           repeat the inference N times (default 1)\n");
}

static int parse_unsigned(const char *text, unsigned *value) {
    char *end = NULL;
    unsigned long parsed = strtoul(text, &end, 10);
    if (!text[0] || !end || *end || parsed == 0 || parsed > 65535) return -1;
    *value = (unsigned)parsed;
    return 0;
}

static int parse_options(int argc, char **argv, struct options *options) {
    memset(options, 0, sizeof(*options));
    options->runs = 1;
    for (int index = 1; index < argc; index++) {
        const char *name = argv[index];
        if (!strcmp(name, "--help") || !strcmp(name, "-h")) { usage(stdout); return 1; }
        if (index + 1 >= argc) { fprintf(stderr, "camera_npu: %s needs a value\n", name); return -1; }
        const char *value = argv[++index];
        if (!strcmp(name, "--model")) options->model = value;
        else if (!strcmp(name, "--frame")) options->frame = value;
        else if (!strcmp(name, "--device")) options->device = value;
        else if (!strcmp(name, "--expected")) options->expected = value;
        else if (!strcmp(name, "--emit-input")) options->emit_input = value;
        else if (!strcmp(name, "--format")) {
            if (!strcmp(value, "yuyv")) options->format = FORMAT_YUYV;
            else if (!strcmp(value, "nv12")) options->format = FORMAT_NV12;
            else if (!strcmp(value, "rgb")) options->format = FORMAT_RGB;
            else { fprintf(stderr, "camera_npu: unknown format %s\n", value); return -1; }
            options->have_format = 1;
        } else if (!strcmp(name, "--width")) {
            if (parse_unsigned(value, &options->width)) { fprintf(stderr, "camera_npu: bad width %s\n", value); return -1; }
            options->have_geometry = 1;
        } else if (!strcmp(name, "--height")) {
            if (parse_unsigned(value, &options->height)) { fprintf(stderr, "camera_npu: bad height %s\n", value); return -1; }
            options->have_geometry = 1;
        } else if (!strcmp(name, "--runs")) {
            if (parse_unsigned(value, &options->runs)) { fprintf(stderr, "camera_npu: bad runs %s\n", value); return -1; }
        } else {
            fprintf(stderr, "camera_npu: unknown option %s\n", name);
            return -1;
        }
    }
    if (!options->model) { fprintf(stderr, "camera_npu: --model is required\n"); return -1; }
    if (options->frame && options->device) {
        fprintf(stderr, "camera_npu: --frame and --device are mutually exclusive\n");
        return -1;
    }
    if (!options->frame && !options->device) {
        fprintf(stderr, "camera_npu: one of --frame or --device is required\n");
        return -1;
    }
    if (!options->have_geometry) {
        fprintf(stderr, "camera_npu: --width and --height are required\n");
        return -1;
    }
    if (!options->have_format) {
        fprintf(stderr, "camera_npu: --format is required\n");
        return -1;
    }
    if (options->emit_input && !options->frame) {
        fprintf(stderr, "camera_npu: --emit-input needs --frame (it never touches a device)\n");
        return -1;
    }
    return 0;
}

/* --------------------------------------------------------------------------- */
/* The conversion contract (identical to examples/camera/convert.py)           */
/* --------------------------------------------------------------------------- */
static size_t frame_bytes(enum format_id format, unsigned width, unsigned height) {
    if (format == FORMAT_YUYV) return (size_t)width * height * 2u;
    if (format == FORMAT_NV12) return (size_t)width * height + 2u * (width / 2u) * (height / 2u);
    return (size_t)width * height * 3u;
}

static size_t expected_frame_bytes(enum format_id format, unsigned width, unsigned height) {
    if (format == FORMAT_YUYV || format == FORMAT_NV12) {
        if (width % 2u) return 0;
        if (format == FORMAT_NV12 && height % 2u) return 0;
    }
    return frame_bytes(format, width, height);
}

/* Floor division by 256 for negative values too, so this matches Python's `>>`. */
static int floor_shift8(int value) {
    return value >= 0 ? value >> 8 : -(((-value) + 255) >> 8);
}

static uint8_t clip_u8(int value) {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return (uint8_t)value;
}

static void decode_pixel(uint8_t y, uint8_t u, uint8_t v, uint8_t *rgb) {
    int du = (int)u - 128, dv = (int)v - 128;
    rgb[0] = clip_u8((int)y + floor_shift8(359 * dv));
    rgb[1] = clip_u8((int)y - floor_shift8(88 * du + 183 * dv));
    rgb[2] = clip_u8((int)y + floor_shift8(454 * du));
}

static int decode_to_rgb(const uint8_t *frame, enum format_id format, unsigned width,
                         unsigned height, uint8_t *rgb) {
    if (format == FORMAT_RGB) {
        memcpy(rgb, frame, (size_t)width * height * 3u);
        return 0;
    }
    if (format == FORMAT_YUYV) {
        for (unsigned y = 0; y < height; y++) {
            size_t row = (size_t)y * width * 2u;
            for (unsigned x = 0; x < width; x += 2u) {
                size_t base = row + (size_t)x * 2u;
                uint8_t u = frame[base + 1], v = frame[base + 3];
                decode_pixel(frame[base], u, v, rgb + ((size_t)y * width + x) * 3u);
                decode_pixel(frame[base + 2], u, v, rgb + ((size_t)y * width + x + 1) * 3u);
            }
        }
        return 0;
    }
    size_t luma_plane = (size_t)width * height;
    for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            size_t chroma = luma_plane + 2u * ((size_t)(y / 2u) * (width / 2u) + x / 2u);
            decode_pixel(frame[(size_t)y * width + x], frame[chroma], frame[chroma + 1],
                         rgb + ((size_t)y * width + x) * 3u);
        }
    }
    return 0;
}

/* Capture buffer -> packed NHWC UINT8 at the model geometry. Returns 0 or -1. */
static int convert_frame(const uint8_t *frame, enum format_id format, unsigned width,
                         unsigned height, uint8_t *packed, unsigned out_width,
                         unsigned out_height) {
    uint8_t *rgb = malloc((size_t)width * height * 3u);
    if (!rgb) { fprintf(stderr, "camera_npu: allocation of the RGB raster failed\n"); return -1; }
    decode_to_rgb(frame, format, width, height, rgb);
    for (unsigned y = 0; y < out_height; y++) {
        unsigned source_y = (unsigned)(((uint64_t)y * height) / out_height);
        for (unsigned x = 0; x < out_width; x++) {
            unsigned source_x = (unsigned)(((uint64_t)x * width) / out_width);
            memcpy(packed + ((size_t)y * out_width + x) * 3u,
                   rgb + ((size_t)source_y * width + source_x) * 3u, 3u);
        }
    }
    free(rgb);
    return 0;
}

static const char *format_name(enum format_id format) {
    return format == FORMAT_YUYV ? "yuyv" : format == FORMAT_NV12 ? "nv12" : "rgb";
}

/* --------------------------------------------------------------------------- */
/* File I/O                                                                    */
/* --------------------------------------------------------------------------- */
static uint8_t *read_exact(const char *path, size_t size) {
    FILE *file = fopen(path, "rb");
    if (!file) { fail(path, "open", errno); return NULL; }
    uint8_t *data = malloc(size ? size : 1u);
    if (!data) { fclose(file); fprintf(stderr, "camera_npu: allocation failed\n"); return NULL; }
    size_t got = fread(data, 1, size, file);
    int trailing = fgetc(file);
    fclose(file);
    if (got != size || trailing != EOF) {
        fprintf(stderr, "camera_npu: %s must be exactly %zu bytes\n", path, size);
        free(data);
        return NULL;
    }
    return data;
}

static int write_exact(const char *path, const uint8_t *data, size_t size) {
    FILE *file = fopen(path, "wb");
    if (!file) return fail(path, "open", errno);
    if (size && fwrite(data, 1, size, file) != size) {
        int error = errno;
        fclose(file);
        return fail(path, "write", error);
    }
    if (fclose(file)) return fail(path, "close", errno);
    return 0;
}

/* --------------------------------------------------------------------------- */
/* V4L2 capture                                                                */
/* --------------------------------------------------------------------------- */
static uint32_t pixel_format(enum format_id format) {
    if (format == FORMAT_YUYV) return V4L2_PIX_FMT_YUYV;
    if (format == FORMAT_NV12) return V4L2_PIX_FMT_NV12;
    return V4L2_PIX_FMT_RGB24;
}

/* A driver that never queues a frame must not hang the program: wait, then report. */
static int wait_readable(int fd, int timeout_ms) {
    for (;;) {
        fd_set readable;
        FD_ZERO(&readable);
        FD_SET(fd, &readable);
        struct timeval timeout;
        timeout.tv_sec = timeout_ms / 1000;
        timeout.tv_usec = (timeout_ms % 1000) * 1000;
        int ready = select(fd + 1, &readable, NULL, NULL, &timeout);
        if (ready > 0) return 0;
        if (ready < 0 && errno == EINTR) continue;
        return ready < 0 ? -errno : -ETIMEDOUT;
    }
}

/* Open the node, negotiate the format, capture one frame with mmap, return it in
 * `*frame` (malloc'd, the caller frees). Returns 0 or 1 with the error already printed. */
static int capture_frame(const char *device, enum format_id format, unsigned width,
                         unsigned height, uint8_t **frame) {
    size_t want = expected_frame_bytes(format, width, height);
    if (!want) {
        fprintf(stderr, "camera_npu: %s needs an even width (and height for nv12)\n", device);
        return 1;
    }
    int fd = open(device, O_RDWR | O_CLOEXEC);
    if (fd < 0) return fail(device, "open", errno);

    struct v4l2_capability capability;
    memset(&capability, 0, sizeof(capability));
    if (ioctl(fd, VIDIOC_QUERYCAP, &capability) < 0) {
        int error = errno;
        close(fd);
        return fail(device, "VIDIOC_QUERYCAP", error);
    }
    if (!(capability.capabilities & V4L2_CAP_VIDEO_CAPTURE) &&
        !(capability.capabilities & V4L2_CAP_DEVICE_CAPS)) {
        fprintf(stderr, "camera_npu: %s is not a video-capture device\n", device);
        close(fd);
        return 1;
    }
    uint32_t capabilities = capability.capabilities;
    if (capabilities & V4L2_CAP_DEVICE_CAPS) capabilities = capability.device_caps;
    if (!(capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        /* Every capture node on the reference board (rkisp_mainpath, the CIF channels and
         * the scale channels) advertises only V4L2_CAP_VIDEO_CAPTURE_MPLANE, so name that
         * instead of leaving a bare refusal. */
        if (capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE)
            fprintf(stderr, "camera_npu: %s is multi-planar (device_caps=0x%08x has "
                            "V4L2_CAP_VIDEO_CAPTURE_MPLANE); this harness captures "
                            "single-planar\n", device, capabilities);
        else
            fprintf(stderr, "camera_npu: %s is not a video-capture device "
                            "(device_caps=0x%08x)\n", device, capabilities);
        close(fd);
        return 1;
    }
    if (!(capabilities & V4L2_CAP_STREAMING)) {
        fprintf(stderr, "camera_npu: %s does not support streaming I/O\n", device);
        close(fd);
        return 1;
    }

    struct v4l2_format negotiated;
    memset(&negotiated, 0, sizeof(negotiated));
    negotiated.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    negotiated.fmt.pix.width = width;
    negotiated.fmt.pix.height = height;
    negotiated.fmt.pix.pixelformat = pixel_format(format);
    negotiated.fmt.pix.field = V4L2_FIELD_ANY;
    if (ioctl(fd, VIDIOC_S_FMT, &negotiated) < 0) {
        int error = errno;
        close(fd);
        return fail(device, "VIDIOC_S_FMT", error);
    }
    if (negotiated.fmt.pix.pixelformat != pixel_format(format) ||
        negotiated.fmt.pix.width != width || negotiated.fmt.pix.height != height ||
        (negotiated.fmt.pix.bytesperline && negotiated.fmt.pix.bytesperline != width) ||
        negotiated.fmt.pix.sizeimage != want) {
        fprintf(stderr, "camera_npu: %s negotiated %.4s %ux%u stride=%u sizeimage=%u, "
                        "wanted %s %ux%u stride=%u sizeimage=%zu\n", device,
                (const char *)&negotiated.fmt.pix.pixelformat, negotiated.fmt.pix.width,
                negotiated.fmt.pix.height, negotiated.fmt.pix.bytesperline,
                negotiated.fmt.pix.sizeimage, format_name(format), width, height, width, want);
        close(fd);
        return 1;
    }

    struct v4l2_requestbuffers request;
    memset(&request, 0, sizeof(request));
    request.count = MMAP_BUFFERS;
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &request) < 0) {
        int error = errno;
        close(fd);
        return fail(device, "VIDIOC_REQBUFS", error);
    }
    if (request.count < 1) {
        fprintf(stderr, "camera_npu: %s granted no capture buffer\n", device);
        close(fd);
        return 1;
    }

    void *mapped[MMAP_BUFFERS];
    size_t mapped_size[MMAP_BUFFERS];
    unsigned buffers = request.count < MMAP_BUFFERS ? request.count : MMAP_BUFFERS;
    for (unsigned index = 0; index < buffers; index++) {
        mapped[index] = MAP_FAILED;
        struct v4l2_buffer buffer;
        memset(&buffer, 0, sizeof(buffer));
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = index;
        if (ioctl(fd, VIDIOC_QUERYBUF, &buffer) < 0) {
            int error = errno;
            close(fd);
            return fail(device, "VIDIOC_QUERYBUF", error);
        }
        mapped_size[index] = buffer.length;
        mapped[index] = mmap(NULL, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                             buffer.m.offset);
        if (mapped[index] == MAP_FAILED) {
            int error = errno;
            close(fd);
            return fail(device, "mmap", error);
        }
        if (ioctl(fd, VIDIOC_QBUF, &buffer) < 0) {
            int error = errno;
            munmap(mapped[index], mapped_size[index]);
            close(fd);
            return fail(device, "VIDIOC_QBUF", error);
        }
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
        int error = errno;
        close(fd);
        return fail(device, "VIDIOC_STREAMON", error);
    }

    int ready = wait_readable(fd, CAPTURE_TIMEOUT_MS);
    if (ready) {
        ioctl(fd, VIDIOC_STREAMOFF, &type);
        for (unsigned index = 0; index < buffers; index++) munmap(mapped[index], mapped_size[index]);
        close(fd);
        if (ready == -ETIMEDOUT) {
            fprintf(stderr, "camera_npu: %s delivered no frame within %d ms "
                            "(nothing is streaming this node)\n", device, CAPTURE_TIMEOUT_MS);
            return 1;
        }
        return fail(device, "select", -ready);
    }

    struct v4l2_buffer ready_buffer;
    memset(&ready_buffer, 0, sizeof(ready_buffer));
    ready_buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ready_buffer.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_DQBUF, &ready_buffer) < 0) {
        int error = errno;
        ioctl(fd, VIDIOC_STREAMOFF, &type);
        for (unsigned index = 0; index < buffers; index++) munmap(mapped[index], mapped_size[index]);
        close(fd);
        return fail(device, "VIDIOC_DQBUF", error);
    }
    if (ready_buffer.index >= buffers || ready_buffer.bytesused < want) {
        fprintf(stderr, "camera_npu: %s dequeued buffer %u with %u of %zu bytes\n", device,
                ready_buffer.index, ready_buffer.bytesused, want);
        ioctl(fd, VIDIOC_STREAMOFF, &type);
        for (unsigned index = 0; index < buffers; index++) munmap(mapped[index], mapped_size[index]);
        close(fd);
        return 1;
    }
    *frame = malloc(want);
    if (!*frame) {
        fprintf(stderr, "camera_npu: allocation of the captured frame failed\n");
        ioctl(fd, VIDIOC_STREAMOFF, &type);
        for (unsigned index = 0; index < buffers; index++) munmap(mapped[index], mapped_size[index]);
        close(fd);
        return 1;
    }
    memcpy(*frame, mapped[ready_buffer.index], want);
    ioctl(fd, VIDIOC_QBUF, &ready_buffer);
    ioctl(fd, VIDIOC_STREAMOFF, &type);
    for (unsigned index = 0; index < buffers; index++) munmap(mapped[index], mapped_size[index]);
    close(fd);
    printf("capture device=%s format=%s %ux%u bytes=%zu\n", device, format_name(format),
           width, height, want);
    return 0;
}

/* --------------------------------------------------------------------------- */
/* NPU                                                                         */
/* --------------------------------------------------------------------------- */
struct result {
    unsigned long long exact_bytes, mismatches, inferences;
    int compared;
};

static void compare(const int8_t *output, size_t bytes, const int8_t *expected,
                    unsigned run, struct result *result) {
    result->inferences++;
    if (!result->compared) return;
    if (!memcmp(output, expected, bytes)) { result->exact_bytes += bytes; return; }
    result->mismatches++;
    size_t at = 0;
    while (at < bytes && output[at] == expected[at]) at++;
    printf("mismatch run=%u first_diff=%zu expected=%d got=%d\n", run, at,
           at < bytes ? (int)expected[at] : 0, at < bytes ? (int)output[at] : 0);
}

static int print_summary(const struct options *options, const ornpu_info *info,
                         size_t input_bytes, int path, const struct result *result) {
    printf("SUMMARY model=%s format=%s input_bytes=%zu output_bytes=%u runs=%u path=%s "
           "inferences=%llu exact_bytes=%llu mismatches=%llu result=%s\n",
           options->model, format_name(options->format), input_bytes, info->output_bytes,
           options->runs, path == PATH_SHARED ? "shared" : "copy", result->inferences,
           result->exact_bytes, result->mismatches,
           result->mismatches == 0 ? "PASS" : "FAIL");
    return result->mismatches == 0 ? 0 : 1;
}

static int run_copy(const struct options *options, const uint8_t *packed, size_t packed_bytes) {
    ornpu_model *model = NULL;
    int rc = ornpu_open(options->model, &model);
    if (rc) return fail(options->model, "ornpu_open", -rc);
    ornpu_info info;
    if (ornpu_get_info(model, &info)) {
        ornpu_close(model);
        fprintf(stderr, "camera_npu: %s: ornpu_get_info failed\n", options->model);
        return 1;
    }
    if (info.input_bytes != packed_bytes) {
        fprintf(stderr, "camera_npu: %s wants %u input bytes, the conversion produced %zu\n",
                options->model, info.input_bytes, packed_bytes);
        ornpu_close(model);
        return 1;
    }
    int8_t *output = malloc(info.output_bytes ? info.output_bytes : 1u);
    int8_t *expected = NULL;
    if (!output) { ornpu_close(model); fprintf(stderr, "camera_npu: allocation failed\n"); return 1; }
    struct result result = {0, 0, 0, 0};
    if (options->expected) {
        expected = (int8_t *)read_exact(options->expected, info.output_bytes);
        if (!expected) { free(output); ornpu_close(model); return 1; }
        result.compared = 1;
    }
    for (unsigned run = 0; run < options->runs; run++) {
        rc = ornpu_run(model, packed, packed_bytes, output, info.output_bytes);
        if (rc) {
            free(output); free(expected); ornpu_close(model);
            return fail(options->model, "ornpu_run", -rc);
        }
        compare(output, info.output_bytes, expected, run, &result);
    }
    int status = print_summary(options, &info, packed_bytes, PATH_COPY, &result);
    free(output);
    free(expected);
    ornpu_close(model);
    return status;
}

static int run_shared_or_copy(const struct options *options, const uint8_t *packed,
                              size_t packed_bytes) {
    ornpu_model *model = NULL;
    int arena_fd = -1;
    int rc = ornpu_open_shared(options->model, &model, &arena_fd);
    if (rc) {
        printf("ZEROCOPY unavailable (ornpu_open_shared rc=%d); falling back to ornpu_run\n", rc);
        return run_copy(options, packed, packed_bytes);
    }
    ornpu_info info;
    struct ornpu_input_view view;
    if (ornpu_get_info(model, &info) || ornpu_input_view(model, 0, &view)) {
        ornpu_close(model);
        printf("ZEROCOPY unavailable (packed input view missing); falling back to ornpu_run\n");
        return run_copy(options, packed, packed_bytes);
    }
    if (view.width != info.width || view.channels != info.input_channels ||
        (size_t)view.batch * view.height * view.width * view.channels != packed_bytes) {
        ornpu_close(model);
        printf("ZEROCOPY view does not match the model input; falling back to ornpu_run\n");
        return run_copy(options, packed, packed_bytes);
    }
    uint8_t *arena = mmap(NULL, info.arena_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, arena_fd, 0);
    if (arena == MAP_FAILED) {
        int error = errno;
        ornpu_close(model);
        return fail(options->model, "mmap(shared arena)", error);
    }
    int8_t *output = malloc(info.output_bytes ? info.output_bytes : 1u);
    int8_t *expected = NULL;
    if (!output) {
        munmap(arena, info.arena_bytes);
        ornpu_close(model);
        fprintf(stderr, "camera_npu: allocation failed\n");
        return 1;
    }
    struct result result = {0, 0, 0, 0};
    if (options->expected) {
        expected = (int8_t *)read_exact(options->expected, info.output_bytes);
        if (!expected) {
            free(output);
            munmap(arena, info.arena_bytes);
            ornpu_close(model);
            return 1;
        }
        result.compared = 1;
    }
    printf("ZEROCOPY arena_fd=%d offset=%llu row_stride=%u %ux%ux%ux%u\n", arena_fd,
           (unsigned long long)view.offset, view.row_stride, view.batch, view.height,
           view.width, view.channels);
    for (unsigned run = 0; run < options->runs; run++) {
        /* The producer writes the engine's own memory, row by row in the arena layout:
         * a V4L2/ISP producer with this stride would write the rows itself. */
        for (uint32_t batch = 0; batch < view.batch; batch++)
            for (uint32_t row = 0; row < view.height; row++)
                memcpy(arena + view.offset +
                           ((uint64_t)batch * view.height + row) * view.row_stride * view.channels,
                       packed + ((uint64_t)batch * view.height + row) * view.width * view.channels,
                       (size_t)view.width * view.channels);
        rc = ornpu_run_prefilled(model, output, info.output_bytes);
        if (rc) {
            free(output); free(expected);
            munmap(arena, info.arena_bytes);
            ornpu_close(model);
            return fail(options->model, "ornpu_run_prefilled", -rc);
        }
        compare(output, info.output_bytes, expected, run, &result);
    }
    int status = print_summary(options, &info, packed_bytes, PATH_SHARED, &result);
    free(output);
    free(expected);
    munmap(arena, info.arena_bytes);
    ornpu_close(model);
    return status;
}

/* --------------------------------------------------------------------------- */
/* main                                                                        */
/* --------------------------------------------------------------------------- */
int main(int argc, char **argv) {
    struct options options;
    int parsed = parse_options(argc, argv, &options);
    if (parsed > 0) return 0;   /* --help */
    if (parsed < 0) { usage(stderr); return 2; }

    /* The model geometry defines the conversion target; inspect needs no device. */
    ornpu_info info;
    int rc = ornpu_inspect(options.model, &info);
    if (rc) return fail(options.model, "ornpu_inspect", -rc);
    if (!info.height || !info.width || !info.input_channels) {
        fprintf(stderr, "camera_npu: %s has no image-shaped input\n", options.model);
        return 1;
    }
    size_t packed_bytes = (size_t)info.batch * info.height * info.width * info.input_channels;

    uint8_t *frame = NULL;
    if (options.frame) {
        size_t want = expected_frame_bytes(options.format, options.width, options.height);
        if (!want) {
            fprintf(stderr, "camera_npu: %s needs an even width (and height for nv12)\n",
                    format_name(options.format));
            return 1;
        }
        frame = read_exact(options.frame, want);
        if (!frame) return 1;
    } else {
        rc = capture_frame(options.device, options.format, options.width, options.height, &frame);
        if (rc) return rc;
    }

    uint8_t *packed = malloc(packed_bytes);
    if (!packed) { free(frame); fprintf(stderr, "camera_npu: allocation failed\n"); return 1; }
    if (convert_frame(frame, options.format, options.width, options.height, packed,
                      info.width, info.height)) {
        free(frame);
        free(packed);
        return 1;
    }
    free(frame);

    if (options.emit_input) {
        rc = write_exact(options.emit_input, packed, packed_bytes);
        if (!rc)
            printf("EMIT model=%s format=%s frame=%ux%u input=%ux%u bytes=%zu\n",
                   options.model, format_name(options.format), options.width, options.height,
                   info.width, info.height, packed_bytes);
        free(packed);
        return rc ? 1 : 0;
    }

    int status = options.device ? run_shared_or_copy(&options, packed, packed_bytes)
                                : run_copy(&options, packed, packed_bytes);
    free(packed);
    return status;
}
