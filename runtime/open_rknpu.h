/* SPDX-License-Identifier: MIT */
#ifndef OPEN_RKNPU_H
#define OPEN_RKNPU_H
#include <stddef.h>
#include <stdint.h>

typedef struct ornpu_model ornpu_model;
typedef struct {
    uint32_t height, width, input_channels, output_channels;
    uint32_t input_bytes, output_bytes;
    float output_scale;
    int32_t output_zero_point;
    uint32_t output_height, output_width;
    float input_scale;
    uint32_t input_zero_point;
    uint32_t batch;
    uint32_t input_tensor_count;
    uint32_t constant_count;
    uint32_t tensor_count;
    uint32_t output_tensor_count;
    uint32_t task_count;          /* submission descriptors in the container */
    uint32_t submission_serial;   /* 1: one descriptor per submission */
    uint32_t engine_runs;         /* ioctls a non-serial container is submitted as */
} ornpu_info;
/* Named tensor table entry (format v5): arena storage plus flat API placement. */
typedef struct {
    char name[24];
    uint32_t role, layout, index;
    uint32_t batch, height, width, channels;
    uint32_t arena_offset, arena_bytes;
    uint32_t api_offset, api_bytes;
} ornpu_tensor_info;
typedef struct {
    char name[24];
    uint32_t byte_offset, bytes, kind;
} ornpu_constant_info;
#define ORNPU_CONSTANT_PACKED_PARAMETERS 1u
#define ORNPU_CONSTANT_PACKED_BIAS 2u
#define ORNPU_CONSTANT_FACTOR 3u
#define ORNPU_TENSOR_INPUT 0u
#define ORNPU_TENSOR_OUTPUT 1u
#define ORNPU_TENSOR_INTERNAL 2u
#define ORNPU_LAYOUT_PACKED 0u
#define ORNPU_LAYOUT_NATIVE16 1u
#define ORNPU_LAYOUT_PACKED_INT8 2u
/* One caller buffer bound to a named tensor for ornpu_run_io(). */
typedef struct {
    uint32_t tensor_index;
    void *data;
    size_t size;
} ornpu_io;

/* The driver also exposes an ACTION ioctl (versions, clock, regulator, bandwidth
 * counters, IOMMU, SRAM). This runtime issues none of it: the only results that carry
 * information on the attached board are the version, the 420 MHz clock and the
 * bandwidth counters, and `GET_VOLT` (action 4) oopses the caller because the device
 * tree has no rknpu regulator. Measured in research/action_probe/README.md. */
/* Driver job flags accepted by ornpu_submit_flags(). */
#define ORNPU_JOB_NONBLOCK 0x2u   /* return before the job completes */
#define ORNPU_JOB_FENCE_IN 0x8u   /* -EINVAL: the attached kernel has no fence support */
#define ORNPU_JOB_FENCE_OUT 0x10u

/* Experimental cross-job pipelining (docs/plans/pipelining-plan.md S4). `ornpu_submit_flags`
 * submits the current model with extra driver job flags (0x2 NONBLOCK, 0x8
 * FENCE_IN, 0x10 FENCE_OUT); for FENCE_IN the value in `*fence_fd` is the input
 * fence, and `*fence_fd` receives the fd the driver allocated for FENCE_OUT. `ornpu_wait_fence` polls it; `ornpu_sync_outputs` makes device writes
 * visible before reading results. */
int ornpu_submit_flags(ornpu_model *model, uint32_t extra_flags, int *fence_fd);
/* Experimental job-field probe (docs/plans/pipelining-plan.md S8): set the submission's
 * core_mask and up to five {task_start, task_number} subcore_task[] windows for the
 * next submissions. RV1106 has one IRQ, so the driver forces core_mask to CORE0 and
 * never reads subcore_task[]; the verified output must not change. */
int ornpu_set_submit_core(ornpu_model *model, uint32_t core_mask,
                          const uint32_t *subcore, uint32_t pairs);
int ornpu_set_input(ornpu_model *model, uint32_t tensor_index, const void *data, size_t size);
int ornpu_sync_outputs(ornpu_model *model);
int ornpu_wait_fence(int fence_fd, int timeout_ms);
/* Fence-free completion (`research/barrier_probe/`): with ORNPU_JOB_NONBLOCK there is
 * no pollable completion fd on this kernel (ORNPU_JOB_FENCE_OUT returns -EINVAL), but
 * jobs run in order per core, so a small blocking submission after a queued one
 * completes only after it:
 *
 *   ornpu_submit_flags(model, ORNPU_JOB_NONBLOCK, NULL);
 *   ornpu_run(barrier, barrier_input, barrier_input_size, barrier_output, barrier_output_size);
 *   ornpu_sync_outputs(model);            // model's outputs are now final
 *
 * That reads each output as soon as it is produced (lag 0) at the cost of one small job
 * per inference; the lag-1 alternative is to drain with the next real inference. */
/* Return 0 or a negative errno. Each model instance is used synchronously. */
int ornpu_inspect(const char *path, ornpu_info *info);
int ornpu_open(const char *path, ornpu_model **model);
int ornpu_get_info(const ornpu_model *model, ornpu_info *info);
int ornpu_get_input_tensor(const ornpu_model *model, uint32_t index, ornpu_tensor_info *info);
int ornpu_get_tensor(const ornpu_model *model, uint32_t index, ornpu_tensor_info *info);
int ornpu_get_constant(const ornpu_model *model, uint32_t index, ornpu_constant_info *info);
/* Replace one complete compiler-described packed constant region. */
int ornpu_set_constant(ornpu_model *model, uint32_t index, const void *data, size_t size);
/* Input: packed NHWC UINT8; real value=(byte-input_zero_point)*input_scale.
 * Output: packed NHWC INT8. No padding in API buffers. */
int ornpu_run(ornpu_model *model, const uint8_t *input, size_t input_size,
              int8_t *output, size_t output_size);
/* Format v5: bind one packed NHWC buffer per external input/output tensor.
 * tensor_index refers to the named tensor table; every external tensor of the
 * model must be supplied exactly once. */
int ornpu_run_io(ornpu_model *model, const ornpu_io *inputs, uint32_t input_count,
                 ornpu_io *outputs, uint32_t output_count);
void ornpu_close(ornpu_model *model);
#endif
