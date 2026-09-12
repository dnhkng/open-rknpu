/* SPDX-License-Identifier: MIT
 *
 * Host C/ABI tests for the RV1103 runtime container loader (checklist T3).
 *
 * The runtime validates a container *before* it opens /dev/rknpu, so every
 * rejection path is reachable on any little-endian Linux host with the system
 * compiler. This program is deliberately libc-only and reads nothing but the
 * files named on its command line; the Python wrapper (tests/test_host_loader.py)
 * builds it against runtime/open_rknpu.c, writes the fixture containers (real
 * ones copied from research/ and crafted malformed ones) and asserts the rc
 * each case produces.
 *
 * Usage:
 *   host_loader '<case>|<op>|<expected>|<path>[|<checks>]' [more cases...]
 *
 *   <case>      label printed with the per-case summary line
 *   <op>        inspect | open
 *   <expected>  accepted return codes, comma separated: 0, -EINVAL, -ENOENT,
 *               -ENODEV, -ENOMEM, -ETIMEDOUT or a decimal int
 *   <path>      container file to load
 *   <checks>    optional comma-separated key=value assertions on ornpu_info,
 *               evaluated only when the call returned 0
 *
 * ornpu_info keys: height, width, input_channels, output_channels, input_bytes,
 * output_bytes, output_zero_point, output_height, output_width,
 * input_zero_point, batch, input_tensor_count, constant_count, tensor_count,
 * output_tensor_count, task_count, submission_serial, engine_runs,
 * output_scale, input_scale.
 *
 * Output is one line per case, machine-readable and greppable:
 *   ok  <case> rc=<n>
 *   FAIL <case> rc=<n> expected=<list> (<why>)
 * followed by a final SUMMARY line. Exit status is 0 only when every case
 * matched, 2 for a usage/parse error.
 */
#include "open_rknpu.h"
#include <dirent.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FIELDS 5

struct code_name {
    const char *name;
    int value;
};

static const struct code_name CODES[] = {
    {"EINVAL", EINVAL},
    {"ENOENT", ENOENT},
    {"ENODEV", ENODEV},
    {"ENOMEM", ENOMEM},
    {"ETIMEDOUT", ETIMEDOUT},
};

/* Parse one symbolic or decimal return code. */
static int parse_code(const char *text, int *value) {
    int negative = 0;
    const char *name = text;
    if (*name == '-') {
        negative = 1;
        name++;
    }
    if (!strcmp(name, "0")) {
        *value = 0;
        return 1;
    }
    for (size_t i = 0; i < sizeof(CODES) / sizeof(CODES[0]); i++) {
        if (!strcmp(name, CODES[i].name)) {
            *value = negative ? -CODES[i].value : CODES[i].value;
            return 1;
        }
    }
    char *end = NULL;
    long parsed = strtol(text, &end, 10);
    if (*text && end && !*end) {
        *value = (int)parsed;
        return 1;
    }
    return 0;
}

/* 1 when `rc` is one of the comma-separated codes in `list`. */
static int expected_matches(const char *list, int rc) {
    char *copy = strdup(list);
    if (!copy) return 0;
    int matched = 0;
    char *cursor = copy;
    char *token;
    while ((token = strsep(&cursor, ",")) != NULL) {
        int wanted;
        if (parse_code(token, &wanted) && wanted == rc) {
            matched = 1;
            break;
        }
    }
    free(copy);
    return matched;
}

/* Number of descriptors this process holds, or -1 when /proc is unavailable. */
static int count_open_fds(void) {
    DIR *dir = opendir("/proc/self/fd");
    if (!dir) return -1;
    int count = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) count++;
    }
    closedir(dir);
    return count - 1; /* the directory's own descriptor is listed too */
}

static int info_field_u32(const ornpu_info *info, const char *key, uint32_t *out) {
    if (!strcmp(key, "height")) *out = info->height;
    else if (!strcmp(key, "width")) *out = info->width;
    else if (!strcmp(key, "input_channels")) *out = info->input_channels;
    else if (!strcmp(key, "output_channels")) *out = info->output_channels;
    else if (!strcmp(key, "input_bytes")) *out = info->input_bytes;
    else if (!strcmp(key, "output_bytes")) *out = info->output_bytes;
    else if (!strcmp(key, "output_zero_point")) *out = (uint32_t)info->output_zero_point;
    else if (!strcmp(key, "output_height")) *out = info->output_height;
    else if (!strcmp(key, "output_width")) *out = info->output_width;
    else if (!strcmp(key, "input_zero_point")) *out = info->input_zero_point;
    else if (!strcmp(key, "batch")) *out = info->batch;
    else if (!strcmp(key, "input_tensor_count")) *out = info->input_tensor_count;
    else if (!strcmp(key, "constant_count")) *out = info->constant_count;
    else if (!strcmp(key, "tensor_count")) *out = info->tensor_count;
    else if (!strcmp(key, "output_tensor_count")) *out = info->output_tensor_count;
    else if (!strcmp(key, "task_count")) *out = info->task_count;
    else if (!strcmp(key, "submission_serial")) *out = info->submission_serial;
    else if (!strcmp(key, "engine_runs")) *out = info->engine_runs;
    else return 0;
    return 1;
}

static int info_field_float(const ornpu_info *info, const char *key, float *out) {
    if (!strcmp(key, "output_scale")) *out = info->output_scale;
    else if (!strcmp(key, "input_scale")) *out = info->input_scale;
    else return 0;
    return 1;
}

/* Evaluate the `key=value,key=value` list; returns 1 and fills `why` on failure. */
static int check_info(const ornpu_info *info, const char *checks, char *why, size_t why_size) {
    if (!checks || !*checks) return 1;
    char *copy = strdup(checks);
    if (!copy) {
        snprintf(why, why_size, "out of memory parsing checks");
        return 0;
    }
    int ok = 1;
    char *cursor = copy;
    char *item;
    while (ok && (item = strsep(&cursor, ",")) != NULL) {
        char *equals = strchr(item, '=');
        if (!equals) {
            snprintf(why, why_size, "malformed check '%s'", item);
            ok = 0;
            break;
        }
        *equals = '\0';
        const char *key = item;
        const char *text = equals + 1;
        uint32_t actual_u32;
        float actual_float;
        if (info_field_u32(info, key, &actual_u32)) {
            char *end = NULL;
            unsigned long wanted = strtoul(text, &end, 10);
            if (!*text || !end || *end || (uint32_t)wanted != actual_u32) {
                snprintf(why, why_size, "%s=%u (want %s)", key, actual_u32, text);
                ok = 0;
            }
        } else if (info_field_float(info, key, &actual_float)) {
            char *end = NULL;
            float wanted = strtof(text, &end);
            if (!*text || !end || *end || wanted != actual_float) {
                snprintf(why, why_size, "%s=%g (want %s)", key, (double)actual_float, text);
                ok = 0;
            }
        } else {
            snprintf(why, why_size, "unknown info key '%s'", key);
            ok = 0;
        }
    }
    free(copy);
    return ok;
}

static int run_case(const char *name, const char *op, const char *expected, const char *path,
                    const char *checks, char *why, size_t why_size) {
    int before = count_open_fds();
    int rc;
    int ok = 1;
    if (!strcmp(op, "inspect")) {
        ornpu_info info;
        memset(&info, 0, sizeof(info));
        rc = ornpu_inspect(path, &info);
        if (rc == 0 && !check_info(&info, checks, why, why_size)) ok = 0;
    } else if (!strcmp(op, "open")) {
        ornpu_model *model = NULL;
        rc = ornpu_open(path, &model);
        /*
         * The host has no /dev/rknpu, so a *valid* container stops at the device
         * open with -ENOENT (no device node) or -ENODEV (node without the driver).
         * That is the expected host behaviour: validation already succeeded by
         * then. A malformed container must fail earlier, in load_program.
         */
        if (rc != 0 && model != NULL) {
            snprintf(why, why_size, "ornpu_open failed but returned a non-NULL model");
            ok = 0;
        }
        if (rc == 0 && model != NULL) ornpu_close(model);
    } else {
        snprintf(why, why_size, "unknown op '%s'", op);
        return 0;
    }
    if (ok && !expected_matches(expected, rc)) {
        snprintf(why, why_size, "expected=%s", expected);
        ok = 0;
    }
    int after = count_open_fds();
    if (ok && before >= 0 && after >= 0 && before != after) {
        snprintf(why, why_size, "file descriptor leak: %d -> %d", before, after);
        ok = 0;
    }
    if (ok) {
        printf("ok  %s rc=%d\n", name, rc);
    } else {
        printf("FAIL %s rc=%d expected=%s (%s)\n", name, rc, expected, why);
    }
    return ok;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s '<case>|<inspect|open>|<expected>|<path>[|<checks>]' ...\n",
                argv[0]);
        return 2;
    }
    unsigned total = 0, failed = 0;
    for (int i = 1; i < argc; i++) {
        char *spec = strdup(argv[i]);
        if (!spec) {
            fprintf(stderr, "out of memory\n");
            return 2;
        }
        char *fields[MAX_FIELDS] = {0};
        unsigned count = 0;
        char *cursor = spec;
        while (count < MAX_FIELDS && (fields[count] = strsep(&cursor, "|")) != NULL) count++;
        if (count < 4) {
            fprintf(stderr, "FAIL <spec> malformed spec '%s'\n", argv[i]);
            free(spec);
            failed++;
            total++;
            continue;
        }
        char why[256] = "";
        int ok = run_case(fields[0], fields[1], fields[2], fields[3],
                          count > 4 ? fields[4] : NULL, why, sizeof(why));
        total++;
        if (!ok) failed++;
        free(spec);
    }
    printf("SUMMARY total=%u failed=%u\n", total, failed);
    return failed ? 1 : 0;
}
