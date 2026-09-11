/* SPDX-License-Identifier: MIT
 * Development-only recorder for our own small probe process. No kernel hooks.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

struct allocation { uint32_t handle, flags; uint64_t size, object, dma, sram; };
static struct allocation allocations[32];
static unsigned count, run;
static int (*real_ioctl)(int, unsigned long, ...);

static void save(const char *suffix, const void *data, size_t size) {
    const char *dir = getenv("OPEN_NPU_CAPTURE");
    if (!dir) return;
    char path[256];
    int n = snprintf(path, sizeof(path), "%s/run%u_%s.bin", dir, run, suffix);
    if (n < 0 || n >= (int)sizeof(path)) return;
    FILE *f = fopen(path, "wb");
    if (!f) { perror("capture fopen"); return; }
    if (fwrite(data, 1, size, f) != size) perror("capture fwrite");
    fclose(f);
}

static void snapshot(const char *phase) {
    for (unsigned i=0; i<count; i++) {
        struct allocation *a = &allocations[i];
        if (!a->size || a->size > 1024*1024) continue;
        void *p = mmap(NULL, a->size, PROT_READ, MAP_SHARED, a->handle, 0);
        if (p == MAP_FAILED) { perror("capture mmap"); continue; }
        char suffix[64];
        snprintf(suffix, sizeof(suffix), "%s_mem%u", phase, i);
        save(suffix, p, a->size);
        munmap(p, a->size);
    }
}

int ioctl(int fd, unsigned long request, ...) {
    va_list ap;
    va_start(ap, request);
    void *arg = va_arg(ap, void *);
    va_end(ap);
    if (!real_ioctl) real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    if (!real_ioctl) { errno = ENOSYS; return -1; }
    int is_npu = _IOC_TYPE(request) == 'r';
    unsigned nr = _IOC_NR(request), size = _IOC_SIZE(request);
    if (is_npu && nr == 1 && arg && run < 4) {
        save("submit", arg, size);
        snapshot("before");
    }
    int rc = real_ioctl(fd, request, arg);
    int saved_errno = errno;
    if (is_npu && nr == 2 && rc == 0 && arg && size >= 32 && count < 32) {
        struct allocation *a = &allocations[count];
        *a = (struct allocation){0};
        __builtin_memcpy(a, arg, size < sizeof(*a) ? size : sizeof(*a));
        fprintf(stderr, "ALLOC %u fd=%u flags=%x size=%llu object=%llx dma=%llx ioctl=%lx\n", count, a->handle, a->flags, (unsigned long long)a->size, (unsigned long long)a->object, (unsigned long long)a->dma, request);
        count++;
    }
    if (is_npu && nr == 4 && rc == 0 && arg) {
        uint32_t handle = *(uint32_t *)arg;
        for (unsigned i=0; i<count; i++) if (allocations[i].handle == handle) allocations[i].size=0;
    }
    if (is_npu && nr == 1 && arg && run < 4) {
        fprintf(stderr, "SUBMIT %u rc=%d size=%u\n", run, rc, size);
        snapshot("after");
        run++;
    }
    errno = saved_errno;
    return rc;
}
