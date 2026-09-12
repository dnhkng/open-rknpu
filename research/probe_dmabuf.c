/* SPDX-License-Identifier: MIT
 *
 * F8 probe: can the RV1103 NPU driver import a dma-buf fd allocated from /dev/rk_dma_heap?
 *
 * The runtime allocates its buffers through the NPU driver's own CREATE ioctl, so a
 * zero-copy input path from V4L2/ISP would need the driver to accept an existing dma-buf.
 * This probe allocates one buffer from the board's Rockchip dma-heap and then calls CREATE
 * with the fd in `handle` for a range of `flags` values, reporting the return code and the
 * device address the driver wrote back. Any success is followed by a DESTROY.
 *
 * Usage: probe_dmabuf [first_flag] [last_flag]
 */
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

struct dma_heap_allocation_data {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};
#define DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0x0, struct dma_heap_allocation_data)

struct allocation { uint32_t handle, flags; uint64_t size, object, dma, sram; };
#define CREATE _IOWR('r', 2, struct allocation)
#define DESTROY _IOWR('r', 4, struct allocation)

int main(int argc, char **argv) {
    unsigned first = argc > 1 ? (unsigned)strtoul(argv[1], NULL, 0) : 0;
    unsigned last = argc > 2 ? (unsigned)strtoul(argv[2], NULL, 0) : 255;
    int heap = open("/dev/rk_dma_heap/rk-dma-heap-cma", O_RDWR | O_CLOEXEC);
    if (heap < 0) { printf("heap: open failed: %s\n", strerror(errno)); return 2; }
    struct dma_heap_allocation_data request = {.len = 4096, .fd_flags = O_RDWR | O_CLOEXEC};
    if (ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &request) < 0) {
        printf("heap: DMA_HEAP_IOCTL_ALLOC failed: %s (errno %d)\n", strerror(errno), errno);
        return 2;
    }
    printf("heap: allocated 4096 bytes, fd=%u\n", request.fd);

    int npu = open("/dev/rknpu", O_RDWR | O_CLOEXEC);
    if (npu < 0) { printf("npu: open failed: %s\n", strerror(errno)); return 2; }
    unsigned accepted = 0;
    for (unsigned flags = first; flags <= last; flags++) {
        struct allocation alloc;
        memset(&alloc, 0, sizeof(alloc));
        alloc.handle = (uint32_t)request.fd;
        alloc.flags = flags;
        alloc.size = 4096;
        int rc = ioctl(npu, CREATE, &alloc);
        if (rc == 0) {
            printf("flags=0x%02x rc=0 handle=%u object=0x%llx dma=0x%llx sram=0x%llx\n",
                   flags, alloc.handle, (unsigned long long)alloc.object,
                   (unsigned long long)alloc.dma, (unsigned long long)alloc.sram);
            accepted++;
            struct allocation release;
            memset(&release, 0, sizeof(release));
            release.handle = alloc.handle;
            release.object = alloc.object;
            ioctl(npu, DESTROY, &release);
        } else if (errno != EINVAL && errno != ENOTTY) {
            printf("flags=0x%02x rc=%d errno=%d (%s)\n", flags, rc, errno, strerror(errno));
        }
    }
    printf("summary: %u of %u flag values accepted a dma-buf fd\n", accepted, last - first + 1);
    close(npu);
    close(heap);
    return accepted ? 0 : 1;
}
