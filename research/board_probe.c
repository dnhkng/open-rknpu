/* SPDX-License-Identifier: MIT
 * Development-only vendor oracle. The eventual runtime must not link RKNN.
 */
#include "rknn_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    rknn_context ctx = 0;
    rknn_tensor_mem *in = NULL, *out = NULL;
    rknn_tensor_attr attrs[4] = {{0}};
    unsigned commands[] = {1, 2, 8, 9};
    int ret = 1;
    void *model = NULL;
    FILE *f = NULL;
    setbuf(stdout, NULL);
    if (argc != 2) { fprintf(stderr, "usage: board_probe model.rknn\n"); return 2; }
#define CHECK(call) do { int rc = (call); if (rc) { fprintf(stderr, "%s: %d\n", #call, rc); goto done; } } while (0)
    printf("ABI attr=%zu mem=%zu ctx=%zu\n", sizeof(attrs[0]), sizeof(*in), sizeof(ctx));
    f = fopen(argv[1], "rb");
    if (!f) { perror("model"); goto done; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    if (size <= 0 || size > 1024*1024) goto done;
    rewind(f);
    model = malloc(size);
    if (!model || fread(model, 1, size, f) != (size_t)size) goto done;
    fclose(f); f = NULL;
    CHECK(rknn_init(&ctx, model, size, 0, NULL));
    for (unsigned i=0; i<4; i++) {
        CHECK(rknn_query(ctx, commands[i], &attrs[i], sizeof(attrs[i])));
        rknn_tensor_attr *a = &attrs[i];
        printf("ATTR %u name=%s dims=", commands[i], a->name);
        for (unsigned j=0; j<a->n_dims; j++) printf("%u,", a->dims[j]);
        printf(" fmt=%d type=%d size=%u stride_size=%u w_stride=%u zp=%d scale=%.9g\n",
               a->fmt, a->type, a->size, a->size_with_stride, a->w_stride, a->zp, a->scale);
    }
    rknn_tensor_attr *logical = &attrs[0], *b = &attrs[3];
    unsigned logical_channels = logical->dims[3];
    rknn_tensor_attr *a = logical_channels > 3 ? &attrs[2] : logical;
    if (logical_channels <= 3) a->type = RKNN_TENSOR_UINT8;
    a->fmt = logical_channels > 3 ? RKNN_TENSOR_NC1HWC2 : RKNN_TENSOR_NHWC;
    a->pass_through = logical_channels > 3;
    in = rknn_create_mem(ctx, a->size_with_stride);
    out = rknn_create_mem(ctx, b->size_with_stride);
    if (!in || !out) { fprintf(stderr, "allocation failed\n"); goto done; }
    printf("MEM input=%u@%llx output=%u@%llx\n", in->size, (unsigned long long)in->phys_addr, out->size, (unsigned long long)out->phys_addr);
    CHECK(rknn_set_io_mem(ctx, in, a));
    CHECK(rknn_set_io_mem(ctx, out, b));
    unsigned height = logical->dims[1], width = logical->dims[2];
    unsigned channels = logical->dims[3];
    unsigned stride = logical->w_stride ? logical->w_stride : width;
    if (logical->n_dims!=4 || logical->dims[0]!=1 || !channels || channels>256 || height>128 || width>128) goto done;
    const char *names[] = {"zero", "constant", "ramp", "impulse"};
    for (unsigned k=0; k<4; k++) {
        memset(in->virt_addr, 0, in->size);
        for (unsigned h=0; h<height; h++) for (unsigned w=0; w<width; w++) for (unsigned c=0; c<channels; c++) {
            unsigned value = k==1 ? 128 : k==2 ? (h*width*channels+w*channels+c)%256 : k==3 && h==3 && w==4 && c==(channels==1 ? 0u : 1u) ? 255 : 0;
            unsigned at = channels>3 ? (c/16)*height*width*16+(h*width+w)*16+c%16 : (h*stride+w)*channels+c;
            if (at < in->size) ((unsigned char *)in->virt_addr)[at] = value;
        }
        CHECK(rknn_mem_sync(ctx, in, RKNN_MEMORY_SYNC_TO_DEVICE));
        CHECK(rknn_run(ctx, NULL));
        CHECK(rknn_mem_sync(ctx, out, RKNN_MEMORY_SYNC_FROM_DEVICE));
        printf("OUTPUT %s ", names[k]);
        for (unsigned j=0; j<b->size_with_stride; j++) printf("%02x", ((unsigned char *)out->virt_addr)[j]);
        putchar('\n');
    }
    ret = 0;
done:
    if (out) rknn_destroy_mem(ctx, out);
    if (in) rknn_destroy_mem(ctx, in);
    if (ctx) rknn_destroy(ctx);
    if (f) fclose(f);
    free(model);
    return ret;
}
