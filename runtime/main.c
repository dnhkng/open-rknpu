/* SPDX-License-Identifier: MIT */
#include "open_rknpu.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int report(int rc,const char *operation) {
    if(rc) fprintf(stderr,"%s: %s\n",operation,strerror(-rc));
    return rc;
}

int main(int argc,char **argv) {
    ornpu_info info;
    if(argc==3 && !strcmp(argv[1],"--inspect")) {
        if(report(ornpu_inspect(argv[2],&info),"inspect")) return 1;
        printf("input: UINT8 NHWC [%u,%u,%u,%u], %u bytes; scale=%.9g zero_point=%u\n",info.batch,info.height,info.width,info.input_channels,info.input_bytes,info.input_scale,info.input_zero_point);
        printf("output: INT8 NHWC [%u,%u,%u,%u], %u bytes; scale=%.9g zero_point=%d\n",info.batch,info.output_height,info.output_width,info.output_channels,info.output_bytes,info.output_scale,info.output_zero_point);
        return 0;
    }
    if(argc!=4) {
        fprintf(stderr,"usage: open-rknpu-run MODEL.bin INPUT.u8 OUTPUT.i8\n       open-rknpu-run --inspect MODEL.bin\n");
        return 2;
    }
    if(report(ornpu_inspect(argv[1],&info),"inspect")) return 1;
    unsigned char input[32*32*32];
    int8_t output[32*32*64];
    if(info.input_bytes>sizeof(input) || info.output_bytes>sizeof(output)) {
        fprintf(stderr,"model exceeds runner buffer limits\n"); return 1;
    }
    FILE *f=fopen(argv[2],"rb");
    if(!f) { perror("input"); return 1; }
    int valid=fread(input,1,info.input_bytes,f)==info.input_bytes && fgetc(f)==EOF && !ferror(f);
    fclose(f);
    if(!valid) { fprintf(stderr,"input must contain exactly %u bytes\n",info.input_bytes); return 1; }
    ornpu_model *model=NULL;
    if(report(ornpu_open(argv[1],&model),"open model")) return 1;
    int rc=ornpu_run(model,input,info.input_bytes,output,info.output_bytes);
    ornpu_close(model);
    if(report(rc,"inference")) return 1;
    f=fopen(argv[3],"wb");
    if(!f) { perror("output"); return 1; }
    valid=fwrite(output,1,info.output_bytes,f)==info.output_bytes;
    if(fclose(f)) valid=0;
    if(!valid) { fprintf(stderr,"output write failed\n"); return 1; }
    return 0;
}
