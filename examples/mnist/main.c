/* SPDX-License-Identifier: MIT
 * Model-specific hybrid MNIST example. Activations are NHWC at the NPU boundary.
 */
#define _POSIX_C_SOURCE 200809L
#include "open_rknpu.h"
#include "weights.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

static void suffix(const int8_t *input, float scale, int zp, int channels, float output[10]) {
    float x[14*14*8], pooled[16*4*4];
    if (channels==8) {
    for (int i=0;i<14*14*8;i++) x[i]=(input[i]-zp)*scale;
    /* Fuse Conv/Relu/MaxPool without allocating the full Conv output.
     * Flatten order is ONNX NCHW: channel, row, column. */
    for (int oc=0;oc<16;oc++) for (int py=0;py<4;py++) for (int px=0;px<4;px++) {
        float best=0;
        for (int dy=0;dy<3;dy++) for (int dx=0;dx<3;dx++) {
            int y=py*3+dy, xx=px*3+dx;
            float sum=conv_b[oc];
            for (int ic=0;ic<8;ic++) for (int ky=0;ky<5;ky++) for (int kx=0;kx<5;kx++) {
                int iy=y+ky-2, ix=xx+kx-2;
                if (iy>=0 && iy<14 && ix>=0 && ix<14)
                    sum+=x[(iy*14+ix)*8+ic]*conv_w[((oc*8+ic)*5+ky)*5+kx];
            }
            if (sum>best) best=sum;
        }
        pooled[(oc*4+py)*4+px]=best;
    }
    } else {
        for (int oc=0;oc<16;oc++) for (int py=0;py<4;py++) for (int px=0;px<4;px++) {
            int best=zp;
            for (int dy=0;dy<3;dy++) for (int dx=0;dx<3;dx++) {
                int v=input[((py*3+dy)*14+px*3+dx)*16+oc];
                if(v>best) best=v;
            }
            pooled[(oc*4+py)*4+px]=(best-zp)*scale;
        }
    }
    for (int j=0;j<10;j++) {
        float sum=dense_b[j];
        for (int k=0;k<256;k++) sum+=pooled[k]*dense_w[k*10+j];
        output[j]=sum;
    }
}

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+t.tv_nsec*1e-9;
}

int main(int argc,char **argv) {
    if (argc!=5) {
        fprintf(stderr,"usage: mnist-run PREFIX.bin INPUTS.u8 OUTPUT.f32 PREFIX_OUTPUT.i8\n");
        return 2;
    }
    ornpu_info info; ornpu_model *model=NULL;
    if (ornpu_inspect(argv[1],&info) || info.height!=28 || info.width!=28 ||
        info.input_channels!=1 || info.output_height!=14 || info.output_width!=14 || (info.output_channels!=8 && info.output_channels!=16)) {
        fprintf(stderr,"expected MNIST 28x28x1 -> 14x14x8 or x16 prefix\n"); return 1;
    }
    FILE *in=fopen(argv[2],"rb"), *out=fopen(argv[3],"wb"), *prefix=fopen(argv[4],"wb");
    if (!in || !out || !prefix) { perror("files"); return 1; }
    int rc=ornpu_open(argv[1],&model);
    if (rc) { fprintf(stderr,"NPU open: %d\n",rc); return 1; }
    uint8_t input[784]; int8_t activation[3136]; float logits[10];
    int count=0; double npu=0,cpu=0; size_t n;
    while ((n=fread(input,1,sizeof(input),in))==sizeof(input)) {
        double start=now();
        rc=ornpu_run(model,input,sizeof(input),activation,info.output_bytes);
        npu+=now()-start;
        if (rc) { fprintf(stderr,"NPU run: %d\n",rc); break; }
        start=now(); suffix(activation,info.output_scale,info.output_zero_point,info.output_channels,logits); cpu+=now()-start;
        if (fwrite(logits,sizeof(float),10,out)!=10 || fwrite(activation,1,info.output_bytes,prefix)!=info.output_bytes) { rc=1; break; }
        count++;
    }
    if (n!=0 || ferror(in) || !count) rc=1;
    ornpu_close(model);
    if (fclose(in)) rc=1;
    if (fclose(out)) rc=1;
    if (fclose(prefix)) rc=1;
    struct rusage usage; getrusage(RUSAGE_SELF,&usage);
    printf("cases=%d npu_ms=%.3f cpu_ms=%.3f total_ms=%.3f maxrss_kib=%ld\n",count,
           count?npu*1000/count:0,count?cpu*1000/count:0,count?(npu+cpu)*1000/count:0,usage.ru_maxrss);
    return rc?1:0;
}
