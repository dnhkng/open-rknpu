/* SPDX-License-Identifier: MIT
 * Hardware integration check: reuse models across different application inputs.
 */
#include "open_rknpu.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc,char **argv) {
    if(argc!=3) return 2;
    unsigned models=(unsigned)strtoul(argv[2],NULL,10),runs=0,values=0;
    if(!models || models>128) return 2;
    for(unsigned i=0;i<models;i++) {
        char path[256];
        snprintf(path,sizeof(path),"%s/model%03u.bin",argv[1],i);
        ornpu_model *model=NULL;
        int rc=ornpu_open(path,&model);
        if(rc) { fprintf(stderr,"open model %u: %d\n",i,rc); return 1; }
        ornpu_info info;
        ornpu_get_info(model,&info);
        size_t described=0;
        for(uint32_t tensor=0;tensor<info.input_tensor_count;tensor++) {
            ornpu_tensor_info ti;
            if(ornpu_get_input_tensor(model,tensor,&ti) || ti.api_offset!=described || !ti.api_bytes) {
                fprintf(stderr,"input tensor descriptor failed\n");ornpu_close(model);return 1;
            }
            described+=ti.api_bytes;
        }
        ornpu_tensor_info invalid;
        if(described!=info.input_bytes || ornpu_get_input_tensor(model,info.input_tensor_count,&invalid)!=-EINVAL) {
            fprintf(stderr,"input tensor descriptor bounds failed\n");ornpu_close(model);return 1;
        }
        uint8_t *input=malloc(info.input_bytes);
        int8_t *output=malloc(info.output_bytes),*expected=malloc(info.output_bytes);
        if(!input || !output || !expected) { free(input);free(output);free(expected);ornpu_close(model);return 1; }
        if(ornpu_run(model,input,info.input_bytes-1,output,info.output_bytes)!=-EINVAL ||
           ornpu_run(model,input,info.input_bytes,output,info.output_bytes-1)!=-EINVAL) {
            fprintf(stderr,"buffer size check failed\n"); free(input);free(output);free(expected);ornpu_close(model); return 1;
        }
        snprintf(path,sizeof(path),"%s/input%03u.u8",argv[1],i);
        FILE *in=fopen(path,"rb");
        snprintf(path,sizeof(path),"%s/expected%03u.i8",argv[1],i);
        FILE *gold=fopen(path,"rb");
        if(!in || !gold) { if(in) fclose(in); if(gold) fclose(gold); free(input);free(output);free(expected);ornpu_close(model); return 1; }
        unsigned model_runs=0;
        for(;;) {
            size_t n=fread(input,1,info.input_bytes,in);
            if(!n && feof(in)) break;
            if(n!=info.input_bytes || fread(expected,1,info.output_bytes,gold)!=info.output_bytes) { rc=-EINVAL; break; }
            rc=ornpu_run(model,input,info.input_bytes,output,info.output_bytes);
            if(rc) break;
            for(unsigned j=0;j<info.output_bytes;j++) if(output[j]!=expected[j]) {
                fprintf(stderr,"model %u run %u output %u: got %d expected %d\n",i,model_runs,j,output[j],expected[j]);
                rc=-ERANGE; break;
            }
            if(rc) break;
            model_runs++; runs++; values+=info.output_bytes;
        }
        if(!model_runs || fgetc(gold)!=EOF || ferror(in) || ferror(gold)) rc=-EINVAL;
        fclose(in); fclose(gold); free(input);free(output);free(expected);ornpu_close(model);
        if(rc) { fprintf(stderr,"model %u failed: %d\n",i,rc); return 1; }
        printf("model %u: %u inputs passed\n",i,model_runs);
        fflush(stdout);
    }
    printf("PASS: %u models, %u inferences, %u output bytes; invalid buffer sizes rejected\n",models,runs,values);
    return 0;
}
