/* SPDX-License-Identifier: MIT
 * Board check for format v5 named-tensor executables (multi-input/multi-output).
 * Streams packed NHWC inputs from inputNNN.u8 and compares every external
 * output, concatenated in tensor-index order, against expectedNNN.i8.
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
        if(ornpu_get_info(model,&info) || !info.tensor_count) {
            fprintf(stderr,"model %u is not a named-tensor executable\n",i);ornpu_close(model);return 1;
        }
        unsigned ni=info.input_tensor_count,no=info.output_tensor_count;
        ornpu_io *ins=calloc(ni,sizeof(*ins)),*outs=calloc(no,sizeof(*outs));
        size_t in_total=0,out_total=0;
        for(unsigned t=0;t<info.tensor_count;t++) {
            ornpu_tensor_info ti;
            if(ornpu_get_tensor(model,t,&ti)) { fprintf(stderr,"tensor descriptor %u failed\n",t);return 1; }
            if(ti.role==ORNPU_TENSOR_INPUT) in_total+=ti.api_bytes;
            else if(ti.role==ORNPU_TENSOR_OUTPUT) out_total+=ti.api_bytes;
        }
        uint8_t *input=malloc(in_total);
        int8_t *expected=malloc(out_total);
        int8_t *output=malloc(out_total);
        if(!ins || !outs || !input || !expected || !output) { fprintf(stderr,"allocation failed\n");return 1; }
        for(unsigned t=0;t<info.tensor_count;t++) {
            ornpu_tensor_info ti;
            ornpu_get_tensor(model,t,&ti);
            if(ti.role==ORNPU_TENSOR_INPUT && ti.index<ni) {
                ins[ti.index]=(ornpu_io){t,input+ti.api_offset,ti.api_bytes};
            } else if(ti.role==ORNPU_TENSOR_OUTPUT && ti.index<no) {
                outs[ti.index]=(ornpu_io){t,output+ti.api_offset,ti.api_bytes};
            }
        }
        /* Wrong external-buffer sizes must be rejected before any submit. */
        if(ornpu_run_io(model,ins,ni,outs,no-1)!=-EINVAL && no>1) {
            fprintf(stderr,"model %u accepted a missing output\n",i);return 1;
        }
        if(ornpu_run_io(model,ins,ni+1,outs,no)!=-EINVAL) {
            fprintf(stderr,"model %u accepted too many inputs\n",i);return 1;
        }
        snprintf(path,sizeof(path),"%s/input%03u.u8",argv[1],i);
        FILE *in=fopen(path,"rb");
        snprintf(path,sizeof(path),"%s/expected%03u.i8",argv[1],i);
        FILE *gold=fopen(path,"rb");
        if(!in || !gold) { fprintf(stderr,"missing fixtures for model %u\n",i);return 1; }
        unsigned model_runs=0;
        for(;;) {
            size_t n=fread(input,1,in_total,in);
            if(!n && feof(in)) break;
            if(n!=in_total || fread(expected,1,out_total,gold)!=out_total) { rc=-EINVAL; break; }
            rc=ornpu_run_io(model,ins,ni,outs,no);
            if(rc) break;
            for(size_t j=0;j<out_total;j++) if(output[j]!=expected[j]) {
                fprintf(stderr,"model %u run %u output %zu: got %d expected %d\n",i,model_runs,j,(int)output[j],(int)expected[j]);
                rc=-ERANGE;break;
            }
            if(rc) break;
            model_runs++;runs++;values+=(unsigned)out_total;
        }
        if(!model_runs || fgetc(gold)!=EOF || ferror(in) || ferror(gold)) rc=-EINVAL;
        fclose(in);fclose(gold);
        free(ins);free(outs);free(input);free(expected);free(output);
        ornpu_close(model);
        if(rc) { fprintf(stderr,"model %u failed: %d\n",i,rc); return 1; }
        printf("model %u: %u inputs passed (%u outputs)\n",i,model_runs,no);
        fflush(stdout);
    }
    printf("PASS: %u v5 models, %u inferences, %u exact output bytes; invalid IO rejected\n",models,runs,values);
    return 0;
}
