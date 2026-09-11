/* SPDX-License-Identifier: MIT
 * Dump the concatenated external outputs of a v5 model over a file of inputs.
 * Debug harness for board mismatches: `board_dump <model.bin> <inputs.u8> <out.i8>`.
 */
#include "open_rknpu.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc,char **argv) {
    if(argc!=4) return 2;
    ornpu_model *model=NULL;
    if(ornpu_open(argv[1],&model)) { fprintf(stderr,"open failed\n"); return 1; }
    ornpu_info info;
    if(ornpu_get_info(model,&info) || !info.tensor_count) { fprintf(stderr,"not a named-tensor executable\n"); return 1; }
    unsigned ni=info.input_tensor_count,no=info.output_tensor_count;
    ornpu_io *ins=calloc(ni,sizeof(*ins)),*outs=calloc(no,sizeof(*outs));
    size_t in_total=0,out_total=0;
    for(unsigned t=0;t<info.tensor_count;t++) {
        ornpu_tensor_info ti;
        if(ornpu_get_tensor(model,t,&ti)) { fprintf(stderr,"tensor %u failed\n",t); return 1; }
        if(ti.role==ORNPU_TENSOR_INPUT) in_total+=ti.api_bytes;
        else if(ti.role==ORNPU_TENSOR_OUTPUT) out_total+=ti.api_bytes;
    }
    uint8_t *input=malloc(in_total); int8_t *output=malloc(out_total);
    if(!ins||!outs||!input||!output) { fprintf(stderr,"allocation failed\n"); return 1; }
    for(unsigned t=0;t<info.tensor_count;t++) {
        ornpu_tensor_info ti;
        ornpu_get_tensor(model,t,&ti);
        if(ti.role==ORNPU_TENSOR_INPUT && ti.index<ni) ins[ti.index]=(ornpu_io){t,input+ti.api_offset,ti.api_bytes};
        else if(ti.role==ORNPU_TENSOR_OUTPUT && ti.index<no) outs[ti.index]=(ornpu_io){t,output+ti.api_offset,ti.api_bytes};
    }
    FILE *in=fopen(argv[2],"rb"),*out=fopen(argv[3],"wb");
    if(!in||!out) { fprintf(stderr,"fixture open failed\n"); return 1; }
    unsigned runs=0;
    for(;;) {
        size_t n=fread(input,1,in_total,in);
        if(!n && feof(in)) break;
        if(n!=in_total) { fprintf(stderr,"short input\n"); return 1; }
        int rc=ornpu_run_io(model,ins,ni,outs,no);
        if(rc) { fprintf(stderr,"run %u failed: %d\n",runs,rc); return 1; }
        if(fwrite(output,1,out_total,out)!=out_total) { fprintf(stderr,"short output\n"); return 1; }
        runs++;
    }
    fclose(in);fclose(out);free(ins);free(outs);free(input);free(output);ornpu_close(model);
    printf("runs=%u outputs=%zu\n",runs,out_total);
    return 0;
}
