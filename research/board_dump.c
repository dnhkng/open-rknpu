/* SPDX-License-Identifier: MIT
 * Execute one input and save the raw logical output for register experiments.
 */
#include "open_rknpu.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc,char **argv) {
    if(argc!=4) return 2;
    ornpu_model *model=NULL;
    int rc=ornpu_open(argv[1],&model);
    if(rc) return 1;
    ornpu_info info;
    ornpu_get_info(model,&info);
    uint8_t *input=malloc(info.input_bytes);
    int8_t *output=malloc(info.output_bytes);
    FILE *in=fopen(argv[2],"rb");
    if(!input || !output || !in || fread(input,1,info.input_bytes,in)!=info.input_bytes) rc=1;
    if(in) fclose(in);
    if(!rc) rc=ornpu_run(model,input,info.input_bytes,output,info.output_bytes);
    FILE *out=rc?NULL:fopen(argv[3],"wb");
    if(!out || fwrite(output,1,info.output_bytes,out)!=info.output_bytes) rc=1;
    if(out) fclose(out);
    free(input);free(output);ornpu_close(model);
    return rc?1:0;
}
