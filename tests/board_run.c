/* SPDX-License-Identifier: MIT
 * Run one model over a file of inputs and write raw outputs (debug harness).
 */
#include "open_rknpu.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc,char **argv) {
    if(argc!=4) return 2;
    ornpu_model *model=NULL; ornpu_info info;
    if(ornpu_open(argv[1],&model) || ornpu_get_info(model,&info)) { fprintf(stderr,"open failed\n"); return 1; }
    uint8_t *input=malloc(info.input_bytes); int8_t *output=malloc(info.output_bytes);
    FILE *in=fopen(argv[2],"rb"),*out=fopen(argv[3],"wb");
    if(!in||!out||!input||!output) return 1;
    unsigned runs=0,rc=0;
    for(;;){
        size_t n=fread(input,1,info.input_bytes,in);
        if(!n&&feof(in))break;
        if(n!=info.input_bytes){rc=1;break;}
        if(ornpu_run(model,input,info.input_bytes,output,info.output_bytes)){rc=1;break;}
        if(fwrite(output,1,info.output_bytes,out)!=info.output_bytes){rc=1;break;}
        runs++;
    }
    fclose(in);fclose(out);free(input);free(output);ornpu_close(model);
    printf("runs=%u rc=%u\n",runs,rc);return rc?1:0;
}
