/* SPDX-License-Identifier: MIT */
#include "../runtime/open_rknpu.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void *read_exact(const char *path,size_t size) {
    FILE *f=fopen(path,"rb");void *p=malloc(size);if(!f||!p) return NULL;
    int ok=fread(p,1,size,f)==size&&fgetc(f)==EOF&&!ferror(f);fclose(f);if(!ok){free(p);return NULL;}return p;
}
int main(int argc,char **argv) {
    if(argc!=5)return 2;
    ornpu_model *m=NULL;ornpu_info info;ornpu_constant_info c;
    if(ornpu_open(argv[1],&m)||ornpu_get_info(m,&info)||info.constant_count!=1||ornpu_get_constant(m,0,&c)||(c.kind!=1&&c.kind!=3)){ornpu_close(m);return 3;}
    void *parameters=read_exact(argv[2],c.bytes),*input=read_exact(argv[3],info.input_bytes);int8_t *output=malloc(info.output_bytes);int rc=4;
    if(parameters&&input&&output&&ornpu_get_constant(m,1,&c)==-EINVAL&&ornpu_set_constant(m,0,parameters,c.bytes-1)==-EINVAL
       &&!ornpu_set_constant(m,0,parameters,c.bytes)&&!ornpu_run(m,input,info.input_bytes,output,info.output_bytes)) {
        FILE *f=fopen(argv[4],"wb");if(f&&fwrite(output,1,info.output_bytes,f)==info.output_bytes&&!fclose(f))rc=0;else if(f)fclose(f);
    }
    free(parameters);free(input);free(output);ornpu_close(m);return rc;
}
