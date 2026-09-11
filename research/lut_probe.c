/* MIT. Development-only replay of trusted small captured programs.
 * No RKNN library. Verifies hardware-only replay against synchronized outputs.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
struct allocation { uint32_t handle,flags; uint64_t size,object,dma,sram; };
struct release { uint32_t handle,reserved; uint64_t object; };
struct sync { uint32_t flags,reserved; uint64_t object,offset,size; };
struct task {uint32_t flags,op,enable,mask,clear,status,amount,offset;uint64_t command;} __attribute__((packed));
struct submit {uint32_t flags,timeout,start,number,counter;int32_t priority;uint64_t task_object,config_object,base,user;uint32_t core;int32_t fence;uint32_t subtasks[10];};
_Static_assert(sizeof(struct task)==40,"task ABI");
_Static_assert(sizeof(struct submit)==104,"submit ABI");
#define CREATE _IOWR('r',2,struct allocation)
#define DESTROY _IOWR('r',4,struct release)
#define SYNC _IOWR('r',5,struct sync)
#define SUBMIT _IOWR('r',1,struct submit)
static int sync_mem(int fd,struct allocation *a,unsigned flags) {
    struct sync s={flags,0,a->object,0,a->size};return ioctl(fd,SYNC,&s);
}
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    FILE *f=fopen(argv[1],"rb");if(!f)return 1;
    unsigned total=0,cases=0;
    for(;;) {
        uint32_t h[5];size_t n=fread(h,1,sizeof(h),f);
        if(!n && feof(f))break;
        if(n!=sizeof(h) || !h[0] || h[0]>100 || !h[1] || h[1]>262144 || h[1]%4096 || h[2]>h[1] || h[3]>h[1]-h[2])return 2;
        uint32_t specs[100][5];
        if(fread(specs,20,h[0],f)!=h[0])return 2;
        struct allocation a[2]={{.flags=10,.size=4096},{.flags=2,.size=h[1]}};
        void *m[2]={MAP_FAILED,MAP_FAILED};unsigned allocated=0;int rc=1;
        int fd=open("/dev/rknpu",O_RDWR|O_CLOEXEC);if(fd<0)return 1;
        unsigned char *expected=malloc(h[3]);if(!expected)return 1;
        for(unsigned i=0;i<2;i++) {
            if(ioctl(fd,CREATE,&a[i])<0)goto cleanup;
            allocated++;
            m[i]=mmap(NULL,a[i].size,PROT_READ|PROT_WRITE,MAP_SHARED,a[i].handle,0);
            if(m[i]==MAP_FAILED)goto cleanup;
            memset(m[i],0,a[i].size);
        }
        if(fread(m[1],1,h[1],f)!=h[1] || fread(expected,1,h[3],f)!=h[3])goto cleanup;
        /* Clear output to prevent matching a stale previous inference. */
        memset((char *)m[1]+h[2],0,h[3]);
        for(unsigned i=0;i<h[0];i++) {
            if(specs[i][0]>=h[1] || specs[i][1]>4096)goto cleanup;
            ((struct task *)m[0])[i]=(struct task){.op=specs[i][4],.enable=specs[i][2],.mask=specs[i][3],.clear=131071,.amount=specs[i][1],.command=a[1].dma+specs[i][0]};
        }
        struct submit s={.flags=h[4],.timeout=1000,.number=h[0],.task_object=a[0].object,.base=a[1].dma};
        if(sync_mem(fd,&a[0],1)<0 || sync_mem(fd,&a[1],1)<0 || ioctl(fd,SUBMIT,&s)<0 || sync_mem(fd,&a[1],2)<0) {perror("submit/sync");goto cleanup;}
        unsigned mismatch=0;
        for(unsigned j=0;j<h[3];j++)if(j%16<3 && ((unsigned char *)m[1])[h[2]+j]!=expected[j]) { if(mismatch<3)printf("offset=%u got=%d expected=%d\n",j,(int)(int8_t)((unsigned char *)m[1])[h[2]+j],(int)(int8_t)expected[j]); mismatch++; }
        printf("case=%u bytes=%u mismatches=%u\n",cases,h[3],mismatch);
        if(mismatch)goto cleanup;
        total+=h[3]/16*3;cases++;rc=0;
cleanup:
        for(unsigned i=0;i<allocated;i++) {
            if(m[i]!=MAP_FAILED)munmap(m[i],a[i].size);
            struct release r={a[i].handle,0,a[i].object};ioctl(fd,DESTROY,&r);close(a[i].handle);
        }
        close(fd);free(expected);
        if(rc){fclose(f);return 1;}
    }
    fclose(f);printf("PASS cases=%u exact_bytes=%u\n",cases,total);return cases?0:1;
}
