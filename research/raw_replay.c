/* SPDX-License-Identifier: MIT
 * Development ioctl harness for captured or independently emitted programs.
 * Optional two-task mode is for the chain4 capture at offsets 0 and 0x440.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

struct allocation { uint32_t handle, flags; uint64_t size, object, dma, sram; };
struct release { uint32_t handle, reserved; uint64_t object; };
struct sync { uint32_t flags, reserved; uint64_t object, offset, size; };
struct task { uint32_t flags, op, enable, mask, clear, status, amount, offset; uint64_t command; } __attribute__((packed));
struct submit { uint32_t flags, timeout, start, number, counter; int32_t priority; uint64_t task_object, config_object, base, user; uint32_t core; int32_t fence; uint32_t subtasks[10]; };
_Static_assert(sizeof(struct allocation)==40, "allocation ABI");
_Static_assert(sizeof(struct task)==40, "task ABI");
_Static_assert(sizeof(struct submit)==104, "submit ABI");
#define CREATE _IOWR('r',2,struct allocation)
#define DESTROY _IOWR('r',4,struct release)
#define SYNC _IOWR('r',5,struct sync)
#define SUBMIT _IOWR('r',1,struct submit)

static int sync_mem(int fd, struct allocation *a, unsigned flags) {
    struct sync s = {flags, 0, a->object, 0, a->size};
    return ioctl(fd, SYNC, &s);
}

int main(int argc, char **argv) {
    int fd=-1, rc=1;
    unsigned displacement = argc >= 3 && !strcmp(argv[2], "--relocate") ? 4096 : 0;
    unsigned height=argc==5 ? (unsigned)atoi(argv[3]) : 8;
    unsigned width=argc==5 ? (unsigned)atoi(argv[4]) : 8;
    unsigned native14=getenv("OPEN_NPU_NATIVE14")!=NULL;
    if(native14) height=width=14;
    unsigned input_channels=native14?1:3;
    unsigned payload_size=native14?16384:8192;
    struct allocation a[2] = {{.flags=10,.size=4096}, {.flags=2,.size=(native14?32768:16384)+displacement}};
    void *map[2] = {MAP_FAILED,MAP_FAILED};
    unsigned allocated=0;
    FILE *f=NULL;
    setbuf(stdout,NULL);
    if ((argc!=2 && !((argc==3 || argc==5) && displacement)) || height<5 || height>(native14?14u:8u) || width<5 || width>(native14?14u:8u)) { fprintf(stderr,"usage: raw_replay commands.bin [--relocate [HEIGHT WIDTH]]\n"); return 2; }
#define TRY(x) do { if ((x)<0) { perror(#x); goto done; } } while(0)
    fd=open("/dev/rknpu",O_RDWR|O_CLOEXEC);
    TRY(fd);
    for(unsigned i=0;i<2;i++) {
        TRY(ioctl(fd,CREATE,&a[i])); allocated++;
        map[i]=mmap(NULL,a[i].size,PROT_READ|PROT_WRITE,MAP_SHARED,a[i].handle,0);
        if(map[i]==MAP_FAILED) { perror("mmap"); goto done; }
        memset(map[i],0,a[i].size);
        printf("RAW allocation %u size=%llu dma=%llx\n",i,(unsigned long long)a[i].size,(unsigned long long)a[i].dma);
    }
    f=fopen(argv[1],"rb");
    if(!f) { perror("fixture"); goto done; }
    unsigned char *base=(unsigned char *)map[1]+displacement;
    if(fread(base,1,payload_size,f)!=payload_size || fgetc(f)!=EOF) { fprintf(stderr,"incorrect fixture length\n"); goto done; }
    fclose(f); f=NULL;
    /* Captured input/output offsets are 0x2000 and 0x3000 relative to base.
       This single arena preserves those offsets under arbitrary relocation. */
    *(struct task *)map[0]=(struct task){.op=1,.enable=29,.mask=768,.clear=131071,.amount=126,.command=a[1].dma+displacement};
    struct submit s={.flags=5,.timeout=1000,.number=1,.task_object=a[0].object,.base=a[1].dma+displacement};
    const char *task_text=getenv("OPEN_NPU_TASKS");
    if(task_text) {
        if(strcmp(task_text,"1") && strcmp(task_text,"2")) { fprintf(stderr,"unsupported task count\n"); goto done; }
        if(!strcmp(task_text,"2")) {
            ((struct task *)map[0])[1]=(struct task){.op=2,.enable=29,.mask=768,.clear=131071,.amount=126,.command=s.base+0x440};
            s.number=2;
        }
    }
    unsigned pooling=getenv("OPEN_NPU_POOL")!=NULL;
    if(pooling) {
        if(s.number!=2 || height!=8 || width!=8) { fprintf(stderr,"pool probe requires two tasks and 8x8 input\n"); goto done; }
        struct task *second=&((struct task *)map[0])[1];
        second->enable=96; second->mask=3072; second->amount=37;
    }
    unsigned pool_levels=getenv("OPEN_NPU_POOL_LEVELS")?3:1;
    if(pool_levels==3) {
        if(!pooling) { fprintf(stderr,"pool levels require pooling mode\n"); goto done; }
        s.number=4;
        for(unsigned i=2;i<4;i++)
            ((struct task *)map[0])[i]=(struct task){.op=i+1,.enable=96,.mask=3072,.clear=131071,
                .amount=37,.command=s.base+0x440+(i-1)*0x180};
    }
    printf("RAW command base=%llx displacement=%u\n",(unsigned long long)s.base,displacement);
    unsigned char *input=base+(native14?16384:8192);
    unsigned char *output=base+(native14?24576:12288);
    const char *names[]={"zero","constant","ramp","impulse"};
    const char *extra_text=getenv("OPEN_NPU_RANDOM_CASES");
    unsigned extra=extra_text ? (unsigned)strtoul(extra_text,NULL,10) : 0;
    if(extra>256) { fprintf(stderr,"too many random cases\n"); goto done; }
    for(unsigned k=0;k<4+extra;k++) {
        memset(input,0,4096); memset(output,0,4096);
        uint32_t random_state=1103+k*0x9e3779b9u;
        for(unsigned h=0;h<height;h++) for(unsigned w=0;w<width;w++) for(unsigned c=0;c<input_channels;c++) {
            unsigned value=k==1?128:k==2?h*width*input_channels+w*input_channels+c:k==3&&h==3&&w==4&&c==(input_channels==1?0u:1u)?255:0;
            if(k>=4) { random_state=1664525u*random_state+1013904223u; value=random_state>>24; }
            input[(h*16+w)*input_channels+c]=value;
        }
        TRY(sync_mem(fd,&a[0],1));
        TRY(sync_mem(fd,&a[1],1));
        TRY(ioctl(fd,SUBMIT,&s));
        TRY(sync_mem(fd,&a[1],2));
        if(k<4) printf("OUTPUT %s ",names[k]); else printf("OUTPUT random%u ",k-4);
        for(unsigned j=0;j<height*width*16/(pooling?(pool_levels==3?64:4):1);j++) printf("%02x",output[j]);
        putchar('\n');
    }
    rc=0;
done:
    if(f) fclose(f);
    for(unsigned i=allocated;i>0;i--) {
        unsigned j=i-1;
        if(map[j]!=MAP_FAILED) munmap(map[j],a[j].size);
        struct release r={a[j].handle,0,a[j].object};
        if(ioctl(fd,DESTROY,&r)<0) perror("destroy");
        close(a[j].handle);
    }
    if(fd>=0) close(fd);
    return rc;
}
