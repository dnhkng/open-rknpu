/* SPDX-License-Identifier: MIT
 * RV1103 runtime. Compiler-generated register payloads are trusted code.
 * Container validation detects incompatible/corrupted files, not hostile ISA.
 *
 * Formats: legacy ORNPUBIN v1/v2 profiles, ORNPUSEQ v3/v4 task table, and
 * ORNPUSEQ v5 named-tensor table (fan-out, several external inputs/outputs).
 */
#include "open_rknpu.h"
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "This runtime currently requires a little-endian host"
#endif

struct header {
    char magic[8];
    uint32_t version, header_size, target, profile, height, width;
    uint32_t input_channels, output_channels, input_stride, output_stride;
    uint32_t payload_size, register_count, input_offset, output_offset, arena_size;
    uint32_t kernel, relu;
    int32_t output_zero_point;
    float output_scale;
    uint32_t checksum, reserved[2];
};
struct allocation { uint32_t handle, flags; uint64_t size, object, dma, sram; };
struct release { uint32_t handle, reserved; uint64_t object; };
struct sync { uint32_t flags, reserved; uint64_t object, offset, size; };
struct task { uint32_t flags, op, enable, mask, clear, status, amount, offset; uint64_t command; } __attribute__((packed));
struct submit { uint32_t flags, timeout, start, number, counter; int32_t priority; uint64_t task_object, config_object, base, user; uint32_t core; int32_t fence; uint32_t subtasks[10]; };
_Static_assert(sizeof(struct header)==96,"model ABI");
_Static_assert(offsetof(struct header,checksum)==84,"checksum ABI");
_Static_assert(sizeof(struct allocation)==40,"allocation ABI");
_Static_assert(sizeof(struct task)==40,"task ABI");
_Static_assert(sizeof(struct submit)==104,"submit ABI");
#define CREATE _IOWR('r',2,struct allocation)
#define DESTROY _IOWR('r',4,struct release)
#define SYNC _IOWR('r',5,struct sync)
#define SUBMIT _IOWR('r',1,struct submit)

#define MAX_TENSORS 64
#define MAX_CONSTANTS 64
#define MAX_TASKS 64
#define ROLE_INPUT 0u
#define ROLE_OUTPUT 1u
#define ROLE_INTERNAL 2u
#define LAYOUT_PACKED 0u
#define LAYOUT_NATIVE16 1u
#define LAYOUT_PACKED_I8 2u

struct task_spec { uint32_t offset, amount, enable, mask; };
struct constant_spec { char name[24]; uint32_t offset, size, kind, reserved; };
struct tensor_spec {
    char name[24];
    uint32_t role, layout, index, batch, height, width, channels, offset, size, reserved;
} __attribute__((packed));
_Static_assert(sizeof(struct constant_spec)==40,"constant ABI");
_Static_assert(sizeof(struct tensor_spec)==64,"tensor ABI");

struct ornpu_model {
    struct header header;
    ornpu_info info;
    unsigned task_count, constant_count, serial, submit_flags;
    unsigned tensor_count, input_count, output_count;
    /* Experimental job-field probe (S8): the submission's core_mask and the five
     * subcore_task[] windows. Both default to zero. */
    uint32_t core_mask, subcore[10];
    /* Engine runs (S10): a non-serial container is submitted as one ioctl per maximal
     * linked run, derived at load time from the task tails. A serial container keeps
     * one run per task (its tails are all terminal). */
    unsigned run_count;
    unsigned run_start[MAX_TASKS], run_tasks[MAX_TASKS];
    struct constant_spec *constants;
    struct tensor_spec *tensors;
    int fd;
    unsigned allocated;
    struct allocation buffers[2];
    void *mapping[2];
};

static uint32_t hash_bytes(uint32_t hash,const void *buffer,size_t size) {
    const unsigned char *bytes=buffer;
    for(size_t i=0;i<size;i++) hash=(hash^bytes[i])*16777619u;
    return hash;
}

static uint64_t tensor_arena_bytes(uint32_t layout,uint32_t batch,uint32_t height,uint32_t width,uint32_t channels) {
    if(layout==LAYOUT_NATIVE16) return (uint64_t)batch*((height*width+3u)/4u)*64u*((channels+15u)/16u);
    if(layout==LAYOUT_PACKED || layout==LAYOUT_PACKED_I8)
        return (uint64_t)batch*height*((width+15u)/16u*16u)*channels;
    return 0;
}

static int read_model(const char *path,struct header *h,uint8_t *payload) {
    FILE *f=fopen(path,"rb");
    if(!f) return -errno;
    int rc=-EINVAL;
    if(fread(h,1,sizeof(*h),f)!=sizeof(*h)) goto done;
    if(memcmp(h->magic,"ORNPUBIN",8) || (h->version!=1 && h->version!=2) || h->header_size!=96 ||
       h->target!=1103 || h->profile<1 || h->profile>8 || h->height<5 || h->height>(h->input_channels==1 && h->profile==1?32u:8u) ||
       h->width<5 || h->width>(h->input_channels==1 && h->profile==1?32u:8u) || (h->input_channels!=1 && h->input_channels!=3) || h->output_channels<1 || h->output_channels>16 ||
       h->input_stride!=((h->width+15)/16*16) || h->output_stride!=16 || h->payload_size!=8192 ||
       h->register_count!=126 || h->input_offset!=8192 || h->output_offset!=12288 ||
       h->arena_size!=12288+((h->height*h->width*16+4095)/4096)*4096 || (h->kernel!=1 && h->kernel!=3 && h->kernel!=5) || h->relu>1 ||
       h->output_zero_point < -128 || h->output_zero_point > 127 ||
       !isfinite(h->output_scale) || h->output_scale<=0) goto done;
    if(h->version==1 && (h->reserved[0] || h->reserved[1])) goto done;
    if(h->version==2) {
        float scale;
        memcpy(&scale,&h->reserved[0],sizeof(scale));
        if(h->profile!=1 || !isfinite(scale) || scale<=0 || h->reserved[1]>255) goto done;
    }
    if(h->profile!=1 && h->input_channels!=3) goto done;
    if((h->profile==2 || h->profile>=7) && (h->height!=8 || h->width!=8 || h->output_channels!=3 ||
                         h->relu!=1 || h->kernel==5)) goto done;
    if(h->profile>=3 && h->profile<=6 && (h->height!=8 || h->width!=8 || h->output_channels!=3 || h->kernel!=1)) goto done;
    if(fread(payload,1,8192,f)!=8192 || fgetc(f)!=EOF || ferror(f)) goto done;
    uint32_t expected=h->checksum;
    h->checksum=0;
    uint32_t actual=hash_bytes(hash_bytes(2166136261u,h,sizeof(*h)),payload,8192);
    h->checksum=expected;
    if(actual!=expected) goto done;
    rc=0;
done:
    fclose(f);
    return rc;
}

static void fill_info(const struct header *h,ornpu_info *info) {
    unsigned out_h=h->profile>=5?1:h->profile>=3?4:h->height;
    unsigned out_w=h->profile>=5?1:h->profile>=3?4:h->width;
    *info=(ornpu_info){h->height,h->width,h->input_channels,h->output_channels,
                     h->height*h->width*h->input_channels,out_h*out_w*h->output_channels,
                     h->output_scale,h->output_zero_point,out_h,out_w,1.0f,0,1,1,0,0,0,0,0,0};
    if(h->version==2) {
        memcpy(&info->input_scale,&h->reserved[0],sizeof(float));
        info->input_zero_point=h->reserved[1];
    }
}

/* Parse a v5 named-tensor table into h/info/tensors. */
static int load_v5(FILE *f,const uint32_t *v,struct header *h,uint8_t **payload,
                   struct task_spec *tasks,struct tensor_spec *tensors,
                   unsigned *count,unsigned *tensor_count,unsigned *serial,ornpu_info *info) {
    int rc=-EINVAL;
    uint32_t extension[4];
    if(fread(extension,sizeof(extension),1,f)!=1) return rc;
    uint32_t tensors_n=extension[0],tensor_size=extension[1],input_count=extension[2],output_count=extension[3];
    unsigned flags=v[19]&255;
    /* v[1] is the declared header size and every field above is read at its fixed v5
       offset, so a contradictory declaration is a malformed container, not a variant.
       The Python decoder has always rejected it; the loader must agree. */
    if(v[1]!=96 || tensor_size!=64 || tensors_n<1 || tensors_n>MAX_TENSORS || input_count<1 || input_count>8 ||
       output_count<1 || output_count>8 || flags>1 || (v[19]>>8)!=0) return rc;
    if(v[13]<1 || v[13]>MAX_TASKS) return rc;
    uint32_t batch=v[21]+1;
    if(batch<1 || batch>16) return rc;
    uint32_t in_layout=v[20];
    if(in_layout>1) return rc;
    uint32_t stride=in_layout?v[3]:(v[3]+15)/16*16;
    if(v[8]!=stride) return rc;
    if(!(v[2]>=1 && v[2]<=1024 && v[3]>=1 && v[3]<=1024 && v[5]>=1 && v[5]<=1024 && v[6]>=1 && v[6]<=1024)) return rc;
    if(fread(tasks,16,v[13],f)!=v[13]) return rc;
    for(unsigned i=0;i<v[13];i++) {
        struct task_spec *t=&tasks[i];
        /* 0x300|0xc00 is the union mask a batched job needs when it mixes engines. */
        unsigned valid_kind=(((t->enable==29 || t->enable==96) &&
                              (t->mask==768 || t->mask==3072 || t->mask==3840)) && t->amount<=256) ||
                            (t->enable==24 && (t->mask==768 || t->mask==3840) &&
                             (t->amount==78 || t->amount==1106));
        if(t->offset%8 || !t->amount || t->amount>1106 ||
           (uint64_t)t->offset+(t->amount+4)*8>v[9] || !valid_kind) return rc;
    }
    if(fread(tensors,sizeof(*tensors),tensors_n,f)!=tensors_n) return rc;
    uint32_t inputs_seen=0,outputs_seen=0;
    for(unsigned i=0;i<tensors_n;i++) {
        struct tensor_spec *t=&tensors[i];
        if(!t->name[0] || !memchr(t->name,0,sizeof(t->name)) || t->role>ROLE_INTERNAL || t->layout>LAYOUT_PACKED_I8 ||
           t->batch<1 || t->batch>16 || t->height<1 || t->height>1024 || t->width<1 || t->width>1024 ||
           t->channels<1 || t->channels>128 || (t->role==ROLE_INPUT && t->layout==LAYOUT_NATIVE16 && t->channels>128)) return rc;
        uint64_t expected=tensor_arena_bytes(t->layout,t->batch,t->height,t->width,t->channels);
        if(!expected || t->size!=expected) return rc;
        if(t->offset<v[9] || (uint64_t)t->offset+t->size>v[10]) return rc;
        for(unsigned j=0;j<i;j++) if(!strncmp(t->name,tensors[j].name,sizeof(t->name))) return rc;
        if(t->role==ROLE_INPUT) inputs_seen++;
        else if(t->role==ROLE_OUTPUT) outputs_seen++;
        if(t->role!=ROLE_INTERNAL) {
            for(unsigned j=0;j<i;j++) if(tensors[j].role!=ROLE_INTERNAL &&
                t->offset<tensors[j].offset+tensors[j].size && tensors[j].offset<t->offset+t->size) return rc;
        } else {
            for(unsigned j=0;j<tensors_n;j++) if(tensors[j].role!=ROLE_INTERNAL &&
                t->offset<tensors[j].offset+tensors[j].size && tensors[j].offset<t->offset+t->size) return rc;
        }
    }
    if(inputs_seen!=input_count || outputs_seen!=output_count) return rc;
    /* contiguous per-role indices */
    for(uint32_t want=0;want<input_count;want++) {
        unsigned found=0;
        for(unsigned i=0;i<tensors_n;i++) if(tensors[i].role==ROLE_INPUT && tensors[i].index==want) found++;
        if(found!=1) return rc;
    }
    for(uint32_t want=0;want<output_count;want++) {
        unsigned found=0;
        for(unsigned i=0;i<tensors_n;i++) if(tensors[i].role==ROLE_OUTPUT && tensors[i].index==want) found++;
        if(found!=1) return rc;
    }
    const struct tensor_spec *primary_in=NULL,*primary_out=NULL;
    for(unsigned i=0;i<tensors_n;i++) {
        if(tensors[i].role==ROLE_INPUT && tensors[i].index==0) primary_in=&tensors[i];
        if(tensors[i].role==ROLE_OUTPUT && tensors[i].index==0) primary_out=&tensors[i];
    }
    if(!primary_in || !primary_out) return rc;
    if(primary_in->height!=v[2] || primary_in->width!=v[3] || primary_in->channels!=v[4] ||
       primary_in->batch!=batch || primary_in->offset!=v[11] ||
       primary_out->height!=v[5] || primary_out->width!=v[6] || primary_out->channels!=v[7] ||
       primary_out->offset!=v[12]) return rc;
    uint64_t payload_size=v[9],arena=v[10];
    if(!payload_size || payload_size>1048576 || payload_size%64 || !arena || arena>4194304 || arena%4096) return rc;
    if(v[11]%64 || v[12]%64 || v[11]<payload_size || v[12]<payload_size) return rc;
    float inscale,outscale;int32_t outzp;
    memcpy(&inscale,&v[14],4);memcpy(&outscale,&v[16],4);memcpy(&outzp,&v[17],4);
    if(!isfinite(inscale) || inscale<=0 || v[15]>255 ||
       !isfinite(outscale) || outscale<=0 || outzp < -128 || outzp>127) return rc;
    *payload=malloc(payload_size);
    if(!*payload) return -ENOMEM;
    if(fread(*payload,1,payload_size,f)!=payload_size || fgetc(f)!=EOF || ferror(f)) return rc;
    /* Hash exactly the on-disk prefix: 96-byte header (checksum zeroed), 16-byte
       extension, task table, tensor table, then payload. */
    uint32_t saved=v[18];
    uint32_t hash=2166136261u;
    uint8_t header_copy[96];
    memcpy(header_copy,"ORNPUSEQ",8);
    memcpy(header_copy+8,v,88);
    memset(header_copy+80,0,4);
    hash=hash_bytes(hash,header_copy,96);
    hash=hash_bytes(hash,extension,sizeof(extension));
    hash=hash_bytes(hash,tasks,v[13]*16);
    hash=hash_bytes(hash,tensors,tensors_n*sizeof(*tensors));
    hash=hash_bytes(hash,*payload,payload_size);
    if(hash!=saved) return rc;
    memset(h,0,sizeof(*h));
    h->profile=primary_in->layout==LAYOUT_NATIVE16?9:0;h->version=2;
    h->height=primary_in->height;h->width=primary_in->width;h->input_channels=primary_in->channels;
    h->output_channels=primary_out->channels;h->input_stride=stride;h->output_stride=16;
    h->payload_size=payload_size;h->arena_size=arena;h->input_offset=v[11];h->output_offset=v[12];
    h->reserved[0]=v[14];h->reserved[1]=v[15];
    *info=(ornpu_info){primary_in->height,primary_in->width,primary_in->channels,primary_out->channels,
                       (uint32_t)(batch*(uint64_t)primary_in->height*primary_in->width*primary_in->channels),
                       (uint32_t)(batch*(uint64_t)primary_out->height*primary_out->width*primary_out->channels),
                       outscale,outzp,primary_out->height,primary_out->width,inscale,v[15],batch,input_count,0,
                       tensors_n,output_count,v[13],flags&1,0};
    *count=v[13];*tensor_count=tensors_n;*serial=flags&1;
    return 0;
}

/* Load either legacy profiles or an explicit compiler-generated task table. */
static int load_program(const char *path,struct header *h,uint8_t **payload,
                        struct task_spec *tasks,struct constant_spec *constants,unsigned *count,
                        unsigned *constant_count,unsigned *serial,ornpu_info *info,
                        struct tensor_spec *tensors,unsigned *tensor_count) {
    *serial=0;*constant_count=0;*tensor_count=0;
    FILE *f=fopen(path,"rb");
    if(!f) return -errno;
    uint8_t raw[96];
    if(fread(raw,1,96,f)!=96) { fclose(f); return -EINVAL; }
    if(memcmp(raw,"ORNPUSEQ",8)) {
        fclose(f);
        *payload=malloc(8192);
        if(!*payload) return -ENOMEM;
        int rc=read_model(path,h,*payload);
        if(rc) return rc;
        fill_info(h,info);
        *count=h->profile>=7?5:h->profile>=5?4:h->profile==1?1:2;
        tasks[0]=(struct task_spec){0,126,29,768};
        for(unsigned i=1;i<*count;i++) {
            unsigned conv=i==1 && (h->profile==2 || h->profile>=7);
            unsigned offset=i==1?0x440:h->profile>=7?0x880+(i-2)*0x180:0x440+(i-1)*0x180;
            tasks[i]=(struct task_spec){offset,conv?126u:37u,conv?29u:96u,conv?768u:3072u};
        }
        info->constant_count=0;return 0;
    }
    uint32_t v[22];memcpy(v,raw+8,sizeof(v));
    if(v[0]==5) {
        int rc=load_v5(f,v,h,payload,tasks,tensors,count,tensor_count,serial,info);
        fclose(f);
        if(rc && *payload) { free(*payload);*payload=NULL; }
        return rc;
    }
    /* version, header size, input HWC, output HWC, stride, payload/arena,
       IO offsets, count, input scale/zp, output scale/zp, checksum, reserved */
    int rc=-EINVAL;
    unsigned flags=v[19]&255,constants_count=v[19]>>8;
    if((v[0]!=3 && v[0]!=4) || v[1]!=96 || flags>3 || constants_count>MAX_CONSTANTS || ((v[0]==3)!=(constants_count==0)) || v[20]>1 || v[21]>15 ||
       !v[2] || v[2]>1024 || !v[3] || v[3]>1024 || (v[20]?(!v[4] || v[4]>128):(v[4]!=1 && v[4]!=3)) ||
       !v[5] || v[5]>1024 || !v[6] || v[6]>1024 || !v[7] || v[7]>128 ||
       v[8]!=(v[20]?v[3]:(v[3]+15)/16*16) || !v[9] || v[9]>1048576 || v[9]%64 ||
       !v[10] || v[10]>4194304 || v[10]%4096 || !v[13] || v[13]>MAX_TASKS) goto done;
    uint32_t batch=v[21]+1;
    uint32_t input_count=(v[19]>>1)+1;
    if(input_count==2 && v[2]%2) goto done;
    if(batch!=1 && !v[20]) goto done;
    uint64_t input_end=(uint64_t)v[11]+batch*(v[20]?((v[2]*v[3]+3)/4)*64*((v[4]+15)/16):v[2]*v[8]*v[4]);
    uint64_t output_end=(uint64_t)v[12]+batch*((v[5]*v[6]+3)/4)*64*((v[7]+15)/16);
    if(v[11]%64 || v[12]%64 || v[11]<v[9] || v[12]<v[9] ||
       input_end>v[10] || output_end>v[10] ||
       !(input_end<=v[12] || output_end<=v[11])) goto done;
    float inscale,outscale;int32_t outzp;
    memcpy(&inscale,&v[14],4);memcpy(&outscale,&v[16],4);memcpy(&outzp,&v[17],4);
    if(!isfinite(inscale) || inscale<=0 || v[15]>255 ||
       !isfinite(outscale) || outscale<=0 || outzp < -128 || outzp>127) goto done;
    if(fread(tasks,16,v[13],f)!=v[13]) goto done;
    for(unsigned i=0;i<v[13];i++) {
        struct task_spec *t=&tasks[i];
        /* 0x300|0xc00 is the union mask a batched job needs when it mixes engines. */
        unsigned valid_kind=(((t->enable==29 || t->enable==96) &&
                              (t->mask==768 || t->mask==3072 || t->mask==3840)) && t->amount<=256) ||
                            (t->enable==24 && (t->mask==768 || t->mask==3840) &&
                             (t->amount==78 || t->amount==1106));
        if(t->offset%8 || !t->amount || t->amount>1106 ||
           (uint64_t)t->offset+(t->amount+4)*8>v[9] ||
           !valid_kind) goto done;
    }
    if(constants_count && fread(constants,sizeof(*constants),constants_count,f)!=constants_count) goto done;
    for(unsigned i=0;i<constants_count;i++) {
        struct constant_spec *c=&constants[i];
        if(!c->name[0] || !memchr(c->name,0,sizeof(c->name)) || !c->size || c->size>v[9] || c->offset>v[9]-c->size ||
           c->kind<1 || c->kind>3 || c->reserved) goto done;
        for(unsigned j=0;j<i;j++) if(!strncmp(c->name,constants[j].name,sizeof(c->name))) goto done;
        for(unsigned j=0;j<v[13];j++) {
            uint64_t command_end=(uint64_t)tasks[j].offset+(tasks[j].amount+4)*8;
            if(c->offset<command_end && tasks[j].offset<(uint64_t)c->offset+c->size) goto done;
        }
    }
    *payload=malloc(v[9]);
    if(!*payload) { rc=-ENOMEM; goto done; }
    if(fread(*payload,1,v[9],f)!=v[9] || fgetc(f)!=EOF || ferror(f)) goto done;
    memset(raw+80,0,4);
    uint32_t hash=hash_bytes(2166136261u,raw,96);
    hash=hash_bytes(hash,tasks,v[13]*16);hash=hash_bytes(hash,constants,constants_count*sizeof(*constants));hash=hash_bytes(hash,*payload,v[9]);
    if(hash!=v[18]) goto done;
    memset(h,0,sizeof(*h));
    h->profile=v[20]?9:0;h->version=2;h->height=v[2];h->width=v[3];h->input_channels=v[4];h->output_channels=v[7];
    h->input_stride=v[8];h->output_stride=16;h->payload_size=v[9];h->arena_size=v[10];
    h->input_offset=v[11];h->output_offset=v[12];h->reserved[0]=v[14];h->reserved[1]=v[15];
    *info=(ornpu_info){v[2],v[3],v[4],v[7],batch*v[2]*v[3]*v[4],batch*v[5]*v[6]*v[7],
                      outscale,outzp,v[5],v[6],inscale,v[15],batch,input_count,constants_count,0,0,v[13],flags&1,0};
    *count=v[13];*constant_count=constants_count;*serial=flags&1;rc=0;
done:
    fclose(f);return rc;
}

int ornpu_inspect(const char *path,ornpu_info *info) {
    if(!path || !info) return -EINVAL;
    struct header h;
    uint8_t *payload=NULL;struct task_spec tasks[MAX_TASKS];struct constant_spec constants[MAX_CONSTANTS];
    struct tensor_spec tensors[MAX_TENSORS];unsigned count,constant_count,serial,tensor_count;
    int rc=load_program(path,&h,&payload,tasks,constants,&count,&constant_count,&serial,info,tensors,&tensor_count);
    info->task_count=count;info->submission_serial=serial;
    free(payload);
    return rc;
}

int ornpu_get_info(const ornpu_model *model,ornpu_info *info) {
    if(!model || !info) return -EINVAL;
    *info=model->info;
    return 0;
}

static const struct tensor_spec *find_tensor(const ornpu_model *model,uint32_t role,uint32_t index) {
    for(unsigned i=0;i<model->tensor_count;i++)
        if(model->tensors[i].role==role && model->tensors[i].index==index) return &model->tensors[i];
    return NULL;
}

static void describe_tensor(const ornpu_model *model,const struct tensor_spec *t,ornpu_tensor_info *info) {
    memset(info,0,sizeof(*info));
    memcpy(info->name,t->name,sizeof(info->name));
    info->role=t->role;info->layout=t->layout;info->index=t->index;
    info->batch=t->batch;info->height=t->height;info->width=t->width;info->channels=t->channels;
    info->arena_offset=t->offset;info->arena_bytes=t->size;
    info->api_bytes=t->batch*t->height*t->width*t->channels;
    uint32_t offset=0;
    for(unsigned i=0;i<model->tensor_count;i++) {
        const struct tensor_spec *o=&model->tensors[i];
        if(o->role!=t->role || o->index>=t->index) continue;
        offset+=o->batch*o->height*o->width*o->channels;
    }
    info->api_offset=offset;
}

int ornpu_get_tensor(const ornpu_model *model,uint32_t index,ornpu_tensor_info *info) {
    if(!model || !info || index>=model->tensor_count) return -EINVAL;
    describe_tensor(model,&model->tensors[index],info);
    return 0;
}

int ornpu_get_input_tensor(const ornpu_model *model,uint32_t index,ornpu_tensor_info *info) {
    if(!model || !info) return -EINVAL;
    if(model->tensor_count) {
        const struct tensor_spec *t=find_tensor(model,ROLE_INPUT,index);
        if(!t) return -EINVAL;
        describe_tensor(model,t,info);
        return 0;
    }
    if(index>=model->info.input_tensor_count) return -EINVAL;
    uint32_t height=model->info.height/model->info.input_tensor_count;
    uint32_t bytes=model->info.batch*height*model->info.width*model->info.input_channels;
    *info=(ornpu_tensor_info){0};
    info->batch=model->info.batch;info->height=height;info->width=model->info.width;
    info->channels=model->info.input_channels;info->api_offset=index*bytes;info->api_bytes=bytes;
    info->role=ROLE_INPUT;info->index=index;info->layout=model->header.profile==9?LAYOUT_NATIVE16:LAYOUT_PACKED;
    return 0;
}

int ornpu_get_constant(const ornpu_model *model,uint32_t index,ornpu_constant_info *info) {
    if(!model || !info || index>=model->constant_count) return -EINVAL;
    const struct constant_spec *c=&model->constants[index];
    memset(info,0,sizeof(*info));memcpy(info->name,c->name,sizeof(info->name));
    info->byte_offset=c->offset;info->bytes=c->size;info->kind=c->kind;return 0;
}

int ornpu_set_constant(ornpu_model *model,uint32_t index,const void *data,size_t size) {
    if(!model || !data || index>=model->constant_count) return -EINVAL;
    const struct constant_spec *c=&model->constants[index];
    if(size!=c->size) return -EINVAL;
    memcpy((uint8_t *)model->mapping[1]+c->offset,data,size);return 0;
}

void ornpu_close(ornpu_model *model) {
    if(!model) return;
    for(unsigned i=model->allocated;i>0;i--) {
        unsigned j=i-1;
        struct allocation *a=&model->buffers[j];
        if(model->mapping[j]!=MAP_FAILED) munmap(model->mapping[j],a->size);
        struct release release={a->handle,0,a->object};
        ioctl(model->fd,DESTROY,&release);
        close(a->handle);
    }
    if(model->fd>=0) close(model->fd);
    free(model->constants);
    free(model->tensors);
    free(model);
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return (uint64_t)ts.tv_sec*1000000000ull+(uint64_t)ts.tv_nsec;
}

static int sync_buffer(ornpu_model *model,unsigned index,unsigned flags) {
    struct allocation *a=&model->buffers[index];
    struct sync s={flags,0,a->object,0,a->size};
    return ioctl(model->fd,SYNC,&s)<0 ? -errno : 0;
}

static int submit_tasks_flags(ornpu_model *model,uint32_t extra_flags,int *fence_fd) {
    struct submit submission={.flags=model->submit_flags|extra_flags,.timeout=1000,
        .number=model->task_count,.task_object=model->buffers[0].object,
        .base=model->buffers[1].dma,.core=model->core_mask};
    for(unsigned i=0;i<10;i++) submission.subtasks[i]=model->subcore[i];
    unsigned submissions=model->serial?model->task_count:model->run_count;
    /* In/out: an input fence fd for FENCE_IN, the driver's fd for FENCE_OUT. */
    submission.fence=(extra_flags&0x8u && fence_fd)?*fence_fd:0;
    if(fence_fd && !(extra_flags&0x8u)) *fence_fd=-1;
    for(unsigned i=0;i<submissions;i++) {
        if(model->serial) { submission.start=i; submission.number=1; }
        else { submission.start=model->run_start[i]; submission.number=model->run_tasks[i]; }
        if(ioctl(model->fd,SUBMIT,&submission)<0) return -errno;
    }
    if(fence_fd) *fence_fd=submission.fence;
    return 0;
}

static int submit_tasks(ornpu_model *model) {
    return submit_tasks_flags(model,0,NULL);
}

static int pack_tensor_input(ornpu_model *model,const struct tensor_spec *t,const uint8_t *src,size_t size) {
    uint64_t api=(uint64_t)t->batch*t->height*t->width*t->channels;
    if(size!=(size_t)api) return -EINVAL;
    uint8_t *base=(uint8_t *)model->mapping[1]+t->offset;
    if(t->layout==LAYOUT_NATIVE16) {
        uint32_t surface=((t->height*t->width+3)/4)*64,planes=(t->channels+15)/16;
        memset(base,(uint8_t)(model->info.input_zero_point-128),t->size);
        for(uint32_t b=0;b<t->batch;b++) for(uint32_t p=0;p<t->height*t->width;p++) for(uint32_t c=0;c<t->channels;c++)
            base[(uint64_t)b*surface*planes+(c/16)*surface+p*16+c%16]=(uint8_t)(src[((uint64_t)b*t->height*t->width+p)*t->channels+c]-128);
    } else {
        uint32_t stride=(t->width+15)/16*16;
        memset(base,model->info.input_zero_point,t->size);
        for(uint32_t b=0;b<t->batch;b++) for(uint32_t r=0;r<t->height;r++)
            memcpy(base+((uint64_t)b*t->height+r)*stride*t->channels,
                   src+((uint64_t)b*t->height+r)*t->width*t->channels,(size_t)t->width*t->channels);
    }
    return 0;
}

static int unpack_tensor_output(ornpu_model *model,const struct tensor_spec *t,int8_t *dst,size_t size) {
    uint64_t api=(uint64_t)t->batch*t->height*t->width*t->channels;
    if(size!=(size_t)api) return -EINVAL;
    const uint8_t *base=(const uint8_t *)model->mapping[1]+t->offset;
    if(t->layout==LAYOUT_NATIVE16) {
        uint32_t surface=((t->height*t->width+3)/4)*64,planes=(t->channels+15)/16;
        for(uint32_t b=0;b<t->batch;b++) for(uint32_t p=0;p<t->height*t->width;p++) for(uint32_t c=0;c<t->channels;c++)
            dst[((uint64_t)b*t->height*t->width+p)*t->channels+c]=(int8_t)base[(uint64_t)b*surface*planes+(c/16)*surface+p*16+c%16];
    } else {
        uint32_t stride=(t->width+15)/16*16;
        for(uint32_t b=0;b<t->batch;b++) for(uint32_t r=0;r<t->height;r++)
            memcpy(dst+((uint64_t)b*t->height+r)*t->width*t->channels,
                   base+((uint64_t)b*t->height+r)*stride*t->channels,(size_t)t->width*t->channels);
    }
    return 0;
}

int ornpu_run_io_timed(ornpu_model *model,const ornpu_io *inputs,uint32_t input_count,
                       ornpu_io *outputs,uint32_t output_count,struct ornpu_timing *timing) {
    uint64_t started=timing?now_ns():0,packed=0,submitted=0,read=0;
    if(!model || !inputs || !outputs) return -EINVAL;
    if(!model->tensor_count) return -EINVAL;
    if(input_count!=model->input_count || output_count!=model->output_count) return -EINVAL;
    for(uint32_t i=0;i<input_count;i++) {
        if(inputs[i].tensor_index>=model->tensor_count || !inputs[i].data) return -EINVAL;
        const struct tensor_spec *t=&model->tensors[inputs[i].tensor_index];
        if(t->role!=ROLE_INPUT) return -EINVAL;
        /* Every external tensor exactly once: a repeated index would silently drop a
         * caller buffer and leave that tensor unbound. */
        for(uint32_t j=0;j<i;j++) if(inputs[j].tensor_index==inputs[i].tensor_index) return -EINVAL;
        int rc=pack_tensor_input(model,t,inputs[i].data,inputs[i].size);
        if(rc) return rc;
    }
    for(uint32_t i=0;i<output_count;i++) {
        if(outputs[i].tensor_index>=model->tensor_count || !outputs[i].data) return -EINVAL;
        const struct tensor_spec *t=&model->tensors[outputs[i].tensor_index];
        if(t->role!=ROLE_OUTPUT) return -EINVAL;
        for(uint32_t j=0;j<i;j++) if(outputs[j].tensor_index==outputs[i].tensor_index) return -EINVAL;
        if(outputs[i].size!=(size_t)((uint64_t)t->batch*t->height*t->width*t->channels)) return -EINVAL;
        memset((uint8_t *)model->mapping[1]+t->offset,0,t->size);
    }
    packed=timing?now_ns():0;
    int rc=sync_buffer(model,0,1);
    if(rc) return rc;
    rc=sync_buffer(model,1,1);
    if(rc) return rc;
    submitted=timing?now_ns():0;
    rc=submit_tasks(model);
    if(rc) return rc;
    read=timing?now_ns():0;
    rc=sync_buffer(model,1,2);
    if(rc) return rc;
    for(uint32_t i=0;i<output_count;i++) {
        const struct tensor_spec *t=&model->tensors[outputs[i].tensor_index];
        rc=unpack_tensor_output(model,t,outputs[i].data,outputs[i].size);
        if(rc) return rc;
    }
    if(timing) {
        uint64_t done=now_ns();
        timing->pack_ns=packed-started;
        timing->submit_ns=read-submitted;
        timing->readback_ns=done-read;
        timing->total_ns=done-started;
    }
    return 0;
}

int ornpu_run_io(ornpu_model *model,const ornpu_io *inputs,uint32_t input_count,
                 ornpu_io *outputs,uint32_t output_count) {
    return ornpu_run_io_timed(model,inputs,input_count,outputs,output_count,NULL);
}

int ornpu_run_timed(ornpu_model *model,const uint8_t *input,size_t input_size,
                    int8_t *output,size_t output_size,struct ornpu_timing *timing) {
    if(!model || !input || !output) return -EINVAL;
    if(model->tensor_count) {
        const struct tensor_spec *in=find_tensor(model,ROLE_INPUT,0),*out=find_tensor(model,ROLE_OUTPUT,0);
        if(!in || !out) return -EINVAL;
        ornpu_io io_in={0,(void *)input,input_size},io_out={0,output,output_size};
        for(unsigned i=0;i<model->tensor_count;i++) {
            if(&model->tensors[i]==in) io_in.tensor_index=i;
            if(&model->tensors[i]==out) io_out.tensor_index=i;
        }
        if(model->input_count!=1 || model->output_count!=1) return -EINVAL;
        return ornpu_run_io_timed(model,&io_in,1,&io_out,1,timing);
    }
    struct header *h=&model->header;
    size_t expected=model->info.batch*h->height*h->width*h->input_channels;
    unsigned output_pixels=model->info.output_height*model->info.output_width;
    if(input_size!=expected || output_size!=model->info.batch*output_pixels*h->output_channels) return -EINVAL;
    uint64_t started=timing?now_ns():0,packed=0,submitted=0,read=0;
    uint8_t *base=model->mapping[1];
    uint8_t *in=base+h->input_offset;
    uint8_t *out=base+h->output_offset;
    unsigned input_surface=((h->height*h->width+3)/4)*64;
    unsigned input_storage=input_surface*((h->input_channels+15)/16);
    if(h->profile==9) memset(in,(uint8_t)(model->info.input_zero_point-128),model->info.batch*input_storage);
    else memset(in,model->info.input_zero_point,model->info.batch*h->height*h->input_stride*h->input_channels);
    unsigned output_surface=((output_pixels+3)/4)*64;
    unsigned output_storage=output_surface*((h->output_channels+15)/16);
    memset(out,0,model->info.batch*output_storage);
    if(h->profile==9) {
        for(unsigned b=0;b<model->info.batch;b++) for(unsigned pixel=0;pixel<h->height*h->width;pixel++)
            for(unsigned c=0;c<h->input_channels;c++) {
                unsigned source_pixel=b*h->height*h->width+pixel;
                if(model->info.input_tensor_count==2) {
                    unsigned tensor_pixels=h->height/2*h->width;
                    unsigned tensor=pixel/tensor_pixels,within=pixel%tensor_pixels;
                    source_pixel=tensor*model->info.batch*tensor_pixels+b*tensor_pixels+within;
                }
                in[b*input_storage+(c/16)*input_surface+pixel*16+c%16]=(uint8_t)(input[source_pixel*h->input_channels+c]-128);
            }
    } else {
        for(unsigned row=0;row<h->height;row++)
            memcpy(in+row*h->input_stride*h->input_channels,input+row*h->width*h->input_channels,h->width*h->input_channels);
    }
    packed=timing?now_ns():0;
    int rc=sync_buffer(model,0,1);
    if(rc) return rc;
    rc=sync_buffer(model,1,1);
    if(rc) return rc;
    submitted=timing?now_ns():0;
    rc=submit_tasks(model);
    if(rc) return rc;
    read=timing?now_ns():0;
    rc=sync_buffer(model,1,2);
    if(rc) return rc;
    for(unsigned b=0;b<model->info.batch;b++) for(unsigned pixel=0;pixel<output_pixels;pixel++)
        for(unsigned c=0;c<h->output_channels;c++)
            output[(b*output_pixels+pixel)*h->output_channels+c]=(int8_t)out[b*output_storage+(c/16)*output_surface+pixel*16+c%16];
    if(timing) {
        uint64_t done=now_ns();
        timing->pack_ns=packed-started;
        timing->submit_ns=read-submitted;
        timing->readback_ns=done-read;
        timing->total_ns=done-started;
    }
    return 0;
}

int ornpu_run(ornpu_model *model,const uint8_t *input,size_t input_size,int8_t *output,size_t output_size) {
    return ornpu_run_timed(model,input,input_size,output,output_size,NULL);
}

int ornpu_submit_flags(ornpu_model *model,uint32_t extra_flags,int *fence_fd) {
    if(!model) return -EINVAL;
    /* Only the documented job bits may be added. */
    if(extra_flags & ~((uint32_t)0x2|0x8|0x10)) return -EINVAL;
    return submit_tasks_flags(model,extra_flags,fence_fd);
}

/* Experimental (docs/plans/pipelining-plan.md S8). The RV1106 config has one IRQ, and the
 * driver both forces core_mask to CORE0 and reads subcore_task[] only when
 * num_irqs > 1, so these fields are inert here: the call exists so the board can
 * show that a nonzero mask/window set does not change the verified output. */
int ornpu_set_submit_core(ornpu_model *model,uint32_t core_mask,const uint32_t *subcore,uint32_t pairs) {
    if(!model || pairs>5 || (pairs && !subcore)) return -EINVAL;
    model->core_mask=core_mask;
    for(unsigned i=0;i<10;i++) model->subcore[i]=i<2*pairs?subcore[i]:0;
    return 0;
}

int ornpu_set_input(ornpu_model *model,uint32_t tensor_index,const void *data,size_t size) {
    if(!model || !data) return -EINVAL;
    if(model->tensor_count) {
        if(tensor_index>=model->tensor_count) return -EINVAL;
        const struct tensor_spec *t=&model->tensors[tensor_index];
        return pack_tensor_input(model,t,data,size);
    }
    if(tensor_index || size!=model->info.input_bytes) return -EINVAL;
    uint8_t *in=(uint8_t *)model->mapping[1]+model->header.input_offset;
    memcpy(in,data,size);
    return 0;
}

int ornpu_sync_outputs(ornpu_model *model) {
    if(!model) return -EINVAL;
    return sync_buffer(model,1,2);
}

int ornpu_wait_fence(int fence_fd,int timeout_ms) {
    if(fence_fd<0 || timeout_ms<0) return -EINVAL;
    struct pollfd entry={.fd=fence_fd,.events=POLLIN};
    int rc=poll(&entry,1,timeout_ms);
    if(rc<0) return -errno;
    if(!rc) return -ETIMEDOUT;
    return 0;
}

/* Engine runs (S10). A non-serial container is submitted as one ioctl per maximal
 * linked run: a task whose tail link (register 0x10) is zero ends its run, a linked
 * task continues it. The compiler's `compose.engine_runs` derives the same list from
 * the same words. */
static void compute_runs(ornpu_model *model,const struct task_spec *tasks) {
    const uint8_t *payload=(const uint8_t *)model->mapping[1];
    unsigned start=0;
    model->run_count=0;
    for(unsigned i=0;i<model->task_count && model->run_count<MAX_TASKS;i++) {
        uint64_t word=0;
        memcpy(&word,payload+tasks[i].offset+(uint64_t)tasks[i].amount*8,sizeof(word));
        if(((word>>16)&0xffffffffu)==0) {
            model->run_start[model->run_count]=start;
            model->run_tasks[model->run_count]=i-start+1;
            model->run_count++;
            start=i+1;
        }
    }
    if(start<model->task_count && model->run_count<MAX_TASKS) {
        model->run_start[model->run_count]=start;
        model->run_tasks[model->run_count]=model->task_count-start;
        model->run_count++;
    }
    model->info.engine_runs=model->run_count;
}

int ornpu_open(const char *path,ornpu_model **result) {
    if(!path || !result) return -EINVAL;
    *result=NULL;
    ornpu_model *model=calloc(1,sizeof(*model));
    if(!model) return -ENOMEM;
    model->fd=-1;
    model->mapping[0]=model->mapping[1]=MAP_FAILED;
    uint8_t *payload=NULL;struct task_spec tasks[MAX_TASKS];struct constant_spec constants[MAX_CONSTANTS];
    struct tensor_spec tensors[MAX_TENSORS];unsigned tensor_count=0;
    int rc=load_program(path,&model->header,&payload,tasks,constants,&model->task_count,&model->constant_count,&model->serial,&model->info,tensors,&tensor_count);
    if(rc) goto failed;
    model->info.task_count=model->task_count;model->info.submission_serial=model->serial;
    model->tensor_count=tensor_count;model->input_count=model->info.input_tensor_count;model->output_count=model->info.output_tensor_count;
    if(model->constant_count) {
        model->constants=malloc(model->constant_count*sizeof(*model->constants));
        if(!model->constants) { rc=-ENOMEM; goto failed; }
        memcpy(model->constants,constants,model->constant_count*sizeof(*model->constants));
    }
    if(model->tensor_count) {
        model->tensors=malloc(model->tensor_count*sizeof(*model->tensors));
        if(!model->tensors) { rc=-ENOMEM; goto failed; }
        memcpy(model->tensors,tensors,model->tensor_count*sizeof(*model->tensors));
    }
    model->fd=open("/dev/rknpu",O_RDWR|O_CLOEXEC);
    if(model->fd<0) { rc=-errno; goto failed; }
    model->buffers[0]=(struct allocation){.flags=10,.size=4096};
    model->buffers[1]=(struct allocation){.flags=2,.size=model->header.arena_size};
    for(unsigned i=0;i<2;i++) {
        struct allocation *a=&model->buffers[i];
        if(ioctl(model->fd,CREATE,a)<0) { rc=-errno; goto failed; }
        model->allocated++;
        model->mapping[i]=mmap(NULL,a->size,PROT_READ|PROT_WRITE,MAP_SHARED,a->handle,0);
        if(model->mapping[i]==MAP_FAILED) { rc=-errno; goto failed; }
        memset(model->mapping[i],0,a->size);
    }
    memcpy(model->mapping[1],payload,model->header.payload_size);
    free(payload);payload=NULL;
    for(unsigned i=0;i<model->task_count;i++)
        ((struct task *)model->mapping[0])[i]=(struct task){.op=i+1,.enable=tasks[i].enable,
            .mask=tasks[i].mask,.clear=131071,.amount=tasks[i].amount,
            .command=model->buffers[1].dma+tasks[i].offset};
    model->submit_flags=5;
    for(unsigned i=0;i<model->task_count;i++) if(tasks[i].amount>256) model->submit_flags=1;
    compute_runs(model,tasks);
    *result=model;
    return 0;
failed:
    free(payload);
    ornpu_close(model);
    return rc;
}
