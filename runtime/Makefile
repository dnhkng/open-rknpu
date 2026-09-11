CC ?= cc
AR ?= ar
CFLAGS ?= -O2 -Wall -Wextra -std=c11
CPPFLAGS += -D_GNU_SOURCE

all: open-rknpu-run libopen-rknpu.a

open-rknpu-run: main.o open_rknpu.o
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

libopen-rknpu.a: open_rknpu.o
	$(AR) rcs $@ $^

%.o: %.c open_rknpu.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o open-rknpu-run libopen-rknpu.a

.PHONY: all clean
