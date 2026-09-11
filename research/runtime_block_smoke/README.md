# C32 to C64 file-runner check

The rebuilt ARM open-rknpu-run executed native_input_blocks_suite/model023.bin
on one input. output.i8 exactly equals expected.i8 (3072 bytes). The former
1024-byte input-buffer limit was raised to32768; output capacity is65536.
