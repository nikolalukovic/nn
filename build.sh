#!/bin/sh

set -xe

clang -Wall -Wextra -O3 -ffast-math -o xor xor.c -lm
