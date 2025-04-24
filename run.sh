#!/bin/bash
set -e
cmake -B build -DCMAKE_BUILD_TYPE=Debug --fresh
cmake --build build
# ./build/evolution
