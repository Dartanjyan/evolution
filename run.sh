#!/bin/bash
set -e
cmake -B build -DCMAKE_BUILD_TYPE=Release --fresh
cmake --build build
./build/evolution
