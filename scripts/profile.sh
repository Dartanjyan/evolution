#!/bin/bash
sudo perf record -F 99 -g -- ./build/evolution

if [ ! -d ./3rd_party/FlameGraph ]; then
  git clone https://github.com/brendangregg/FlameGraph 3rd_party/FlameGraph
fi

sudo perf script | ./3rd_party/FlameGraph/stackcollapse-perf.pl > profile.perf-folded
./3rd_party/FlameGraph/flamegraph.pl profile.perf-folded > perf.svg
