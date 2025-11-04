#!/bin/bash
set -e
git clone https://github.com/nlohmann/json.git $(realpath $(dirname $0))/../3rd_party/json --depth=1 -b v3.12.0
