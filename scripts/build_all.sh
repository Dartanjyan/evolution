#!/bin/bash
set -e

bash $(realpath $(dirname $0))/build_chipmunk2d.sh
bash $(realpath $(dirname $0))/build_wx.sh
git clone https://gitlab.com/libeigen/eigen.git $(realpath $(dirname $0)/..)/3rd_party/eigen --depth=1

echo ------------------------------------------------------------
echo "All dependencies built and installed to $(realpath $(dirname $0))/../install"
