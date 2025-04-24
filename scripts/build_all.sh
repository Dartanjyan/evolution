#!/bin/bash
set -e

bash $(realpath $(dirname $0))/build_chipmunk2d.sh
bash $(realpath $(dirname $0))/build_wx.sh

echo ------------------------------------------------------------
echo "All dependencies built and installed to $(realpath $(dirname $0))/../install"