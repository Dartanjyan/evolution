#!/bin/bash
set -e

WX_DIR="$(realpath $(realpath $(dirname $0))/../3rd_party/wxWidgets-win)"
SRC_DIR="${WX_DIR}/src"
INSTALL_DIR="${WX_DIR}/install"

mkdir -p "${SRC_DIR}"
git clone https://github.com/wxWidgets/wxWidgets.git "${SRC_DIR}"
git config --global --add safe.directory ${SRC_DIR}
cd "${SRC_DIR}"
git checkout v3.2.8
git submodule update --init 3rdparty/*

cmake -B build \
    --fresh \
    -DCMAKE_SYSTEM_NAME=Windows \
    -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc \
    -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DwxBUILD_CMAKE_SUPPORT=ON \
    -DwxBUILD_SHARED=ON \
    -DwxBUILD_TESTS=OFF \
    -DwxBUILD_EXAMPLES=OFF
cmake --build build
cmake --install build
#rm -rf ${SRC_DIR}

echo ------------------------------------------------------------
echo "wxWidgets built for Windows and installed to ${INSTALL_DIR}"
