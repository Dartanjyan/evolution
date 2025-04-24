#!/bin/bash
set -e

WX_DIR="$(realpath $(dirname $0))/../3rd_party/wxWidgets"
SRC_DIR="${WX_DIR}/src"
INSTALL_DIR="${WX_DIR}/install"

mkdir -p "${SRC_DIR}"
git clone https://github.com/wxWidgets/wxWidgets.git "${SRC_DIR}/wxWidgets_external"
cd "${SRC_DIR}/wxWidgets_external"
git checkout v3.2.6
git submodule update --init 3rdparty/*

cmake -B build \
    --fresh \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DwxBUILD_CMAKE_SUPPORT=ON \
    -DwxBUILD_SHARED=ON \
    -DwxBUILD_TESTS=OFF \
    -DwxBUILD_EXAMPLES=OFF \
    -DwxBUILD_COMPONENTS="core;base"
cmake --build build --config Release
cmake --install build
rm -rf ${SRC_DIR}

echo ------------------------------------------------------------
echo "wxWidgets built and installed to ${INSTALL_DIR}"
