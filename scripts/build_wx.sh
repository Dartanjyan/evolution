#!/bin/bash
set -e

WX_DIR="$(realpath $(dirname $0))/../3rd_party/wxWidgets"

SRC_DIR="${WX_DIR}/src"

INSTALL_DIR="${WX_DIR}/install"

if [ ! -d "${SRC_DIR}/wxWidgets_external" ]; then
    mkdir -p "${SRC_DIR}"
    git clone https://github.com/wxWidgets/wxWidgets.git "${SRC_DIR}/wxWidgets_external"
    cd "${SRC_DIR}/wxWidgets_external"
    git checkout v3.2.6
    git submodule update --init 3rdparty/*
else
    echo "wxWidgets source already exists."
    cd "${SRC_DIR}/wxWidgets_external"
fi

cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DwxBUILD_CMAKE_SUPPORT=ON --fresh
cmake --build build --config Release
cmake --install build

echo "wxWidgets built and installed to ${INSTALL_DIR}"
