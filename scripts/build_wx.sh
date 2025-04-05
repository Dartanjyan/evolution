#!/bin/bash
set -e

WX_DIR="$(realpath $(dirname $0))/../3rd_party/wxWidgets"

SRC_DIR="${WX_DIR}/src"

INSTALL_DIR="${WX_DIR}/install"

if [ ! -d "${SRC_DIR}/wxWidgets_external" ]; then
    mkdir -p "${SRC_DIR}"
    git clone https://github.com/wxWidgets/wxWidgets.git "${SRC_DIR}/wxWidgets_external"
    cd "${SRC_DIR}/wxWidgets_external"
    git submodule update --init 3rdparty/*
    git checkout v3.2.6
else
    echo "wxWidgets source already exists."
    cd "${SRC_DIR}/wxWidgets_external"
fi

./configure --prefix="${INSTALL_DIR}"
if [ -f make.flag ]; then
    rm make.flag
fi
if [ -f make\ install.flag ]; then
    rm make\ install.flag
fi
touch make.flag
make
rm make.flag
touch make\ install.flag
make install
rm make\ install.flag

echo "wxWidgets built and installed to ${INSTALL_DIR}"
