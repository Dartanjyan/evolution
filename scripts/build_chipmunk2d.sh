#!/bin/bash
set -e

CHIPMUNK_DIR="$(realpath $(dirname $0))/../3rd_party/Chipmunk2D"
SRC_DIR="${CHIPMUNK_DIR}/src"
INSTALL_DIR="${CHIPMUNK_DIR}/install"

mkdir -p "${SRC_DIR}"
git clone https://github.com/slembcke/Chipmunk2D "${SRC_DIR}"
cd "${SRC_DIR}"

cmake -B build \
    --fresh \
    -DBUILD_DEMOS=OFF \
    -DBUILD_SHARED=ON \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DCMAKE_BUILD_TYPE=Debug
cmake --build build
cmake --install build
rm -rf ${SRC_DIR}
echo ------------------------------------------------------------
echo "Chipmunk2D built and installed to ${INSTALL_DIR}"