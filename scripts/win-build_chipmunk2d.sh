#!/bin/bash
set -e

CHIPMUNK_DIR="$(realpath $(dirname $0))/../3rd_party/Chipmunk2D-win"
SRC_DIR="${CHIPMUNK_DIR}/src"
INSTALL_DIR="${CHIPMUNK_DIR}/install"

if [ ! -d "${SRC_DIR}" ]; then
    mkdir -p "${SRC_DIR}"
    git clone https://github.com/slembcke/Chipmunk2D "${SRC_DIR}" --depth=1
fi

cd "${SRC_DIR}"

HOST=x86_64-w64-mingw32

mkdir -p build-win
cd build-win

cmake .. \
    -DCMAKE_SYSTEM_NAME=Windows \
    -DCMAKE_C_COMPILER=${HOST}-gcc \
    -DCMAKE_CXX_COMPILER=${HOST}-g++ \
    -DCMAKE_RC_COMPILER=${HOST}-windres \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DCMAKE_BUILD_TYPE=Release \
    --fresh \
    -DBUILD_DEMOS=OFF \
    -DBUILD_SHARED=ON \
    -DBUILD_SHARED_LIBS=ON

make -j$(($(nproc)-1))
make install

rm -rf ${SRC_DIR}
echo ------------------------------------------------------------
echo "Chipmunk2D built and installed to ${INSTALL_DIR}"
