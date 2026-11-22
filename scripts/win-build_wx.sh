#!/bin/bash
set -e

WX_DIR="$(realpath $(realpath $(dirname $0))/../3rd_party/wxWidgets-win)"
SRC_DIR="${WX_DIR}/src"
INSTALL_DIR="${WX_DIR}/install"

if [ ! -d "${SRC_DIR}" ]; then
    mkdir -p "${SRC_DIR}"
    git clone https://github.com/wxWidgets/wxWidgets.git "${SRC_DIR}" --depth=1 -b v3.2.8
    git config --global --add safe.directory ${SRC_DIR}
    cd "${SRC_DIR}"
    git submodule update --init
else
    cd "${SRC_DIR}"
fi

mkdir -p build-msw
cd build-msw
HOST=x86_64-w64-mingw32
../configure \
  --host=$HOST \
  --build=x86_64-pc-linux-gnu \
  --with-msw \
  --enable-unicode \
  --enable-shared \
  --prefix=$INSTALL_DIR \
  CC=${HOST}-gcc CXX=${HOST}-g++ AR=${HOST}-ar RANLIB=${HOST}-ranlib WINDRES=${HOST}-windres

make -j$(($(nproc)-1))
make install

rm -rf ${SRC_DIR}

echo ------------------------------------------------------------
echo "wxWidgets built for Windows and installed to ${INSTALL_DIR}"
