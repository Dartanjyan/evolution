#!/bin/bash
git clone https://gitlab.com/libeigen/eigen.git $(realpath $(dirname $0)/..)/3rd_party/eigen --depth=1
