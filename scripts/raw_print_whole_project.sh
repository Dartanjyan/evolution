#!/bin/bash
find $1 -type f \( -name "*.cpp" -o -name "*.h" \) -exec sh -c 'echo -e "\n{}\n"; cat {}; echo' \;
