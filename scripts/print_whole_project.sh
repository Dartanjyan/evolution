#!/bin/bash
find $1 -type f \( -name "*.cpp" -o -name "*.h" \) -exec sh -c 'echo -e "\n\033[1;36m=== Файл: {}\033[0m\n"; cat {}; echo' \;
