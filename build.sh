#!/bin/bash 

if [ -d "build" ]; then
    rm -rf build
fi

if [ -d "sdk_out" ]; then
    rm -rf sdk_out
fi

# Build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../sdk_out ..
make -j4 && make install

# example
cp -r ../examples ../sdk_out
cd ../sdk_out/examples
mkdir build && cd build
cmake ..
make -j4 
