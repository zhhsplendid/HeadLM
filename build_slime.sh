mkdir -p build
cd build
cmake ../Slime
make -j`nproc`
cp csrc/_slime_c.*.so ../Slime/slime/


