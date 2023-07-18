cmake ../kida/ -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS="-mavx2 -fopenmp -mtune=native -fsycl -Wno-all -Wno-inconsistent-missing-override"
make -j
