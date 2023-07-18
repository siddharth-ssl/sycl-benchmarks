cmake ../kida/ -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS="-mavx2 -fopenmp -mtune=native -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 -Wno-all -Wno-inconsistent-missing-override"
make -j
