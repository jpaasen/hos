

cuda_add_library(HuygensOnSpeed CudaUtils.cpp DisplayResponse.cu HuygensOnCPU.cpp HuygensOnGPU.cu HuygensSimulator.cpp ObservationArea.cpp PaintTool.cpp PointSource.cpp Source.cpp)

target_link_libraries( HuygensOnSpeed ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUDART_LIBRARY})
