
cmake_minimum_required(VERSION 3.3)

SET(CMAKE_CXX_FLAGS "-Wall -fPIC -fno-rtti")
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0077 NEW)

project(mlir-jl-tblgen
LANGUAGES
   CXX
   C
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  ${LLVM_CMAKE_DIR}
  ${MLIR_CMAKE_DIR}
)
include(AddLLVM)
include(HandleLLVMOptions)
include(AddMLIR)

message(STATUS "Found LLVM: ${LLVM_VERSION}")
message(STATUS "Found MLIR: ${MLIR_VERSION}")

add_definitions(${LLVM_DEFINITIONS})
add_definitions(${MLIR_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

add_executable(mlir-jl-tblgen 
    mlir-jl-tblgen.cc
    jl-generators.cc)

target_link_libraries(mlir-jl-tblgen PUBLIC
    MLIRTableGen
    LLVMTableGen
    )

install(TARGETS mlir-jl-tblgen DESTINATION bin)