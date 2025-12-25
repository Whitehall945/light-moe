cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(light_moe VERSION 0.1.0 LANGUAGES CXX CUDA)

# ============================================================================
# Options
# ============================================================================
option(LIGHT_MOE_BUILD_TESTS "Build unit tests" ON)
option(LIGHT_MOE_BUILD_BENCHMARKS "Build benchmarks" ON)
option(LIGHT_MOE_BUILD_PYTHON "Build Python bindings" ON)

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# CUDA Configuration
# ============================================================================
find_package(CUDAToolkit REQUIRED)

# Detect GPU architecture if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Default to Turing (2080 Ti) and Ampere
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
    message(STATUS "No CUDA architectures specified, using default: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -DNDEBUG")

# ============================================================================
# Third Party Dependencies
# ============================================================================

# CUTLASS - check if exists as submodule
set(CUTLASS_DIR ${CMAKE_SOURCE_DIR}/third_party/cutlass)
if(EXISTS ${CUTLASS_DIR}/include)
    message(STATUS "Found CUTLASS at ${CUTLASS_DIR}")
    set(CUTLASS_INCLUDE_DIR ${CUTLASS_DIR}/include)
else()
    message(WARNING "CUTLASS not found. Please run: git submodule update --init --recursive")
    message(WARNING "Or manually clone CUTLASS to ${CUTLASS_DIR}")
endif()

# NCCL
find_package(NCCL QUIET)
if(NOT NCCL_FOUND)
    # Try to find NCCL in CUDA toolkit
    find_library(NCCL_LIBRARY nccl HINTS ${CUDAToolkit_LIBRARY_DIR})
    find_path(NCCL_INCLUDE_DIR nccl.h HINTS ${CUDAToolkit_INCLUDE_DIRS})
    if(NCCL_LIBRARY AND NCCL_INCLUDE_DIR)
        message(STATUS "Found NCCL: ${NCCL_LIBRARY}")
    else()
        message(WARNING "NCCL not found. Distributed features will be disabled.")
    endif()
endif()

# ============================================================================
# Include Directories
# ============================================================================
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

if(CUTLASS_INCLUDE_DIR)
    include_directories(${CUTLASS_INCLUDE_DIR})
endif()

if(NCCL_INCLUDE_DIR)
    include_directories(${NCCL_INCLUDE_DIR})
endif()

# ============================================================================
# Library Target
# ============================================================================
file(GLOB_RECURSE LIGHT_MOE_SOURCES
    src/core/*.cpp
    src/core/*.cu
    src/ops/*.cpp
    src/ops/*.cu
    src/ops/cute/*.cu
    src/ops/quantization/*.cu
    src/comm/*.cpp
    src/comm/*.cu
)

# Create library (will be empty initially, sources added as developed)
add_library(light_moe SHARED ${LIGHT_MOE_SOURCES})

target_link_libraries(light_moe
    CUDA::cudart
    CUDA::cublas
)

if(NCCL_LIBRARY)
    target_link_libraries(light_moe ${NCCL_LIBRARY})
    target_compile_definitions(light_moe PRIVATE LIGHT_MOE_USE_NCCL)
endif()

set_target_properties(light_moe PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# Python Bindings
# ============================================================================
if(LIGHT_MOE_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development)
    
    if(Python3_FOUND)
        # Try to find pybind11
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        
        if(PYBIND11_CMAKE_DIR)
            list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
            find_package(pybind11 CONFIG)
            
            if(pybind11_FOUND)
                pybind11_add_module(_light_moe_C python/light_moe/_C/bindings.cpp)
                target_link_libraries(_light_moe_C PRIVATE light_moe)
            else()
                message(WARNING "pybind11 CMake config not found. Python bindings disabled.")
            endif()
        else()
            message(WARNING "pybind11 not found. Install with: pip install pybind11")
        endif()
    else()
        message(WARNING "Python3 not found. Python bindings disabled.")
    endif()
endif()

# ============================================================================
# Tests
# ============================================================================
if(LIGHT_MOE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()

# ============================================================================
# Benchmarks
# ============================================================================
if(LIGHT_MOE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ============================================================================
# Installation
# ============================================================================
install(TARGETS light_moe
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/light_moe
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "Light-MoE Configuration Summary")
message(STATUS "================================")
message(STATUS "Version:           ${PROJECT_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "Build tests:       ${LIGHT_MOE_BUILD_TESTS}")
message(STATUS "Build benchmarks:  ${LIGHT_MOE_BUILD_BENCHMARKS}")
message(STATUS "Build Python:      ${LIGHT_MOE_BUILD_PYTHON}")
message(STATUS "")
