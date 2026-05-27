# ExecuTorch find/link helper.
#
# Sets HAVE_EXECUTORCH and provides target_link_executorch() function.
#
# Usage:
#   -DET_BUILD=<path>          Override ExecuTorch build directory
#   -DEXECUTORCH_ROOT=<path>   Override ExecuTorch source root
#
# For Android, defaults to cmake-out-android; for host, cmake-out-host.

set(EXECUTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/executorch" CACHE PATH "ExecuTorch source root")

if(ANDROID)
    set(ET_BUILD "${EXECUTORCH_ROOT}/cmake-out-${ANDROID_ABI}" CACHE PATH "ExecuTorch build directory")
else()
    set(ET_BUILD "${EXECUTORCH_ROOT}/cmake-out-host" CACHE PATH "ExecuTorch build directory")
endif()

if(EXISTS "${ET_BUILD}/libexecutorch.a")
    message(STATUS "Found ExecuTorch: ${ET_BUILD}/libexecutorch.a")
    set(HAVE_EXECUTORCH ON)
else()
    message(WARNING
        "ExecuTorch not found at ${ET_BUILD}.\n"
        "Building without ExecuTorch support (resampler + beam search only).\n"
        "To enable, build ExecuTorch first or set -DET_BUILD=<path>")
    set(HAVE_EXECUTORCH OFF)
endif()

# Pre-built ExecuTorch static library setup (both host and Android)
if(HAVE_EXECUTORCH)
    set(ET_INCLUDE_DIRS
        "${EXECUTORCH_ROOT}/.."
        "${EXECUTORCH_ROOT}/runtime/core/portable_type/c10"
        "${EXECUTORCH_ROOT}/backends/xnnpack/third-party/cpuinfo/include"
        "${EXECUTORCH_ROOT}/backends/xnnpack/third-party/pthreadpool/include"
    )

    # Use selective ops lib if available (built with EXECUTORCH_SELECT_OPS_LIST)
    if(EXISTS "${ET_BUILD}/libexecutorch_selected_kernels.a")
        set(ET_OPS_LIB "${ET_BUILD}/libexecutorch_selected_kernels.a")
        message(STATUS "Using selective ops: ${ET_OPS_LIB}")
    else()
        set(ET_OPS_LIB "${ET_BUILD}/kernels/portable/libportable_ops_lib.a")
        message(STATUS "Using full portable ops: ${ET_OPS_LIB}")
    endif()

    # Whole-archive libs (need --whole-archive for op/backend registration)
    set(ET_WHOLE_ARCHIVE_LIBS
        "${ET_BUILD}/libexecutorch.a"
        "${ET_OPS_LIB}"
        "${ET_BUILD}/backends/xnnpack/libxnnpack_backend.a"
    )

    # Regular libs
    set(ET_REGULAR_LIBS
        "${ET_BUILD}/extension/module/libextension_module_static.a"
        "${ET_BUILD}/extension/tensor/libextension_tensor.a"
        "${ET_BUILD}/extension/data_loader/libextension_data_loader.a"
        "${ET_BUILD}/extension/flat_tensor/libextension_flat_tensor.a"
        "${ET_BUILD}/extension/named_data_map/libextension_named_data_map.a"
        "${ET_BUILD}/extension/threadpool/libextension_threadpool.a"
        "${ET_BUILD}/libexecutorch_core.a"
        "${ET_BUILD}/kernels/portable/libportable_kernels.a"
        "${ET_BUILD}/backends/xnnpack/third-party/XNNPACK/libXNNPACK.a"
        "${ET_BUILD}/backends/xnnpack/third-party/XNNPACK/libxnnpack-microkernels-prod.a"
        "${ET_BUILD}/backends/xnnpack/third-party/pthreadpool/libpthreadpool.a"
        "${ET_BUILD}/backends/xnnpack/third-party/cpuinfo/libcpuinfo.a"
    )

    # flatccrt
    if(NOT ANDROID AND EXISTS "${ET_BUILD}/third-party/flatcc_ep/lib/libflatccrt.a")
        list(APPEND ET_REGULAR_LIBS "${ET_BUILD}/third-party/flatcc_ep/lib/libflatccrt.a")
    endif()

    # KleidiAI (ARM optimized GEMM)
    if(EXISTS "${ET_BUILD}/kleidiai/libkleidiai.a")
        list(APPEND ET_REGULAR_LIBS "${ET_BUILD}/kleidiai/libkleidiai.a")
    endif()


    add_library(executorch INTERFACE)
    target_include_directories(executorch INTERFACE ${ET_INCLUDE_DIRS})
    target_compile_definitions(executorch INTERFACE C10_USING_CUSTOM_GENERATED_MACROS)

    # Whole archive libs
    if(ET_WHOLE_ARCHIVE_LIBS)
        target_link_libraries(executorch INTERFACE
            "-Wl,--whole-archive"
            ${ET_WHOLE_ARCHIVE_LIBS}
            "-Wl,--no-whole-archive"
        )
    endif()

    # Regular libs
    foreach(lib ${ET_REGULAR_LIBS})
        target_link_libraries(executorch INTERFACE ${lib})
    endforeach()
endif()

# Helper function to link ExecuTorch to a target
function(target_link_executorch target_name)
    target_include_directories(${target_name} PRIVATE ${ET_INCLUDE_DIRS})
    target_compile_definitions(${target_name} PRIVATE C10_USING_CUSTOM_GENERATED_MACROS)

    foreach(lib ${ET_WHOLE_ARCHIVE_LIBS})
        target_link_libraries(${target_name} PRIVATE
            -Wl,--whole-archive ${lib} -Wl,--no-whole-archive)
    endforeach()

    foreach(lib ${ET_REGULAR_LIBS})
        target_link_libraries(${target_name} PRIVATE ${lib})
    endforeach()

    if(ANDROID)
        target_link_libraries(${target_name} PRIVATE log)
    else()
        target_link_libraries(${target_name} PRIVATE pthread dl)
    endif()
endfunction()
