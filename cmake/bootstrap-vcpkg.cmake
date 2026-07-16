cmake_minimum_required(VERSION 3.21)

get_filename_component(SLM_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(SLM_LOCAL_VCPKG_ROOT "${SLM_ROOT}/.local/vcpkg")

if(NOT DEFINED VCPKG_ROOT OR VCPKG_ROOT STREQUAL "")
    if(DEFINED ENV{VCPKG_ROOT} AND NOT "$ENV{VCPKG_ROOT}" STREQUAL "")
        set(VCPKG_ROOT "$ENV{VCPKG_ROOT}")
    else()
        set(VCPKG_ROOT "${SLM_LOCAL_VCPKG_ROOT}")
    endif()
endif()

cmake_path(ABSOLUTE_PATH VCPKG_ROOT NORMALIZE OUTPUT_VARIABLE VCPKG_ROOT)
cmake_path(ABSOLUTE_PATH SLM_LOCAL_VCPKG_ROOT NORMALIZE
           OUTPUT_VARIABLE SLM_LOCAL_VCPKG_ROOT)

if(NOT DEFINED VCPKG_REPOSITORY OR VCPKG_REPOSITORY STREQUAL "")
    set(VCPKG_REPOSITORY "https://github.com/microsoft/vcpkg.git")
endif()

file(READ "${SLM_ROOT}/vcpkg.json" VCPKG_MANIFEST)
string(JSON VCPKG_COMMIT ERROR_VARIABLE VCPKG_JSON_ERROR
       GET "${VCPKG_MANIFEST}" "builtin-baseline")
if(VCPKG_JSON_ERROR)
    message(FATAL_ERROR
        "Could not read builtin-baseline from ${SLM_ROOT}/vcpkg.json: "
        "${VCPKG_JSON_ERROR}")
endif()

if(WIN32)
    set(VCPKG_EXECUTABLE "${VCPKG_ROOT}/vcpkg.exe")
else()
    set(VCPKG_EXECUTABLE "${VCPKG_ROOT}/vcpkg")
endif()
set(VCPKG_TOOLCHAIN "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")

# A separately managed VCPKG_ROOT is accepted as-is. The project-owned checkout
# is kept on the same revision as vcpkg.json's builtin-baseline.
if(EXISTS "${VCPKG_EXECUTABLE}" AND EXISTS "${VCPKG_TOOLCHAIN}"
   AND NOT VCPKG_ROOT STREQUAL SLM_LOCAL_VCPKG_ROOT)
    message(STATUS "Using external vcpkg from ${VCPKG_ROOT}")
    return()
endif()

find_package(Git REQUIRED)

set(VCPKG_CHECKOUT_READY FALSE)
if(EXISTS "${VCPKG_ROOT}/.git")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${VCPKG_ROOT}" rev-parse HEAD
        RESULT_VARIABLE GIT_HEAD_RESULT
        OUTPUT_VARIABLE GIT_HEAD
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(GIT_HEAD_RESULT EQUAL 0 AND GIT_HEAD STREQUAL VCPKG_COMMIT)
        set(VCPKG_CHECKOUT_READY TRUE)
    endif()
elseif(EXISTS "${VCPKG_ROOT}")
    file(GLOB VCPKG_ROOT_CONTENTS "${VCPKG_ROOT}/*")
    if(VCPKG_ROOT_CONTENTS)
        message(FATAL_ERROR
            "Cannot bootstrap vcpkg: ${VCPKG_ROOT} exists but is not a git "
            "checkout. Choose another location with -DVCPKG_ROOT=<path>.")
    endif()
endif()

if(EXISTS "${VCPKG_EXECUTABLE}" AND EXISTS "${VCPKG_TOOLCHAIN}"
   AND VCPKG_CHECKOUT_READY)
    message(STATUS "Using pinned vcpkg from ${VCPKG_ROOT}")
    return()
endif()

function(slm_run_checked)
    execute_process(
        COMMAND ${ARGV}
        RESULT_VARIABLE COMMAND_RESULT
        COMMAND_ECHO STDOUT
    )
    if(NOT COMMAND_RESULT EQUAL 0)
        message(FATAL_ERROR "Command failed with exit code ${COMMAND_RESULT}")
    endif()
endfunction()

message(STATUS "Bootstrapping vcpkg at ${VCPKG_ROOT}")
file(MAKE_DIRECTORY "${VCPKG_ROOT}")

if(NOT EXISTS "${VCPKG_ROOT}/.git")
    slm_run_checked("${GIT_EXECUTABLE}" -C "${VCPKG_ROOT}" init -q)
    slm_run_checked("${GIT_EXECUTABLE}" -C "${VCPKG_ROOT}"
                    remote add origin "${VCPKG_REPOSITORY}")
endif()

if(NOT VCPKG_CHECKOUT_READY)
    slm_run_checked("${GIT_EXECUTABLE}" -C "${VCPKG_ROOT}"
                    fetch --depth 1 origin "${VCPKG_COMMIT}")
    slm_run_checked("${GIT_EXECUTABLE}" -C "${VCPKG_ROOT}"
                    checkout --detach "${VCPKG_COMMIT}")
endif()

if(WIN32)
    slm_run_checked(cmd /c "${VCPKG_ROOT}/bootstrap-vcpkg.bat"
                    -disableMetrics)
else()
    slm_run_checked(sh "${VCPKG_ROOT}/bootstrap-vcpkg.sh"
                    -disableMetrics)
endif()
