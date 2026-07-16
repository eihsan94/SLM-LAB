cmake_minimum_required(VERSION 3.21)

project(slm_learning_lab VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Keep the executable path identical for single- and multi-config generators.
foreach(SLM_CONFIG IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER "${SLM_CONFIG}" SLM_CONFIG_UPPER)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${SLM_CONFIG_UPPER}
        "${CMAKE_BINARY_DIR}/bin")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${SLM_CONFIG_UPPER}
        "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${SLM_CONFIG_UPPER}
        "${CMAKE_BINARY_DIR}/lib")
endforeach()

find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(implot CONFIG REQUIRED)
find_package(implot3d CONFIG REQUIRED)

add_subdirectory(libs/math)
add_subdirectory(libs/engine)

# A learner adds only lessons/<topic>/lesson.cpp. CMake discovers it and
# regenerates the executable automatically; no lesson list is maintained.
file(GLOB_RECURSE SLM_LESSON_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/lessons/*/lesson.cpp"
)

add_executable(slm_lab
    main.cpp
    ${SLM_LESSON_SOURCES}
)

target_link_libraries(slm_lab PRIVATE slm::engine)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(slm_lab PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(slm_lab PRIVATE /W4)
endif()
