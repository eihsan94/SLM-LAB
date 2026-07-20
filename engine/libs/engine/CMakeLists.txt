add_library(slm_engine STATIC
    src/controls.cpp
    src/lesson.cpp
    src/show.cpp
    src/viewer.cpp
)
add_library(slm::engine ALIAS slm_engine)

target_include_directories(slm_engine
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(slm_engine
    PUBLIC
        slm::math
    PRIVATE
        imgui::imgui
        implot::implot
        implot3d::implot3d
        glfw
        OpenGL::GL
)

target_compile_features(slm_engine PUBLIC cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(slm_engine PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(slm_engine PRIVATE /W4)
endif()
