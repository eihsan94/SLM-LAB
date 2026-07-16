add_library(slm_math STATIC
    src/matrix.cpp
)
add_library(slm::math ALIAS slm_math)

target_include_directories(slm_math
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_features(slm_math PUBLIC cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(slm_math PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(slm_math PRIVATE /W4)
endif()
