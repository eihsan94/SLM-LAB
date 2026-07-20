cmake_minimum_required(VERSION 3.21)

get_filename_component(SLM_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

if(NOT DEFINED LESSON_ID OR LESSON_ID STREQUAL "")
    message(FATAL_ERROR
        "A lesson ID is required. Example: make lesson dot_product")
endif()

# The ID becomes both a directory name and part of a C++ function name.
if(NOT LESSON_ID MATCHES "^[a-z][a-z0-9_]*$")
    message(FATAL_ERROR
        "Invalid lesson ID '${LESSON_ID}'. Use lowercase snake_case, starting "
        "with a letter; for example: matrix_multiplication")
endif()

if(NOT DEFINED LESSON_TEMPLATE OR LESSON_TEMPLATE STREQUAL "")
    set(LESSON_TEMPLATE "blank")
endif()

set(SLM_LESSON_TEMPLATES blank)
if(NOT LESSON_TEMPLATE IN_LIST SLM_LESSON_TEMPLATES)
    list(JOIN SLM_LESSON_TEMPLATES ", " SLM_LESSON_TEMPLATE_NAMES)
    message(FATAL_ERROR
        "Unknown lesson template '${LESSON_TEMPLATE}'. Available templates: "
        "${SLM_LESSON_TEMPLATE_NAMES}")
endif()

function(slm_title_from_id input output)
    string(REPLACE "_" ";" words "${input}")
    set(result "")

    foreach(word IN LISTS words)
        string(SUBSTRING "${word}" 0 1 first_letter)
        string(TOUPPER "${first_letter}" first_letter)
        string(LENGTH "${word}" word_length)

        if(word_length GREATER 1)
            string(SUBSTRING "${word}" 1 -1 rest_of_word)
        else()
            set(rest_of_word "")
        endif()

        if(result STREQUAL "")
            set(result "${first_letter}${rest_of_word}")
        else()
            string(APPEND result " ${first_letter}${rest_of_word}")
        endif()
    endforeach()

    set(${output} "${result}" PARENT_SCOPE)
endfunction()

function(slm_escape_cpp_string input output)
    string(REPLACE "\\" "\\\\" escaped "${input}")
    string(REPLACE "\"" "\\\"" escaped "${escaped}")
    string(REPLACE "\r" "" escaped "${escaped}")
    string(REPLACE "\n" "\\n" escaped "${escaped}")
    set(${output} "${escaped}" PARENT_SCOPE)
endfunction()

if(NOT DEFINED LESSON_TITLE OR LESSON_TITLE STREQUAL "")
    slm_title_from_id("${LESSON_ID}" LESSON_TITLE)
endif()

if(NOT DEFINED LESSON_CATEGORY OR LESSON_CATEGORY STREQUAL "")
    set(LESSON_CATEGORY "Uncategorized")
endif()

if(NOT DEFINED LESSON_ORDER OR LESSON_ORDER STREQUAL "")
    set(LESSON_ORDER 100)
endif()

if(NOT LESSON_ORDER MATCHES "^[0-9]+$")
    message(FATAL_ERROR
        "Invalid lesson order '${LESSON_ORDER}'. ORDER must be a non-negative "
        "whole number.")
endif()

set(LESSON_DIRECTORY "${SLM_ROOT}/lessons/${LESSON_ID}")
set(LESSON_SOURCE "${LESSON_DIRECTORY}/lesson.cpp")

if(EXISTS "${LESSON_SOURCE}")
    message(FATAL_ERROR
        "Lesson '${LESSON_ID}' already exists at ${LESSON_SOURCE}. Nothing "
        "was overwritten.")
endif()

# Protect against a duplicate registration if a lesson was placed in a folder
# whose name does not match its SLM_LESSON ID.
file(GLOB EXISTING_LESSON_SOURCES "${SLM_ROOT}/lessons/*/lesson.cpp")
foreach(existing_source IN LISTS EXISTING_LESSON_SOURCES)
    file(READ "${existing_source}" existing_contents)
    string(REGEX MATCH
        "SLM_LESSON\\([ \t\r\n]*${LESSON_ID}[ \t\r\n]*,"
        duplicate_registration
        "${existing_contents}")
    if(duplicate_registration)
        message(FATAL_ERROR
            "Lesson ID '${LESSON_ID}' is already registered by "
            "${existing_source}. Nothing was created.")
    endif()
endforeach()

slm_escape_cpp_string("${LESSON_TITLE}" LESSON_TITLE_CPP)
slm_escape_cpp_string("${LESSON_CATEGORY}" LESSON_CATEGORY_CPP)

set(LESSON_TEMPLATE_FILE
    "${CMAKE_CURRENT_LIST_DIR}/templates/lesson-${LESSON_TEMPLATE}.cpp.in")
if(NOT EXISTS "${LESSON_TEMPLATE_FILE}")
    message(FATAL_ERROR "Missing lesson template: ${LESSON_TEMPLATE_FILE}")
endif()

file(MAKE_DIRECTORY "${LESSON_DIRECTORY}")
configure_file("${LESSON_TEMPLATE_FILE}" "${LESSON_SOURCE}" @ONLY
               NEWLINE_STYLE UNIX)

message(STATUS "Created lessons/${LESSON_ID}/lesson.cpp")
message(STATUS "Run it with: make run LESSON=${LESSON_ID}")
