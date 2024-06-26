cmake_minimum_required(VERSION 3.25)

project(GPU-homework C CXX)


add_library(${PROJECT_NAME})


option(engine_debug "Turn on debug mode for library" OFF)


target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_23)


if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
            /W4
            /WX
    )
else ()
    target_compile_options(${PROJECT_NAME} PRIVATE
            -Wall
            -Wextra
            -pedantic-errors
            -Wconversion
            -Werror
            -Wno-missing-field-initializers
    )
endif ()


target_precompile_headers(${PROJECT_NAME} PRIVATE
        <algorithm>
        <concepts>
        <expected>
        <fstream>
        <functional>
        <limits>
        <memory>
        <print>
        <ranges>
        <set>
        <span>
        <sstream>
        <stdexcept>
        <string>
        <string_view>
        <thread>
        <type_traits>
        <unordered_map>
        <utility>
        <vector>
        <variant>
)

include(dependencies.cmake)


target_include_directories(${PROJECT_NAME} PUBLIC src)

add_subdirectory(src)


unset(engine_debug CACHE)


###########
## TESTS ##
###########
add_subdirectory(example)
