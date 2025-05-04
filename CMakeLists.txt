cmake_minimum_required(VERSION 3.10)
project(evolution VERSION 0.1.0 LANGUAGES CXX)

add_compile_options(
    -Wall -Wextra -Werror -Wno-unused
    -march=native
    -fstack-protector-strong
)

set(3RD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd_party")

option(USE_SYSTEM_WX "Use system-wide installed wxWidgets" OFF)

if(USE_SYSTEM_WX)
    message(STATUS "Using system-installed wxWidgets")
else()
    set(wxWidgets_ROOT_DIR "${3RD_PARTY_DIR}/wxWidgets/install/")
    list(APPEND CMAKE_PREFIX_PATH "${wxWidgets_ROOT_DIR}")
endif()

find_package(wxWidgets REQUIRED COMPONENTS core base)
include(${wxWidgets_USE_FILE})

# Setting Chipmunk2D
set(CHIPMUNK_ROOT "${3RD_PARTY_DIR}/Chipmunk2D/install")
message(STATUS "Using Chipmunk2D from ${CHIPMUNK_ROOT}")
set(CHIPMUNK_LIB "${CHIPMUNK_ROOT}/lib/libchipmunk.so")
if(EXISTS ${CHIPMUNK_LIB} AND EXISTS "${CHIPMUNK_ROOT}/include")
    include_directories(${CHIPMUNK_ROOT}/include)
    add_library(chipmunk SHARED IMPORTED)
    set_target_properties(chipmunk PROPERTIES
        IMPORTED_LOCATION ${CHIPMUNK_LIB}
    )
else()
    message(FATAL_ERROR "${CHIPMUNK_LIB} not found")
endif()

file(GLOB_RECURSE SRC_FILES "src/*.cpp")

add_executable(evolution ${SRC_FILES})
target_link_libraries(evolution ${wxWidgets_LIBRARIES} chipmunk)
target_include_directories(evolution PRIVATE
    src/entities
    src/use_cases
    src/interfaces
    src/frameworks
    src/dto
)
