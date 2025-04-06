cmake_minimum_required(VERSION 3.10)
project(evolution VERSION 0.1.0 LANGUAGES CXX)

set(3RD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd_party")

option(USE_SYSTEM_WX "Use system-wide installed wxWidgets" OFF)


if(USE_SYSTEM_WX)
    message(STATUS "Using system-installed wxWidgets")
    find_package(wxWidgets REQUIRED COMPONENTS core base)
    include(${wxWidgets_USE_FILE})
else()
    set(wxWidgets_ROOT_DIR "${3RD_PARTY_DIR}/wxWidgets/install")
    set(wxWidgets_DIR "${wxWidgets_ROOT_DIR}/lib/cmake/wxWidgets")
    set(wxWidgets_USE_FILE "${wxWidgets_ROOT_DIR}/lib/cmake/wxWidgets/wxWidgetsConfig.cmake")

    message(STATUS "Using wxWidgets from ${wxWidgets_ROOT_DIR}")
    find_package(wxWidgets REQUIRED COMPONENTS core base NO_DEFAULT_PATH)
    include(${wxWidgets_USE_FILE})
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

file(GLOB_RECURSE SRC_FILES "src/*.cpp")

add_executable(evolution ${SRC_FILES})
target_link_libraries(evolution ${wxWidgets_LIBRARIES})
target_include_directories(evolution PRIVATE
    src/entities
    src/use_cases
    src/interfaces
    src/frameworks
)
