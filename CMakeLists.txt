cmake_minimum_required(VERSION 3.10)
project(evolution VERSION 0.1.0 LANGUAGES C CXX)

set(3RD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd_party")

find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

include(ExternalProject)                                ExternalProject_Add(
    wxWidgets_external
    GIT_REPOSITORY https://github.com/wxWidgets/wxWidgets.git                                                       GIT_TAG        v3.2.6
    PREFIX         ${3RD_PARTY_DIR}/wxWidgets
    DOWNLOAD_DIR   ${CMAKE_BINARY_DIR}/downloads            CONFIGURE_COMMAND cd ${3RD_PARTY_DIR}/wxWidgets/src/wxWidgets_external/ && ./configure --prefix=${3RD_PARTY_DIR}/wxWidgets/install && cd ${3RD_PARTY_DIR}/../
    BUILD_COMMAND make -C ${3RD_PARTY_DIR}/wxWidgets/src/wxWidgets_external/
    INSTALL_COMMAND make -C ${3RD_PARTY_DIR}/wxWidgets/src/wxWidgets_external/ install
)
set(wxWidgets_ROOT_DIR "${3RD_PARTY_DIR}/wxWidgets/install")
find_package(wxWidgets REQUIRED COMPONENTS core base)
include(${wxWidgets_USE_FILE})                          

# Recursively find all .cpp files
file(GLOB_RECURSE SRC_FILES "src/*.cpp")

# Add executable target
add_executable(evolution ${SRC_FILES})

# Include wxWidgets libs
target_link_libraries(evolution ${wxWidgets_LIBRARIES})

# Set header files directories
target_include_directories(evolution PRIVATE
    src/entities
    src/use_cases
    src/interfaces
    src/frameworks
)
