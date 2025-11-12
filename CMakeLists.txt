cmake_minimum_required(VERSION 3.10)
project(evolution VERSION 0.1.0 LANGUAGES CXX)

add_compile_options(
    -Wall -Wextra -Werror -Wno-unused
    -fstack-protector-strong
    -g
    -std=c++20
)

set(3RD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd_party")
option(CROSS_WIN "Build for Windows using MinGW" OFF)

# --- Определяем платформу ---
if(CROSS_WIN)
    message(STATUS ">>> Cross-compiling for Windows (MinGW)")
    set(CMAKE_SYSTEM_NAME Windows)
    set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)
    set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)
    set(WX_ROOT "${3RD_PARTY_DIR}/wxWidgets-win/install")
    set(CHIPMUNK_ROOT "${3RD_PARTY_DIR}/Chipmunk2D-win/install")
else()
    message(STATUS ">>> Building for Linux")
    set(WX_ROOT "${3RD_PARTY_DIR}/wxWidgets/install")
    set(CHIPMUNK_ROOT "${3RD_PARTY_DIR}/Chipmunk2D/install")
endif()

# --- wxWidgets ---
if(CROSS_WIN)
    message(STATUS "Using wxWidgets (manual include/lib mode for Windows)")
    include_directories(
        "${WX_ROOT}/include/wx-3.2"
        "${WX_ROOT}/lib/wx/include/x86_64-w64-mingw32-msw-unicode-3.2"
    )
    link_directories("${WX_ROOT}/lib")

    # Укажи точные имена файлов в lib/
    set(wxWidgets_LIBRARIES
        "${WX_ROOT}/lib/libwx_baseu-3.2-x86_64-w64-mingw32.dll.a"
        "${WX_ROOT}/lib/libwx_mswu_core-3.2-x86_64-w64-mingw32.dll.a"
    )
else()
    message(STATUS "Using wxWidgets for Linux (via find_package)")
    find_package(wxWidgets REQUIRED COMPONENTS core base)
    include(${wxWidgets_USE_FILE})
endif()

# --- Chipmunk2D ---
message(STATUS "Using Chipmunk2D from ${CHIPMUNK_ROOT}")

if(CROSS_WIN)
    set(CHIPMUNK_LIB "${CHIPMUNK_ROOT}/lib/libchipmunk.dll.a")
else()
    set(CHIPMUNK_LIB "${CHIPMUNK_ROOT}/lib/libchipmunk.so")
endif()

if(EXISTS ${CHIPMUNK_LIB} AND EXISTS "${CHIPMUNK_ROOT}/include")
    include_directories(${CHIPMUNK_ROOT}/include)
    add_library(chipmunk SHARED IMPORTED)
    set_target_properties(chipmunk PROPERTIES IMPORTED_LOCATION ${CHIPMUNK_LIB})
else()
    message(FATAL_ERROR "Chipmunk library not found: ${CHIPMUNK_LIB}")
endif()

# --- Исходники ---
file(GLOB_RECURSE SRC_FILES "src/*.cpp")
add_executable(evolution ${SRC_FILES})

target_link_libraries(evolution ${wxWidgets_LIBRARIES} chipmunk)
target_include_directories(evolution PRIVATE
    src/entities
    src/use_cases
    src/interfaces
    src/frameworks
    src/dto
    3rd_party/json/single_include
)

# --- Упаковка для Windows ---
if(CROSS_WIN)
    find_program(SEVEN_ZIP_EXECUTABLE 7z)
    if(NOT SEVEN_ZIP_EXECUTABLE)
        message(FATAL_ERROR "7z not found! Please install p7zip.")
    endif()

    add_custom_command(TARGET evolution POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Packing Windows build into 7z..."
        COMMAND "${SEVEN_ZIP_EXECUTABLE}" a "${CMAKE_BINARY_DIR}/evolution-win.7z"
            "${CMAKE_BINARY_DIR}/evolution.exe"
            ${wxWidgets_LIBRARIES}
            "${WX_ROOT}/bin/wxbase32u_gcc_custom.dll"
            "${WX_ROOT}/bin/wxmsw32u_core_gcc_custom.dll"
            "${CHIPMUNK_ROOT}/bin/libchipmunk.dll"
            "/usr/x86_64-w64-mingw32/bin/libgcc_s_seh-1.dll"
            "/usr/x86_64-w64-mingw32/bin/libwinpthread-1.dll"
            "/usr/x86_64-w64-mingw32/bin/libstdc++-6.dll"

        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Creating 7z archive with exe and DLLs"
    )
endif()

