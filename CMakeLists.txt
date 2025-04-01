cmake_minimum_required(VERSION 3.10)
project(evolution VERSION 0.1.0 LANGUAGES C CXX)

# Найти wxWidgets
find_package(wxWidgets REQUIRED COMPONENTS core base)

# Включить файлы wxWidgets
include(${wxWidgets_USE_FILE})

# Рекурсивный поиск всех .cpp файлов
file(GLOB_RECURSE SRC_FILES "src/*.cpp")

# Добавить исполняемый файл
add_executable(evolution ${SRC_FILES})

# Подключить библиотеки wxWidgets
target_link_libraries(evolution ${wxWidgets_LIBRARIES})

# Установить пути для заголовочных файлов
target_include_directories(evolution PRIVATE
    src/entities
    src/use_cases
    src/interfaces
    src/frameworks
)
