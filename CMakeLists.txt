cmake_minimum_required(VERSION 3.10)
project(RadonTemplateMatching)

# C++11を利用するように設定
set(CMAKE_CXX_STANDARD 11)

# プロファイリング用のフラグ設定
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pg")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -pg")

# OpenCVを検索
find_package(OpenCV REQUIRED)

# OpenMPの設定
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found. Compiling without parallelization.")
endif()

# ライブラリを生成 (radon_template_matching)
add_library(radon_template_matching STATIC radon_template_matching.cpp)

# OpenCVライブラリをradon_template_matchingライブラリにリンク
target_link_libraries(radon_template_matching ${OpenCV_LIBS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(radon_template_matching OpenMP::OpenMP_CXX)
endif()

# 実行ファイル (main) を生成し、radon_template_matchingライブラリとリンク
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} radon_template_matching)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

