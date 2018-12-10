cmake_minimum_required(VERSION 2.6)
project(imchar)

set(CMAKE_INCLUDE_CURRENT_DIR ON)


#############################################
# OpenMP support
option(USE_OPENMP "use openmp as multi-thread implement" ON)

# Compiler variables
set(BUILD_SHARED_LIBS ON)
set(OPENCV_LIBS /usr/local/OpenCV/lib)
set(BUILD_DEPENDENCE_DIR ${PROJECT_SOURCE_DIR}/../Dependencies/lib)
set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-w -g -W -Wall -fopenmp")
set(CMAKE_CXX_FLAGS ${CMAKE_C_FLAGS})
set(CMAKE_INSTALL_PREFIX $HOME/im2video)
set(CMAKE_SHARED_LINKER_FLAGS -fopenmp)
set(CMAKE_EXE_LINKER_FLAGS "-L${BUILD_DEPENDENCE_DIR} -L${OPENCV_LIBS} -L/usr/lib64 -L/usr/local/ffmpeg/lib")
#set(BUILD_BIN_DIR ${PROJECT_SOURCE_DIR}/../Bin)     # this value can be defferent rely on developer

# Compiler options
add_definitions()
#############################################


#############################################
# Compile file option
# Get whole source files that need to compile
aux_source_directory(. SOURCE_FILES)
set(SOURCE_FILES ${SOURCE_FILES})

# Compile application to executable program
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

#############################################


#############################################
# Dependence library setting
target_link_libraries(${PROJECT_NAME} 
pthread
opencv_core
opencv_features2d
opencv_imgproc
opencv_flann
opencv_highgui
#opencv_nonfree
opencv_objdetect
glog
gfortran
avcodec
avutil
avformat
swscale
)

# Properties setting
set_target_properties(${PROJECT_NAME} PROPERTIES
RUNTIME_OUTPUT_DIRECTORY "${BUILD_BIN_DIR}"
ARCHIVE_OUTPUT_DIRECTORY "${BUILD_BIN_DIR}"
)

# Include path
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/../Dependencies/include
/usr/local/OpenCV/include
/usr/local/include
/usr/local/ffmpeg/include
)
#############################################
