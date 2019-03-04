project(test2)
cmake_minimum_required(VERSION 2.8)
FIND_PACKAGE(PkgConfig)
pkg_check_modules (LIBUSB REQUIRED libusb-1.0)
include_directories (${LIBUSB_INCLUDE_DIRS})
link_directories (${LIBUSB_LIBRARY_DIRS})

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

SET(EIGEN_PATH /usr/include/eigen3/Eigen)



FIND_PATH(OPEN_NI_INCLUDE "XnOpenNI.h" "OpenNIConfig.h" HINTS "$ENV{OPEN_NI_INCLUDE}" "/usr/include/ni" "/usr/include/openni" "/opt/ros/groovy/include/openni_camera")
FIND_LIBRARY(OPEN_NI_LIBRARY NAMES OpenNI libOpenNI HINTS $ENV{OPEN_NI_LIB} "/usr/lib")
LINK_DIRECTORIES($ENV{OPEN_NI_LIB})
INCLUDE_DIRECTORIES(${OPEN_NI_INCLUDE} ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} ${EIGEN_PATH})
LINK_LIBRARIES(${OPEN_NI_LIBRARY})
SET(XN_NITE_INCLUDE   /home/jubu/OpenNI/NITE_v1.5.2.23/Include)
FIND_LIBRARY(XN_NITE_LIBRARY NAMES libXnVNite_1_5_2.so HINTS $ENV{XN_NITE_INSTALL_PATH} "/usr/lib")
LINK_DIRECTORIES($ENV{XN_NITE_LIB_INSTALL_PATH} "/usr/lib")
INCLUDE_DIRECTORIES(${XN_NITE_INCLUDE})
LINK_LIBRARIES(${XN_NITE_LIBRARY})

add_executable (helloDemo  SceneDrawer.cpp opengles.cpp SkeletonTracker.cpp eigen_extras.cpp skeleton_helper.cpp timer.cpp user_t.cpp Vaecoder.cpp  main2.cpp Integrator.cpp)

target_link_libraries(helloDemo ${SSE_FLAGS} ${OPEN_NI_LIBRARIES} ${XN_NITE_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )
