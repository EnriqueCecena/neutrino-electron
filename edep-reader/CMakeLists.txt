cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(EDEPTree LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_PLATFORM_INDEPENDENT_CODE ON)

# Locate EDep-sim
#find_package(EDepSim)

find_package(EDepSim NAMES EDepSim REQUIRED)

find_package(ROOT REQUIRED COMPONENTS Geom Physics Matrix MathCore RIO Net Tree)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-unused-parameter")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include $ENV{EDEPSIM_IO_DIR})
include(${ROOT_USE_FILE})

# set output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_library(EDEPTree SHARED src/EDEPTree.cpp src/EDEPTrajectory.cpp src/EDEPHit.cpp)
target_include_directories(EDEPTree PUBLIC 
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"  
  "$<INSTALL_INTERFACE:include>")
target_link_libraries(EDEPTree PUBLIC EDepSim::edepsim_io)


if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "default install path" FORCE)
endif()

configure_file(setup.sh "${CMAKE_INSTALL_PREFIX}/setup.sh" COPYONLY)

install(TARGETS EDEPTree
  EXPORT EDEPTreeTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
  LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")

install(
   DIRECTORY "${CMAKE_BINARY_DIR}/."
   DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
   FILES_MATCHING 
   PATTERN "*.pcm"
   PATTERN "*.rootmap"
   PATTERN "lib" EXCLUDE
   PATTERN "bin" EXCLUDE
   PATTERN "CMakeFiles" EXCLUDE)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
  DESTINATION "${CMAKE_INSTALL_PREFIX}"
  PATTERN "Linkdef.h" EXCLUDE)

#############################################################
#
# Prepare the package so that it can be used with the find_package interface.
#
#############################################################

# Include module with function 'write_basic_package_version_file'
include(CMakePackageConfigHelpers)

# Build the targets description so that the package can be configured
# using find_package.
install(EXPORT EDEPTreeTargets
  NAMESPACE EDEPTree::
  DESTINATION lib/cmake/EDEPTree)

# Write the 'EDEPTreeConfigVersion.cmake' file which can be used to
# check if a version meets the requested properties.
write_basic_package_version_file(
  EDEPTreeConfigVersion.cmake
  VERSION 1.2.3
  COMPATIBILITY SameMajorVersion)

# Write the 'EDEPTreeConfig.cmake' file so that a user package can
# access this with find_package.
configure_package_config_file(
  PackageConfig.cmake.in
  EDEPTreeConfig.cmake
  PATH_VARS CMAKE_INSTALL_PREFIX
  INSTALL_DESTINATION lib/cmake/EDEPTree)

# Install the config files.
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/EDEPTreeConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/EDEPTreeConfigVersion.cmake
  DESTINATION lib/cmake/EDEPTree )

