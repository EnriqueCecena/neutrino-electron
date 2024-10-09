cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project(EDEPReader LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_PLATFORM_INDEPENDENT_CODE ON)

# Locate EDep-sim
find_package(EDepSim)

find_library(EDEPTree NAMES EDEPTree PATHS ENV LD_LIBRARY_PATH)
if(NOT EDEPTree)
  message(FATAL_ERROR "Failed to find the EDEPTree library.")
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-unused-parameter")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include $ENV{EDEPTree_DIR})
message($ENV{EDEPTree_DIR})

# set output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Creates EDEPReader executable.
add_executable(EDEPReader EDEPReader.cpp)
target_link_libraries(EDEPReader PUBLIC EDepSim::edepsim_io ${EDEPTree})

add_executable(EDEPDisplay EDEPDisplay.cpp)
target_link_libraries(EDEPDisplay PUBLIC EDepSim::edepsim_io ${EDEPTree})

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "default install path" FORCE)
endif()

install(TARGETS EDEPReader EDEPDisplay DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

