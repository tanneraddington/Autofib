cmake_minimum_required(VERSION 3.21)
include(FetchContent)

project(smoother_uterus)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GTSAM needs to be built in Release mode for best performance
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

#-----------------------------------------------------------
# ROS packages
#-----------------------------------------------------------

set(smoother_uterus_AMENT_DEPENDENCIES
  ament_index_cpp
  cv_bridge
  geometry_msgs
  rclcpp
  sensor_msgs
  tf2_ros
  visualization_msgs
)

foreach(package_name IN LISTS smoother_uterus_AMENT_DEPENDENCIES)
  find_package(${package_name} REQUIRED)
endforeach()

#-----------------------------------------------------------
# Non-ROS packages
#-----------------------------------------------------------

find_package(OpenCV 4.6 REQUIRED)
find_package(yaml-cpp 0.8 REQUIRED)

#-----------------------------------------------------------
# GTSAM
#-----------------------------------------------------------

# Do NOT use find_package(GTSAM), since other installs might exist

set(GTSAM_WITH_TBB              ON  CACHE BOOL "" FORCE)
set(GTSAM_BUILD_TESTS           OFF CACHE BOOL "" FORCE)
set(GTSAM_BUILD_PYTHON          OFF CACHE BOOL "" FORCE)
set(GTSAM_BUILD_EXAMPLES_ALWAYS OFF CACHE BOOL "" FORCE)
set(GTSAM_BUILD_UNSTABLE        OFF CACHE BOOL "" FORCE)

set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  GTSAM
  GIT_REPOSITORY https://github.com/borglab/gtsam.git
  GIT_TAG        4.3a1
)

FetchContent_MakeAvailable(GTSAM)

message(STATUS "smoother_uterus: using bundled GTSAM from FetchContent")
message(STATUS "smoother_uterus: GTSAM source = ${gtsam_SOURCE_DIR}")

#-----------------------------------------------------------
# ONNX Runtime (prebuilt GPU binary via FetchContent)
#-----------------------------------------------------------

FetchContent_Declare(
  onnxruntime-gpu
  URL https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-gpu-1.19.2.tgz
  URL_HASH SHA256=4d1c10f0b410b67261302c6e18bb1b05ba924ca9081e3a26959e0d12ab69f534
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnxruntime-gpu)

add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime-gpu_SOURCE_DIR}/include"
  IMPORTED_SONAME    libonnxruntime.so.1
  IMPORTED_LOCATION  ${onnxruntime-gpu_SOURCE_DIR}/lib/libonnxruntime.so.1.19.2
)
add_library(onnxproviders::shared SHARED IMPORTED)
set_target_properties(onnxproviders::shared PROPERTIES
  IMPORTED_SONAME   libonnxruntime_providers_shared.so
  IMPORTED_LOCATION ${onnxruntime-gpu_SOURCE_DIR}/lib/libonnxruntime_providers_shared.so
)
add_library(onnxproviders::cuda SHARED IMPORTED)
set_target_properties(onnxproviders::cuda PROPERTIES
  IMPORTED_SONAME   libonnxruntime_providers_cuda.so
  IMPORTED_LOCATION ${onnxruntime-gpu_SOURCE_DIR}/lib/libonnxruntime_providers_cuda.so
)

#-----------------------------------------------------------
# smoother_uterus executable
#-----------------------------------------------------------

set(smoother_uterus_SOURCES
  src/ros_node/SmootherNode.cpp
  src/ros_node/SmootherPipeline.cpp
  src/ros_node/SolutionPublisher.cpp
  src/ros_node/CalibrationIo.cpp
  src/solver/SmootherSolver.cpp
  src/ros_node/KeypointModel.cpp
  src/solver/CosseratTwistFactor.cpp
  src/solver/TipProjectionFactor.cpp
  src/solver/PositionPriorFactor.cpp
  src/solver/ConstantVelocityFactor.cpp
)

add_executable(smoother_uterus src/ros_node/main.cpp ${smoother_uterus_SOURCES})

target_compile_features(smoother_uterus PUBLIC cxx_std_17)
target_include_directories(smoother_uterus PRIVATE src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(smoother_uterus
  gtsam
  onnxruntime::onnxruntime
  yaml-cpp::yaml-cpp
  ${OpenCV_LIBS}
)
ament_target_dependencies(smoother_uterus ${smoother_uterus_AMENT_DEPENDENCIES})

set(_onnx_libdir "${onnxruntime-gpu_SOURCE_DIR}/lib")
set_target_properties(smoother_uterus PROPERTIES
  BUILD_RPATH   "${_onnx_libdir}"
  INSTALL_RPATH "${_onnx_libdir}"
)

#-----------------------------------------------------------
# Install
#-----------------------------------------------------------

install(TARGETS smoother_uterus DESTINATION lib/${PROJECT_NAME})

install(
  IMPORTED_RUNTIME_ARTIFACTS
    onnxruntime::onnxruntime
    onnxproviders::shared
    onnxproviders::cuda
  LIBRARY DESTINATION lib
)

install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY models/ DESTINATION share/${PROJECT_NAME}/models)

#-----------------------------------------------------------
# Tests
#-----------------------------------------------------------

if(BUILD_TESTING)
  find_package(ament_cmake_gmock REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_ros REQUIRED)

  set(smoother_uterus_tests_AMENT_DEPENDENCIES
    camera_info_manager
    image_geometry
    image_transport
    rosgraph_msgs
  )

  foreach(package_name IN LISTS smoother_uterus_tests_AMENT_DEPENDENCIES)
    find_package(${package_name} REQUIRED)
  endforeach()

  # Integration tests must be isolated to prevent cross-talk between ROS nodes; see:
  # https://medium.com/@yifeicheng/test-isolation-and-avoiding-cross-talk-in-ros-2-3731f54e9ccf
  ament_add_ros_isolated_gtest(
    smoother_uterus_tests
    test/TestMain.cpp
    test/InitializationTest.cpp
    test/TfTest.cpp
    # Ideally we would have an intermediate library to link to
    ${smoother_uterus_SOURCES}
  )

  ament_target_dependencies(smoother_uterus_tests
    ${smoother_uterus_AMENT_DEPENDENCIES}
    ${smoother_uterus_tests_AMENT_DEPENDENCIES}
  )

  target_compile_features(smoother_uterus_tests PRIVATE cxx_std_17)
  target_include_directories(smoother_uterus_tests PRIVATE src ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(smoother_uterus_tests
    gtsam
    onnxruntime::onnxruntime
    yaml-cpp::yaml-cpp
    ${OpenCV_LIBS}
  )

  add_custom_command(
    TARGET smoother_uterus_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_SOURCE_DIR}/test/test_assets/"
      "${CMAKE_CURRENT_BINARY_DIR}/test_assets/"
    COMMENT "Copying test assets..."
  )
endif()

ament_package()
