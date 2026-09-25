cmake_minimum_required(VERSION 3.5)
project(ndi_aurora_interface)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_BUILD_TYPE Release)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)

file(GLOB_RECURSE NDI_SRC "external/ndicapi/src/*.cpp")

add_executable(pose_publisher
  src/pose_publisher.cpp
  ${NDI_SRC}
)

ament_target_dependencies(pose_publisher
  rclcpp
  geometry_msgs
)

target_include_directories(pose_publisher PRIVATE
  external/ndicapi/include
)

install(TARGETS pose_publisher
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
