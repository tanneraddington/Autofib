cmake_minimum_required(VERSION 3.8)
project(motor_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(dynamixel_sdk REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

include_directories(include)



add_executable(motor_node src/motor_node.cpp)
add_executable(endocam_motor_node src/endocam_motor.cpp)
target_include_directories(motor_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
  target_include_directories(endocam_motor_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)

target_compile_features(motor_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_compile_features(endocam_motor_node PUBLIC c_std_99 cxx_std_17) 

ament_target_dependencies(motor_node
  dynamixel_sdk
  rclcpp
  std_msgs
  geometry_msgs
)

ament_target_dependencies(endocam_motor_node
  dynamixel_sdk
  rclcpp
  std_msgs
  geometry_msgs
)



install(TARGETS motor_node
  DESTINATION lib/${PROJECT_NAME})
  install(TARGETS endocam_motor_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
