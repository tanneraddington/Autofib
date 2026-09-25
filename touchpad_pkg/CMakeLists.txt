cmake_minimum_required(VERSION 3.8)
project(touchpad_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

include_directories(include)

#add_executable(virtuoso_node src/virtuoso_node.cpp)
add_executable(touchpad_serial_r src/touchpad_serial_r.cpp)
ament_target_dependencies(touchpad_serial_r rclcpp std_msgs geometry_msgs std_srvs)

add_executable(touchpad_serial_l src/touchpad_serial_l.cpp)
ament_target_dependencies(touchpad_serial_l rclcpp std_msgs geometry_msgs std_srvs)

#add_executable(combine_control src/combine_control.cpp)
#ament_target_dependencies(combine_control rclcpp std_msgs)

target_include_directories(touchpad_serial_r PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_include_directories(touchpad_serial_l PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_compile_features(touchpad_serial_r PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_compile_features(touchpad_serial_l PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

# ament_target_dependencies(virtuoso_ui_node_l
# virtuoso_ui_node_r
# rclcpp
# geometry_msgs
# std_msgs
# std_srvs
# )

install(TARGETS 
  touchpad_serial_r
  touchpad_serial_l
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  touchpad_serial_R.py
  touchpad_serial_L.py
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

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

