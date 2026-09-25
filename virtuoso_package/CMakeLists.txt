cmake_minimum_required(VERSION 3.8)
project(virtuoso_package)

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
add_executable(virtuoso_ui_node_l src/virtuoso_ui_node_l.cpp)
ament_target_dependencies(virtuoso_ui_node_l rclcpp std_msgs geometry_msgs std_srvs)

add_executable(ikin_control_l src/ikin_control_l.cpp)
ament_target_dependencies(ikin_control_l rclcpp std_msgs geometry_msgs std_srvs)

add_executable(ikin_control_r src/ikin_control_r.cpp)
ament_target_dependencies(ikin_control_r rclcpp std_msgs geometry_msgs std_srvs)

add_executable(virtuoso_ui_node_r src/virtuoso_ui_node_r.cpp)
ament_target_dependencies(virtuoso_ui_node_r rclcpp std_msgs geometry_msgs std_srvs)

add_executable(combine_control src/combine_control.cpp)
ament_target_dependencies(combine_control rclcpp std_msgs)

target_include_directories(virtuoso_ui_node_l PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
  target_include_directories(ikin_control_l PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
  target_include_directories(ikin_control_r PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_include_directories(virtuoso_ui_node_r PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_compile_features(virtuoso_ui_node_l PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_compile_features(virtuoso_ui_node_r PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

# ament_target_dependencies(virtuoso_ui_node_l
# virtuoso_ui_node_r
# rclcpp
# geometry_msgs
# std_msgs
# std_srvs
# )

install(TARGETS virtuoso_ui_node_l virtuoso_ui_node_r combine_control ikin_control_l ikin_control_r
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  src/Virtuoso_UI_R.py
  src/Virtuoso_UI_L.py
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
