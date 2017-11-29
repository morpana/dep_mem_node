cmake_minimum_required(VERSION 2.8.3)
project(dep_memory)

find_package(catkin REQUIRED COMPONENTS
  roboy_dep
  roboy_communication_middleware
  rospy
)

catkin_python_setup()