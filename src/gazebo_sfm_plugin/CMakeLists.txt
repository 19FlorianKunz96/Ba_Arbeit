cmake_minimum_required(VERSION 3.8)
project(gazebo_sfm_plugin)

# Entspricht -std=c++11 aus deiner Vorlage (hier etwas moderner):
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ament statt catkin
find_package(ament_cmake REQUIRED)

# Gazebo (Classic)
find_package(gazebo REQUIRED)

# Wie in deiner Vorlage: Boost::thread (falls nicht genutzt, kannst du es entfernen)
find_package(Boost REQUIRED COMPONENTS thread)

# OPTIONAL: Falls das Social-Force-Model als .so/.a vorliegt, hier automatisch finden und linken.
# Wenn header-only, bleibt das einfach ohne Effekt.
find_library(SFM_LIBRARY
  NAMES lightsfm sfm social_force
  PATHS /usr/local/lib /usr/lib /usr/local/lib64 /usr/lib64
)
find_path(SFM_INCLUDE_DIR
  NAMES sfm/SFM.h utils/Angle.h
  PATHS /usr/local/include /usr/include
)

add_library(PedestrianSFMPlugin SHARED
  src/PedestrianSFMPlugin.cpp
)

# Entspricht deinen include_directories()
target_include_directories(PedestrianSFMPlugin
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    /usr/local/include           # wie in deiner Vorlage (für SFM-Header)
    ${GAZEBO_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

# Entspricht deinem target_link_libraries(), nur ohne catkin
target_link_libraries(PedestrianSFMPlugin
  ${GAZEBO_LIBRARIES}
  ${Boost_LIBRARIES}
)

# Wenn SFM-Lib gefunden wurde, zusätzlich linken & Includes setzen
if(SFM_LIBRARY)
  target_link_libraries(PedestrianSFMPlugin ${SFM_LIBRARY})
endif()
if(SFM_INCLUDE_DIR)
  target_include_directories(PedestrianSFMPlugin PUBLIC ${SFM_INCLUDE_DIR})
endif()

# Installation (ament-Standardziele statt CATKIN_*_DESTINATION)
install(TARGETS PedestrianSFMPlugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

# Launch
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/launch)
  install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
endif()

# Worlds
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/worlds)
  install(DIRECTORY worlds DESTINATION share/${PROJECT_NAME})
endif()

# Models
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models)
  install(DIRECTORY models DESTINATION share/${PROJECT_NAME})
endif()

# Media
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/media)
  install(DIRECTORY media DESTINATION share/${PROJECT_NAME})
endif()


# ament-Exports (ersetzt catkin_package)
ament_export_include_directories(include)
ament_export_libraries(PedestrianSFMPlugin)
ament_package()
