cmake_minimum_required(VERSION 3.16)

project(Strategy_assessment LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define Python virtual environment interpreter
set(PYTHON_VENV ${CMAKE_CURRENT_SOURCE_DIR}/venv/bin/python)
set(Python_EXECUTABLE ${PYTHON_VENV})

# Define C++ executable target
add_executable(Strategy_assessment
    main.cpp
    experiment.cpp
    experiment1.cpp
)

# Optional: run csv_plotting.py automatically after build (not after execution)
add_custom_command(
    TARGET Strategy_assessment POST_BUILD
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/csv_plotting.py
    COMMENT "Running Python visualization after C++ build..."
)

#[[ Optional: target to run only plotting manually
add_custom_target(run_plot
    COMMAND ${PYTHON_VENV} ${CMAKE_CURRENT_SOURCE_DIR}/csv_plotting.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/csv_plotting.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running Python plot script manually using virtualenv"
)]]

# Main experiment + visualization as full chain
add_custom_target(run_all
    COMMAND Strategy_assessment
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/csv_plotting.py
    DEPENDS Strategy_assessment
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "▶️ Running Strategy_assessment + CSV visualization"
)


# Install binary and optional Python script
include(GNUInstallDirs)
install(TARGETS Strategy_assessment
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES csv_plotting.py DESTINATION ${CMAKE_INSTALL_BINDIR})


#[[cmake_minimum_required(VERSION 3.16)

project(Strategy_assessment LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PYTHON_VENV ${CMAKE_CURRENT_SOURCE_DIR}/venv/bin/python)
#set(Python_EXECUTABLE ${PYTHON_VENV})

# Define C++ executable target
add_executable(Strategy_assessment
    main.cpp
    experiment.cpp
    experiment1.cpp
)

#///add_custom_command(
    TARGET Strategy_assessment POST_BUILD
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/csv_plotting.py
    COMMENT "Running Python visualization after C++ run..."
///)


# Python interpreter from local virtual environment

# Optional: Custom target to run the Python plot script after building C++
add_custom_target(run_plot
    COMMAND ${PYTHON_VENV} ${CMAKE_CURRENT_SOURCE_DIR}/csv_plotting.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/csv_plotting.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running Python plot script using virtualenv"
)

# Install binary and optional Python script
include(GNUInstallDirs)
install(TARGETS Strategy_assessment
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES csv_plotting.py DESTINATION ${CMAKE_INSTALL_BINDIR})
]]


