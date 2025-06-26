cmake_minimum_required(VERSION 3.16)

project(Strategy_assessment LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define C++ executable target
add_executable(Strategy_assessment
    main.cpp
    experiment.cpp
    experiment1.cpp
)

# Python interpreter from local virtual environment
set(PYTHON_VENV ${CMAKE_CURRENT_SOURCE_DIR}/venv/bin/python)

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

