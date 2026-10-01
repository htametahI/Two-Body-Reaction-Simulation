cmake_minimum_required(VERSION 3.16)
project(S2223_Sim LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(Geant4 REQUIRED ui_all vis_all)
include(${Geant4_USE_FILE})

# An explicit source list makes additions/removals visible to the build system.
# The common library target also lets regression tests exercise production code.
add_library(sim_core STATIC
  src/action.cxx src/analysis.cxx src/construction.cxx src/event.cxx
  src/generator.cxx src/input.cxx src/mode.cxx src/output.cxx
  src/physics.cxx src/reconstruction.cxx src/run.cxx src/s3.cxx src/stepping.cxx)
target_compile_features(sim_core PUBLIC cxx_std_17)
target_include_directories(sim_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(sim_core PUBLIC ${Geant4_LIBRARIES})
add_executable(sim sim.cxx)
target_link_libraries(sim PRIVATE sim_core)
add_custom_target(S2223_Sim DEPENDS sim)

configure_file(run.mac run.mac COPYONLY)
configure_file(vis.mac vis.mac COPYONLY)

include(CTest)
if(BUILD_TESTING)
  foreach(check core kinematics single_ring)
    add_executable(test_${check} tests/${check}.cxx)
    target_link_libraries(test_${check} PRIVATE sim_core)
    file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/test-output/${check})
    add_test(NAME ${check} COMMAND test_${check})
    set_tests_properties(${check} PROPERTIES
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test-output/${check})
  endforeach()
  # ROOT-file verification requires Python with uproot. The native tests above
  # remain available on systems that do not have the analysis Python packages.
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import uproot, pandas"
      RESULT_VARIABLE HAVE_ANALYSIS_PYTHON OUTPUT_QUIET ERROR_QUIET)
    if(HAVE_ANALYSIS_PYTHON EQUAL 0)
      add_test(NAME single_ring_outputs COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/verify_single_ring.py
        ${PROJECT_BINARY_DIR}/test-output/single_ring)
      set_tests_properties(single_ring PROPERTIES FIXTURES_SETUP single_ring_data)
      set_tests_properties(single_ring_outputs PROPERTIES FIXTURES_REQUIRED single_ring_data)
      add_test(NAME analysis_beam_provenance COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/analysis_io.py)
    else()
      message(STATUS "Python ROOT/analysis checks disabled: install uproot and pandas or set Python3_EXECUTABLE")
    endif()
    add_test(NAME simulation_workflow COMMAND ${Python3_EXECUTABLE}
      ${PROJECT_SOURCE_DIR}/tests/workflow.py $<TARGET_FILE:sim> ${PROJECT_SOURCE_DIR})
  endif()
endif()
