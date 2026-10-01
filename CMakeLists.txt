cmake_minimum_required(VERSION 3.16)
project(TwoBodyReactionSimulation LANGUAGES CXX)

# Geant4 with interactive visualization support.
find_package(Geant4 REQUIRED ui_all vis_all)
include(${Geant4_USE_FILE})

# Build the simulation.
add_executable(sim
    sim.cxx
    src/action.cxx
    src/analysis.cxx
    src/construction.cxx
    src/event.cxx
    src/generator.cxx
    src/input.cxx
    src/mode.cxx
    src/output.cxx
    src/physics.cxx
    src/reconstruction.cxx
    src/run.cxx
    src/s3.cxx
    src/stepping.cxx
)

target_compile_features(sim PRIVATE cxx_std_17)
target_include_directories(sim PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(sim PRIVATE ${Geant4_LIBRARIES})

# Place the runtime macros beside the executable.
configure_file(run.mac run.mac COPYONLY)
configure_file(vis.mac vis.mac COPYONLY)
