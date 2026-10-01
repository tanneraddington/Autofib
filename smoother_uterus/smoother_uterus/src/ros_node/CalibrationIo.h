#pragma once

#include <yaml-cpp/yaml.h>

#include <solver/SmootherSolver.h>

struct LoadedCalibration {
    SmootherCalibration calibration;
    int64_t age_seconds;
};

void save_calibration_yaml(const SmootherCalibration& c, const std::string& filename);

LoadedCalibration load_calibration_yaml(const std::string& filename);