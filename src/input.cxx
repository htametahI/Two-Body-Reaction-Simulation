#include "input.hh"
#include "G4SystemOfUnits.hh"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {
[[noreturn]] void Invalid(const std::string &path, std::size_t line,
                          const std::string &reason) {
  throw std::runtime_error(path + ":" + std::to_string(line) + ": " + reason);
}

std::string Trim(const std::string &line) {
  const auto begin = line.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return {};
  return line.substr(begin, line.find_last_not_of(" \t\r\n") - begin + 1);
}

std::vector<double> Values(const std::string &line, const std::string &path,
                           std::size_t lineNumber) {
  std::istringstream input(line.substr(0, line.find('#')));
  std::vector<double> values;
  std::string token;
  while (input >> token) {
    try {
      std::size_t consumed = 0;
      const double value = std::stod(token, &consumed);
      if (consumed != token.size() || !std::isfinite(value)) {
        Invalid(path, lineNumber, "expected a finite number, got '" + token + "'");
      }
      values.push_back(value);
    } catch (const std::invalid_argument &) {
      Invalid(path, lineNumber, "invalid number '" + token + "'");
    } catch (const std::out_of_range &) {
      Invalid(path, lineNumber, "number out of range: '" + token + "'");
    }
  }
  return values;
}
}

std::vector<BeamState> ReadBeamFile(const std::string &path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot open beam file: " + path);
  std::vector<BeamState> states;
  std::string line;
  std::size_t lineNumber = 0;
  while (std::getline(input, line)) {
    ++lineNumber;
    line = Trim(line);
    if (line.empty() || line.front() == '#') continue;
    const auto v = Values(line, path, lineNumber);
    if (v.size() != 8) Invalid(path, lineNumber, "expected 8 beam columns");
    if (v[0] <= 0.0) Invalid(path, lineNumber, "beam kinetic energy must be positive");
    if (v[7] < 0.0) Invalid(path, lineNumber, "reaction depth must be nonnegative");
    const double norm = std::hypot(v[4], v[5], v[6]);
    if (!std::isfinite(norm) || norm == 0.0) {
      Invalid(path, lineNumber, "beam direction must have a finite, nonzero length");
    }
    states.push_back({v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]});
  }
  if (input.bad()) throw std::runtime_error("Failed reading beam file: " + path);
  if (states.empty()) throw std::runtime_error(path + ": no beam rows");
  return states;
}

// read in DWBA files
DwbaDistribution ReadDwbaFile(const std::string &path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot open DWBA file: " + path);
  DwbaDistribution result;
  std::vector<double> sigma;
  std::string line;
  std::size_t lineNumber = 0;
  while (std::getline(input, line)) {
    ++lineNumber;
    line = Trim(line);
    if (line.empty() || line.front() == '#' || line.front() == '@') continue;
    if (line == "END") break;
    const auto v = Values(line, path, lineNumber);
    if (v.size() < 2) Invalid(path, lineNumber, "expected angle and cross section");
    if (v[0] < 0.0 || v[0] > 180.0) Invalid(path, lineNumber, "angle must be in [0, 180] degrees");
    if (v[1] < 0.0) Invalid(path, lineNumber, "cross section must be nonnegative");
    const auto theta = v[0] * deg;
    if (!result.theta.empty() && theta <= result.theta.back()) {
      Invalid(path, lineNumber, "angles must be strictly increasing");
    }
    result.theta.push_back(theta);
    sigma.push_back(v[1]);
  }
  if (input.bad()) throw std::runtime_error("Failed reading DWBA file: " + path);
  if (result.theta.size() < 2) throw std::runtime_error(path + ": need at least two angles");
  result.cdf.resize(result.theta.size(), 0.0);
  for (std::size_t i = 1; i < result.theta.size(); ++i) {
    const auto t0 = result.theta[i - 1];
    const auto t1 = result.theta[i];
    // dOmega contributes sin(theta); the constant 2*pi cancels on normalization.
    const auto area = 0.5 * (sigma[i - 1] * std::sin(t0) +
                             sigma[i] * std::sin(t1)) * (t1 - t0);
    result.cdf[i] = result.cdf[i - 1] + area;
  }
  const auto total = result.cdf.back();
  if (!std::isfinite(total) || total <= 0.0) {
    throw std::runtime_error(path + ": integrated angular weight must be finite and positive");
  }
  for (auto &value : result.cdf) value /= total;
  return result;
}
