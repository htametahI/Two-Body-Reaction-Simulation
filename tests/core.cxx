#include "input.hh"
#include "output.hh"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>

namespace {
void Require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error(message);
}
void Put(const std::string &path, const std::string &text) {
  std::ofstream out;
  Output::Open(out, path);
  out << text;
  Output::Close(out, path);
}
void Reject(const std::function<void()> &call, const std::string &diagnostic) {
  try { call(); } catch (const std::exception &error) {
    Require(std::string(error.what()).find(diagnostic) != std::string::npos,
            "Missing diagnostic '" + diagnostic + "' in " + error.what());
    return;
  }
  throw std::runtime_error("Invalid input/output was accepted: " + diagnostic);
}
}

int main() {
  try {
    Put("valid_beam.dat", "  # comment\n66 0 0 0.25 0 0 1 0.5\n");
    const auto beam = ReadBeamFile("valid_beam.dat");
    Require(beam.size() == 1 && beam[0].depth == 0.5, "Beam parsing/units");
    for (const auto &line : {"66 0 0 0 0 0 0 0", "-1 0 0 0 0 0 1 0",
                             "nan 0 0 0 0 0 1 0", "66 0 0 0 0 0 1 -1",
                             "66 0 0 0 0 0 1", "66 0 0 0 0 0 1 0 garbage"}) {
      Put("bad_beam.dat", std::string("# header\n") + line + '\n');
      Reject([] { ReadBeamFile("bad_beam.dat"); }, "bad_beam.dat:2:");
    }
    Reject([] { ReadBeamFile("missing_beam.dat"); }, "Cannot open beam");
    Put("valid_dwba.dat", "  @title x\n0 1\n90 1\n180 1\n END\n");
    const auto distribution = ReadDwbaFile("valid_dwba.dat");
    Require(distribution.cdf.size() == 3 && distribution.cdf.front() == 0. &&
            distribution.cdf.back() == 1. && std::abs(distribution.cdf[1] - 0.5) < 1e-12,
            "DWBA CDF normalization");
    for (const auto &data : {"0 1\n0 2\n", "90 1\n45 1\n", "0 -1\n90 1\n",
                             "0 1\n181 1\n", "0 inf\n90 1\n", "0 1\n90 junk\n"}) {
      Put("bad_dwba.dat", data);
      Reject([] { ReadDwbaFile("bad_dwba.dat"); }, "bad_dwba.dat:");
    }
    Put("zero_dwba.dat", "0 0\n90 0\n180 0\n");
    Reject([] { ReadDwbaFile("zero_dwba.dat"); }, "positive");
    Put("parent_is_file", "blocked");
    Reject([] { std::ofstream out; Output::Open(out, "parent_is_file/result.csv"); }, "Cannot create output directory");
    std::filesystem::create_directory("is_directory");
    Reject([] { std::ofstream out; Output::Open(out, "is_directory"); }, "Cannot open output file");
    Reject([] { std::ofstream out; out.setstate(std::ios::badbit); Output::Check(out, "failed.csv"); }, "failed.csv");
    Require(Output::WithSuffix("dir.with.dots/result.root", "_truth.csv") == "dir.with.dots/result_truth.csv", "Output path suffix");
    Put("nested/results/good.csv", "a,b\n1,2\n");
    const auto identity = Output::FileIdentityJson("nested/results/good.csv");
    Require(identity.find("size_bytes\":8") != std::string::npos, "File identity size");
    std::cout << "PASS: validated beam/DWBA input and checked output failures\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
