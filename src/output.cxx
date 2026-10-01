#include "output.hh"
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace Output {
void EnsureParentDirectory(const std::string &path) {
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) {
    std::error_code error;
    std::filesystem::create_directories(parent, error);
    if (error) {
      throw std::runtime_error("Cannot create output directory for " + path +
                               ": " + error.message());
    }
  }
}

void Open(std::ofstream &stream, const std::string &path) {
  if (stream.is_open()) {
    throw std::logic_error("Output stream is already open: " + path);
  }
  EnsureParentDirectory(path);
  stream.clear();
  stream.open(path, std::ios::trunc);
  if (!stream) {
    throw std::runtime_error("Cannot open output file: " + path);
  }
  stream << std::setprecision(17);
}

void Check(const std::ofstream &stream, const std::string &path) {
  if (!stream) {
    throw std::runtime_error("Failed writing output file: " + path);
  }
}

void Close(std::ofstream &stream, const std::string &path) {
  if (!stream.is_open()) return;
  stream.flush();
  Check(stream, path);
  stream.close();
  Check(stream, path);
}

std::string WithSuffix(const std::string &path, const std::string &suffix) {
  auto result = std::filesystem::path(path);
  result.replace_extension();
  return result.string() + suffix;
}

std::string JsonString(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 0x20) {
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
    } else out << c;
  }
  out << '"';
  return out.str();
}

std::string FileIdentityJson(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot fingerprint input file: " + path);
  // FNV-1a is a content fingerprint for provenance, not a security checksum.
  std::uint64_t hash = UINT64_C(14695981039346656037);
  std::uint64_t size = 0;
  char buffer[65536];
  while (input.read(buffer, sizeof(buffer)) || input.gcount()) {
    for (std::streamsize i = 0; i < input.gcount(); ++i) {
      hash ^= static_cast<unsigned char>(buffer[i]);
      hash *= UINT64_C(1099511628211);
      ++size;
    }
  }
  if (input.bad()) throw std::runtime_error("Failed reading input file: " + path);
  std::ostringstream fingerprint;
  fingerprint << std::hex << std::setw(16) << std::setfill('0') << hash;
  return "{\"path\":" + JsonString(std::filesystem::absolute(path).lexically_normal().string()) +
         ",\"size_bytes\":" + std::to_string(size) +
         ",\"fnv1a64\":" + JsonString(fingerprint.str()) + "}";
}
}
