#ifndef OUTPUT_HH
#define OUTPUT_HH

#include <fstream>
#include <string>

// Check both buffered writes and the final flush/close: an open file alone
// does not guarantee that the event data reached the filesystem.
namespace Output {
void EnsureParentDirectory(const std::string &path);
void Open(std::ofstream &stream, const std::string &path);
void Check(const std::ofstream &stream, const std::string &path);
void Close(std::ofstream &stream, const std::string &path);
std::string WithSuffix(const std::string &path, const std::string &suffix);
std::string JsonString(const std::string &value);
std::string FileIdentityJson(const std::string &path);
}

#endif
