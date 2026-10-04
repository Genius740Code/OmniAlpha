#pragma once
#include <cstdint>
#include <limits>
namespace gai {
// splitmix64; satisfies UniformRandomBitGenerator so it works with <random> distributions.
struct Rng {
  using result_type = uint64_t;
  uint64_t s;
  explicit Rng(uint64_t seed = 0x9E3779B97F4A7C15ULL) : s(seed) {}
  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return std::numeric_limits<uint64_t>::max(); }
  result_type operator()() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  uint32_t below(uint32_t n) { return (uint32_t)(((uint64_t)(uint32_t)((*this)() >> 32) * n) >> 32); }
  double uniform() { return ((*this)() >> 11) * (1.0 / 9007199254740992.0); }
};
inline uint64_t mix64(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
}  // namespace gai
