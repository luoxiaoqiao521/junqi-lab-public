// Explicit, platform-independent PRNG (splitmix64). std distributions are
// avoided because their output differs between standard library versions.
#pragma once
#include <cstdint>
#include <cstddef>

namespace jq {

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  // Uniform in [0, n) by rejection; n > 0.
  uint32_t below(uint32_t n) {
    uint64_t limit = UINT64_MAX - UINT64_MAX % n;
    uint64_t x;
    do { x = next(); } while (x >= limit);
    return static_cast<uint32_t>(x % n);
  }
};

inline uint64_t mix_seed(uint64_t a, uint64_t b) {
  Rng r(a * 0x100000001B3ull ^ b);
  return r.next();
}

inline uint64_t fnv1a(const void* data, size_t n, uint64_t h = 0xCBF29CE484222325ull) {
  const unsigned char* p = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 0x100000001B3ull; }
  return h;
}

}  // namespace jq
