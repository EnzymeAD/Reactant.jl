#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

void *reactantReserveAddressRange(size_t nbytes);
void reactantReleaseAddressRange(void *base, size_t nbytes);

namespace {

TEST(AddressReservationTest, ReserveAndReleaseSingleByte) {
  void *ptr = reactantReserveAddressRange(1);
  ASSERT_NE(ptr, nullptr);
  reactantReleaseAddressRange(ptr, 1);
}

TEST(AddressReservationTest, ReserveAndReleasePageAlignedAndMultiPage) {
  constexpr size_t kSizes[] = {4096, 65536, 16 * 1024 * 1024};
  for (size_t nbytes : kSizes) {
    SCOPED_TRACE(nbytes);
    void *ptr = reactantReserveAddressRange(nbytes);
    ASSERT_NE(ptr, nullptr);
    const uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(start % 4096, 0u);
    const uintptr_t end =
        reinterpret_cast<uintptr_t>(static_cast<char *>(ptr) + nbytes);
    EXPECT_EQ(end - start, nbytes);
    reactantReleaseAddressRange(ptr, nbytes);
  }
}

TEST(AddressReservationTest, MultipleConcurrentReservationsAreDisjoint) {
  constexpr size_t kChunkSize = 64 * 1024;
  constexpr int kNumChunks = 8;
  std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
  std::vector<void *> ptrs;
  ranges.reserve(kNumChunks);
  ptrs.reserve(kNumChunks);

  for (int i = 0; i < kNumChunks; ++i) {
    void *ptr = reactantReserveAddressRange(kChunkSize);
    ASSERT_NE(ptr, nullptr) << "at chunk " << i;
    ptrs.push_back(ptr);
    const uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    ranges.emplace_back(start, start + kChunkSize);
  }

  for (int i = 0; i < kNumChunks; ++i) {
    for (int j = i + 1; j < kNumChunks; ++j) {
      SCOPED_TRACE(i);
      SCOPED_TRACE(j);
      const bool disjoint = (ranges[i].second <= ranges[j].first) ||
                            (ranges[j].second <= ranges[i].first);
      EXPECT_TRUE(disjoint);
    }
  }

  for (void *ptr : ptrs) {
    reactantReleaseAddressRange(ptr, kChunkSize);
  }
}

TEST(AddressReservationDeathTest, ReservedMemoryIsInaccessible) {
  constexpr size_t kSize = 4096;
  void *ptr = reactantReserveAddressRange(kSize);
  ASSERT_NE(ptr, nullptr);
  EXPECT_DEATH(
      { *static_cast<volatile char *>(ptr) = 42; }, "");
  EXPECT_DEATH(
      { (void)*static_cast<volatile char *>(ptr); }, "");
  reactantReleaseAddressRange(ptr, kSize);
}

}  // namespace
