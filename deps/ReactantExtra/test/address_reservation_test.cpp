// Unit tests for virtual address range reservation and release in
// AddressReservation.cpp.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

// Reserves an inaccessible virtual address range of `size_in_bytes` bytes.
// Returns nullptr on failure.
void *reactantReserveAddressRange(size_t size_in_bytes);

// Releases a virtual address range previously reserved by
// `reactantReserveAddressRange`.
void reactantReleaseAddressRange(void *base, size_t size_in_bytes);

TEST(AddressReservationTest, ReserveAndReleaseSingleByte) {
  void *address = reactantReserveAddressRange(1);
  ASSERT_NE(address, nullptr);
  reactantReleaseAddressRange(address, 1);
}

TEST(AddressReservationTest, ReserveAndReleasePageAlignedAndMultiPage) {
  constexpr size_t kPageSizeInBytes = 4096;
  constexpr size_t kSizesInBytes[] = {kPageSizeInBytes, 16 * kPageSizeInBytes,
                                      16 * 1024 * 1024};
  for (size_t size_in_bytes : kSizesInBytes) {
    SCOPED_TRACE(size_in_bytes);
    void *address = reactantReserveAddressRange(size_in_bytes);
    ASSERT_NE(address, nullptr);
    uintptr_t start = reinterpret_cast<uintptr_t>(address);
    EXPECT_EQ(start % kPageSizeInBytes, 0u);
    uintptr_t end = reinterpret_cast<uintptr_t>(static_cast<char *>(address) +
                                                size_in_bytes);
    EXPECT_EQ(end - start, size_in_bytes);
    reactantReleaseAddressRange(address, size_in_bytes);
  }
}

TEST(AddressReservationTest, MultipleConcurrentReservationsAreDisjoint) {
  constexpr size_t kChunkSizeInBytes = 64 * 1024;
  constexpr int kNumChunks = 8;
  std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
  std::vector<void *> addresses;
  ranges.reserve(kNumChunks);
  addresses.reserve(kNumChunks);

  for (int i = 0; i < kNumChunks; ++i) {
    void *address = reactantReserveAddressRange(kChunkSizeInBytes);
    ASSERT_NE(address, nullptr) << "at chunk " << i;
    addresses.push_back(address);
    uintptr_t start = reinterpret_cast<uintptr_t>(address);
    ranges.emplace_back(start, start + kChunkSizeInBytes);
  }

  for (int i = 0; i < kNumChunks; ++i) {
    for (int j = i + 1; j < kNumChunks; ++j) {
      SCOPED_TRACE(i);
      SCOPED_TRACE(j);
      bool is_disjoint = (ranges[i].second <= ranges[j].first) ||
                         (ranges[j].second <= ranges[i].first);
      EXPECT_TRUE(is_disjoint);
    }
  }

  for (void *address : addresses) {
    reactantReleaseAddressRange(address, kChunkSizeInBytes);
  }
}

TEST(AddressReservationDeathTest, ReservedMemoryIsInaccessible) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  constexpr size_t kPageSizeInBytes = 4096;
  void *address = reactantReserveAddressRange(kPageSizeInBytes);
  ASSERT_NE(address, nullptr);
  EXPECT_DEATH({ *static_cast<volatile char *>(address) = 42; }, "");
  EXPECT_DEATH({ (void)*static_cast<volatile char *>(address); }, "");
  reactantReleaseAddressRange(address, kPageSizeInBytes);
}
