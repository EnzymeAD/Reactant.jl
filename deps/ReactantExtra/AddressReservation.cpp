// Reserves inaccessible address ranges used as opaque allocation handles:
// mmap/PROT_NONE on POSIX, VirtualAlloc/MEM_RESERVE on Windows. This lives in
// its own translation unit because <windows.h> defines function-style macros
// (ZeroMemory, IGNORE, ...) that break every header included after it.

#include <cstddef>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#endif

// Returns nullptr on failure.
void *reactantReserveAddressRange(size_t nbytes) {
#ifdef _WIN32
  return VirtualAlloc(nullptr, nbytes, MEM_RESERVE, PAGE_NOACCESS);
#else
  void *base = mmap(nullptr, nbytes, PROT_NONE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  return base == MAP_FAILED ? nullptr : base;
#endif
}

void reactantReleaseAddressRange(void *base, size_t nbytes) {
#ifdef _WIN32
  (void)nbytes;
  VirtualFree(base, 0, MEM_RELEASE);
#else
  munmap(base, nbytes);
#endif
}
