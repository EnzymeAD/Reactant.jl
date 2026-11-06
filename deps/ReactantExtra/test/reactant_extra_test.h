// Shared C ABI declarations and test utilities for ReactantExtra unit tests.

#ifndef THIRD_PARTY_REACTANT_JL_DEPS_REACTANTEXTRA_TEST_REACTANT_EXTRA_TEST_H_
#define THIRD_PARTY_REACTANT_JL_DEPS_REACTANTEXTRA_TEST_REACTANT_EXTRA_TEST_H_

#include <cstddef>
#include <cstdint>
#include <string>

// Opaque C++ runtime handle managed by reactantXLAInit and reactantXLADeInit.
struct LinkableRuntime;

extern "C" {

// Global error callback invoked by ReactantExtra when a C ABI entry point
// encounters a recoverable error.
extern void (*ReactantThrowError)(const char *);

// LinkableRuntime C ABI (API.cpp).
void reactantXLAThrow(const char *message);
void reactantXLAInit(LinkableRuntime **runtime, const char *backend);
void reactantXLADeInit(LinkableRuntime **runtime);
void *reactantXLAMalloc(LinkableRuntime **runtime, uint64_t primitive_type,
                        uint64_t rank, uint64_t *shape);
void reactantXLAFree(LinkableRuntime **runtime, void *buffer);
void reactantXLAMemcpy(LinkableRuntime **runtime, void *destination,
                       void *source, size_t size_in_bytes, int32_t direction);
void reactantXLAExec(LinkableRuntime **runtime, const char *module_text,
                     int64_t num_arguments, void **arguments,
                     int64_t num_constants, const int64_t *constants);

} // extern "C"

namespace reactant_test {

// XLA PrimitiveType / IFRT DType::Kind numeric codes.
enum XlaPrimitiveType : uint64_t {
  kPred = 1,
  kS8 = 2,
  kS16 = 3,
  kS32 = 4,
  kS64 = 5,
  kU8 = 6,
  kU16 = 7,
  kU32 = 8,
  kU64 = 9,
  kF16 = 10,
  kF32 = 11,
  kF64 = 12,
};

// Transfer direction codes accepted by reactantXLAMemcpy.
enum XlaMemcpyDirection : int32_t {
  kMemcpyHostToDevice = 1,
  kMemcpyDeviceToHost = 2,
  kMemcpyDeviceToDevice = 3,
};

// RAII guard that installs a non-aborting ReactantThrowError handler and
// records the last error message reported during its scope.
class ScopedErrorHandler {
public:
  ScopedErrorHandler() : previous_handler_(ReactantThrowError) {
    lastError().clear();
    ReactantThrowError = &recordError;
  }
  ~ScopedErrorHandler() { ReactantThrowError = previous_handler_; }

  // Returns the most recent error message captured in this scope, or an empty
  // string if no error was reported.
  static const std::string &error() { return lastError(); }

  // Clears the recorded error message.
  static void clear() { lastError().clear(); }

private:
  static void recordError(const char *message) {
    lastError() = message ? message : "unknown ReactantExtra error";
  }

  static std::string &lastError() {
    static std::string *error = new std::string();
    return *error;
  }

  // Previous error callback restored upon destruction.
  void (*previous_handler_)(const char *);
};

} // namespace reactant_test

#endif // THIRD_PARTY_REACTANT_JL_DEPS_REACTANTEXTRA_TEST_REACTANT_EXTRA_TEST_H_
