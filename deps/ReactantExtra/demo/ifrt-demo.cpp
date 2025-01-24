#include "API.h"

int main() {
    // 1. init MLIR registry and passes
    MlirDialectRegistry registry = MlirDialectRegistryCreate();
    InitializeRegistryAndPasses(registry);

    // 2. init PjRt client (CPU)
    // Q1: what is `node_id` here? `num_nodes` is the number of processes?
    uint8_t async = false;
    int node_id = 0;
    int num_nodes = 1;
    xla::PjRtClient* client = MakeCPUClient(async, node_id, num_nodes);

    // 3. parse MLIR
    MlirContext mlir_ctx = mlirContextCreateWithRegistry(registry, false);
    RegisterDialects(mlir_ctx);
    const char* mlir_code = "
        module {
            func.func @main(%arg0: tensor<4x4xf64>) -> tensor<4x4xf64> {
                %0 = stablehlo.sine %arg0 : tensor<4x4xf64>
                return %0 : tensor<4x4xf64>
            }
        }
    ";
    MlirModule mlir_mod = mlirModuleCreateParse(mlir_code, registry);

    // 4. compile MLIR module to XLA executable
    xla::PjRtLoadedExecutable* loaded_exec = ClientCompile(client, mlir_code);

    // 5. create input array
    float64_t* ptr = new float64_t[16];
    int64_t shape[2] = {4, 4};
    size_t dim = 2;
    uint64_t prim_type = 12;  // float64
    for (int i = 0; i < 16; i++) {
        ptr[i] = 1.0 + i;
    }

    int default_device_idx = 0;
    xla::PjRtDevice* device = ClientGetDevice(client, default_device_idx);

    xla::PjRtBuffer* buffer = ArrayFromHostBuffer(client, ptr, prim_type, dim, &shape, device);

    // 6. execute computation
    int num_args = 1;
    PjRtBuffer** op_args = new PjRtBuffer*[num_args];
    op_args[0] = buffer;
    uint8_t* is_arg_donatable = new uint8_t[num_args];
    is_arg_donatable[0] = false;
    int num_results = 1;
    PjRtBuffer** op_results = new PjRtBuffer*[num_results];
    uint8_t futures;
    FutureType** future_results = new FutureType*[num_results];
    XLAExecute(loaded_exec, num_args, op_args, is_arg_donatable, num_results, op_results, &futures, future_results);
}

// TODO ask about communication: Google TPU cluster have several communication layers, we just want control plan for the high-speed traffic in the clusters

// Is IFRT usable in clusters aside of TPU clusters?

// -> There is a ifrt::Topology class -> QUESTION: where does Topology and Sharding class fit and how do we configure this?

// Array = [Shard, Shard] // divided by block array where each is block is a shard
// [Array, Array, ...] = loaded_executable->Execute([Array, Array, ...])

