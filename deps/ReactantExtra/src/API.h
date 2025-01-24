#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "xla/tsl/profiler/rpc/profiler_server.h"

typedef PjRtFuture<> FutureType;

extern "C" void ReactantHandleCuResult(uint32_t curesult);
extern "C" MlirAttribute mlirComplexAttrDoubleGet(MlirContext ctx,MlirType type,double real,double imag);
extern "C" MlirAttribute mlirComplexAttrDoubleGetChecked(MlirLocation loc,MlirType type,double real,double imag);
extern "C" void InitializeLogs();
extern "C" void SetLogLevel(int level);
extern "C" void SetModuleLogLevel(const char *module_pattern, int level);
extern "C" char *GetDefaultTargetTriple(void);
extern "C" MLIR_CAPI_EXPORTED MlirAttribute enzymeActivityAttrGet(MlirContext ctx, int32_t val);
extern "C" tsl::ProfilerSession* CreateProfilerSession(uint32_t device_tracer_level, uint32_t host_tracer_level);
extern "C" void ProfilerSessionCollectData(tsl::ProfilerSession *session,const char *path);
extern "C" void ProfilerSessionDelete(tsl::ProfilerSession *session);
extern "C" int64_t ProfilerActivityStart(const char *name, int level);
extern "C" void ProfilerActivityEnd(int64_t id);
extern "C" tsl::profiler::ProfilerServer *ProfilerServerStart(int32_t port);
extern "C" void ProfilerServerStop(tsl::profiler::ProfilerServer *server);
extern "C" PjRtClient *MakeCPUClient(uint8_t asynchronous, int node_id, int num_nodes);
extern "C" PjRtClient *MakeGPUClient(int node_id, int num_nodes,int *allowed_devices,int num_allowed_devices,const char *platform_name,const char **error);
extern "C" const PJRT_Api *LoadPjrtPlugin(const char *device_type,const char *library_path,const char **error);
extern "C" int InitializePjrtPlugin(const char *device_type,const char**error);
extern "C" PjRtClient *GetCApiClient(const char *device_type);
extern "C" PjRtClient *MakeTPUClient(const char *tpu_path, const char **error);
extern "C" int ClientNumDevices(PjRtClient *client);
extern "C" int ClientNumAddressableDevices(PjRtClient *client);
extern "C" int ClientProcessIndex(PjRtClient *client);
extern "C" PjRtDevice *ClientGetDevice(PjRtClient *client, int device_id);
extern "C" PjRtDevice *ClientGetAddressableDevice(PjRtClient *client,int device_id);
extern "C" void PjRtDeviceGetAllocatorStats(PjRtDevice *device,JLAllocatorStats *jlstats);
extern "C" void ExecutableFree(xla::PjRtLoadedExecutable *exec);
extern "C" PjRtDevice *BufferToDevice(PjRtBuffer *Buffer);
extern "C" PjRtClient *BufferToClient(PjRtBuffer *Buffer);
extern "C" PjRtClient *DeviceToClient(PjRtDevice *Device);
extern "C" void PjRtBufferFree(PjRtBuffer *Buffer);
extern "C" int32_t ReactantCudaDriverGetVersion();
extern "C" void *UnsafeBufferPointer(PjRtBuffer *buffer);
extern "C" PjRtBuffer *ArrayFromHostBuffer(PjRtClient *client, void *data,uint64_t ptype, size_t dim,int64_t *cshape,PjRtDevice *device);
extern "C" uint8_t BufferOnCPU(PjRtBuffer *buffer);
extern "C" PjRtBuffer *CopyBufferToDevice(PjRtBuffer *buffer,PjRtDevice *dst_device);
extern "C" void BufferToHost(PjRtBuffer *buffer, void *data);
extern "C" void FreeClient(PjRtClient *client);
extern "C" void RegisterCustomCallTarget(const char *name, void *address, const char *platform);
extern "C" MlirModule ConvertLLVMToMLIR(LLVMModuleRef lmod, MlirContext cctx);
extern "C" MlirModule ConvertLLVMStrToMLIR(const char *lmod, MlirContext cctx);
extern "C" xla::PjRtLoadedExecutable *ClientCompile(PjRtClient *client,MlirModule cmod);
extern "C" void FreeFuture(FutureType *Future);
extern "C" uint8_t FutureIsReady(FutureType *Future);
extern "C" void FutureAwait(FutureType *Future);
extern "C" void XLAExecute(xla::PjRtLoadedExecutable *exec, int num_args,PjRtBuffer **op_args, uint8_t *is_arg_donatable,int num_results, PjRtBuffer **op_results,uint8_t *futures, FutureType **future_results);
extern "C" void RegisterDialects(MlirContext cctx);
extern "C" void InitializeRegistryAndPasses(MlirDialectRegistry creg);
extern "C" void ReactantFuncSetArgAttr(MlirOperation op, intptr_t pos, MlirStringRef name, MlirAttribute attr);
extern "C" MlirOperation LinkInModule(MlirModule prevModC, MlirModule newModC,const char *entryfn);
