/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_SHARED_MEMORY_H_
#define XENIA_GPU_METAL_METAL_SHARED_MEMORY_H_

// Metal shared memory attempts bytes-no-copy aliasing on unified-memory
// devices and falls back to staged uploads when unsupported.

#include <vector>

#include "xenia/gpu/shared_memory.h"
#include "xenia/gpu/trace_writer.h"
#include "xenia/ui/metal/metal_api.h"

namespace xe {
namespace gpu {
namespace metal {

class MetalCommandProcessor;
class MetalSharedMemory : public SharedMemory {
 public:
  MetalSharedMemory(MetalCommandProcessor& command_processor, Memory& memory,
                    TraceWriter& trace_writer);
  ~MetalSharedMemory() override;
  bool Initialize();
  void Shutdown();
  void ClearCache() override;

  MTL::Buffer* GetBuffer() const { return buffer_; }
  const uint8_t* GetXboxRamBase() const {
    return static_cast<const uint8_t*>(memory().TranslatePhysical(0));
  }

  // For trace dump, simplified - just make buffer available for reading
  void UseForReading() {
    // No state transitions needed in Metal
  }
  // Override pure virtual function from SharedMemory
  bool UploadRanges(const std::pair<uint32_t, uint32_t>* upload_page_ranges,
                    uint32_t num_upload_ranges) override;

  // Returns true if there is GPU-written data for the trace. The buffer is
  // CPU-visible, so nothing is submitted - the caller must await the GPU.
  bool InitializeTraceSubmitDownloads();
  void InitializeTraceCompleteDownloads();

  // The GPU reads this buffer when it executes a draw, which can be after the
  // guest has already reused the memory for later data. Draws record the
  // submission that reads each page here, and uploads that would overwrite
  // such pages first wait for that submission. Only valid without zero-copy
  // aliasing, where the guest can't modify the buffer directly.
  bool is_ordering_uploads() const { return !use_zero_copy_; }
  void MarkRangeUsedByGpu(uint32_t start, uint32_t length,
                          uint64_t submission);

 private:
  MetalCommandProcessor& command_processor_;
  TraceWriter& trace_writer_;
  MTL::Buffer* buffer_ = nullptr;
  bool use_zero_copy_ = false;

  // The latest submission whose draws read the range, 0 if none did.
  uint64_t GetRangeLastGpuUse(uint32_t start, uint32_t length) const;

  // Per guest page, the latest submission whose draws read it. Only touched
  // from the command processor thread.
  std::vector<uint64_t> gpu_last_used_submission_;
  bool gpu_used_any_ = false;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif
