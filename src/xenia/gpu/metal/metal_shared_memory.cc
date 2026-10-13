/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_shared_memory.h"

#include <algorithm>
#include <atomic>

#include "xenia/base/logging.h"
#include "xenia/base/memory.h"
#include "xenia/base/profiling.h"
#include "xenia/gpu/gpu_flags.h"
#include "xenia/gpu/metal/metal_command_processor.h"

DEFINE_bool(
    metal_ordered_shared_memory_uploads, true,
    "Keep the Metal shared memory as a separate buffer and wait for the GPU "
    "before overwriting data that queued draws still have to read, instead of "
    "aliasing guest memory (shared_memory_zero_copy). Aliasing lets the GPU "
    "see guest memory rewritten after a draw was issued, which can misplace "
    "or flicker geometry.",
    "Metal");

namespace xe {
namespace gpu {
namespace metal {

MetalSharedMemory::MetalSharedMemory(MetalCommandProcessor& command_processor,
                                     Memory& memory, TraceWriter& trace_writer)
    : SharedMemory(memory),
      command_processor_(command_processor),
      trace_writer_(trace_writer) {}

MetalSharedMemory::~MetalSharedMemory() { Shutdown(); }

bool MetalSharedMemory::Initialize() {
  // Try to alias guest memory on unified-memory devices and fall back to a
  // dedicated shared buffer when not supported.
  // Initialize base class
  InitializeCommon();

  const ui::metal::MetalProvider& provider =
      command_processor_.GetMetalProvider();
  MTL::Device* device = provider.GetDevice();

  if (!device) {
    XELOGE("Metal device is null in MetalSharedMemory::Initialize");
    return false;
  }

  // Create Metal buffer - similar to D3D12's approach
  // On Apple Silicon, ResourceStorageModeShared gives CPU/GPU access
  void* xbox_ram = memory().TranslatePhysical(0);
  if (!xbox_ram) {
    XELOGE("Metal shared memory: Xbox RAM is null");
    return false;
  }

  if (cvars::shared_memory_zero_copy &&
      !cvars::metal_ordered_shared_memory_uploads) {
    size_t system_page_size = xe::memory::page_size();
    if (reinterpret_cast<uintptr_t>(xbox_ram) % system_page_size == 0) {
      buffer_ = device->newBuffer(xbox_ram, kBufferSize,
                                  MTL::ResourceStorageModeShared, nullptr);
      if (buffer_) {
        use_zero_copy_ = true;
        XELOGD("Metal shared memory: using bytes-no-copy buffer");
      } else {
        XELOGW("Metal shared memory: bytes-no-copy buffer creation failed");
      }
    } else {
      XELOGW(
          "Metal shared memory: Xbox RAM not page-aligned for bytes-no-copy");
    }
  }

  if (!buffer_) {
    buffer_ = device->newBuffer(kBufferSize, MTL::ResourceStorageModeShared);
  }
  if (!buffer_) {
    XELOGE("Failed to create Metal shared memory buffer");
    return false;
  }

  // For trace dump, do initial full copy; UploadRanges handles incremental
  // updates for normal runs.
  if (!use_zero_copy_) {
    if (xbox_ram) {
      memcpy(buffer_->contents(), xbox_ram, kBufferSize);
    }
  } else {
    XELOGD("Metal shared memory: skipping initial copy (zero-copy)");
  }

  gpu_used_pages_.assign(
      (size_t(kBufferSize >> page_size_log2()) + 63) / 64, 0);
  gpu_used_any_ = false;
  XELOGI("Metal shared memory: {} uploads",
         use_zero_copy_ ? "zero-copy aliased (unordered)"
                        : "copied and ordered against queued draws");

  return true;
}

void MetalSharedMemory::MarkRangeUsedByGpu(uint32_t start, uint32_t length) {
  if (use_zero_copy_ || !length || start >= kBufferSize) {
    return;
  }
  length = std::min(length, kBufferSize - start);
  uint32_t page_first = start >> page_size_log2();
  uint32_t page_last = (start + length - 1) >> page_size_log2();
  for (uint32_t page = page_first; page <= page_last; ++page) {
    gpu_used_pages_[page >> 6] |= uint64_t(1) << (page & 63);
  }
  gpu_used_any_ = true;
}

bool MetalSharedMemory::IsRangeUsedByGpu(uint32_t start,
                                         uint32_t length) const {
  if (!gpu_used_any_ || !length || start >= kBufferSize) {
    return false;
  }
  length = std::min(length, kBufferSize - start);
  uint32_t page_first = start >> page_size_log2();
  uint32_t page_last = (start + length - 1) >> page_size_log2();
  for (uint32_t page = page_first; page <= page_last; ++page) {
    if (gpu_used_pages_[page >> 6] & (uint64_t(1) << (page & 63))) {
      return true;
    }
  }
  return false;
}

void MetalSharedMemory::ClearGpuUsedRanges() {
  std::fill(gpu_used_pages_.begin(), gpu_used_pages_.end(), uint64_t(0));
  gpu_used_any_ = false;
}

void MetalSharedMemory::ClearCache() { SharedMemory::ClearCache(); }

bool MetalSharedMemory::UploadRanges(
    const std::pair<uint32_t, uint32_t>* upload_page_ranges,
    uint32_t num_upload_ranges) {
  SCOPE_profile_cpu_f("gpu");
  // Copy modified ranges from Xbox memory to Metal buffer when not using
  // bytes-no-copy shared memory.
  if (!buffer_ || num_upload_ranges == 0) {
    return true;
  }

  uint8_t* buffer_data = nullptr;
  uint8_t* xbox_data = nullptr;
  if (!use_zero_copy_) {
    void* xbox_ram = memory().TranslatePhysical(0);
    if (!xbox_ram) {
      XELOGE("MetalSharedMemory::UploadRanges: Xbox RAM is null");
      return false;
    }
    buffer_data = static_cast<uint8_t*>(buffer_->contents());
    xbox_data = static_cast<uint8_t*>(xbox_ram);
  }

  const uint32_t page_size = 1u << page_size_log2();

  uint32_t merged_start = 0;
  uint32_t merged_end = 0;
  bool have_merged = false;

  auto flush_merged_range = [&](uint32_t start, uint32_t end) {
    if (end <= start) {
      return;
    }
    uint32_t length = end - start;
    if (!use_zero_copy_ && IsRangeUsedByGpu(start, length)) {
      // Queued draws still have to read the old contents of these pages.
      static std::atomic<uint32_t> wait_count{0};
      uint32_t count = ++wait_count;
      if (count <= 8 || (count % 256) == 0) {
        XELOGI(
            "Metal shared memory: waiting for the GPU before overwriting "
            "{:08X}-{:08X} (#{})",
            start, end, count);
      }
      command_processor_.WaitForGpuBeforeSharedMemoryOverwrite();
      ClearGpuUsedRanges();
    }
    MakeRangeValid(start, length, false);
    if (!use_zero_copy_) {
      memcpy(buffer_data + start, xbox_data + start, length);
    }
  };

  for (uint32_t i = 0; i < num_upload_ranges; ++i) {
    const auto& range = upload_page_ranges[i];
    uint32_t start = range.first * page_size;
    uint32_t end = start + range.second * page_size;
    if (start >= kBufferSize) {
      continue;
    }
    if (end > kBufferSize) {
      end = kBufferSize;
    }
    trace_writer_.WriteMemoryRead(start, end - start);

    if (!have_merged) {
      merged_start = start;
      merged_end = end;
      have_merged = true;
      continue;
    }

    // Merge overlapping/adjacent ranges.
    if (start <= merged_end) {
      if (end > merged_end) {
        merged_end = end;
      }
    } else {
      flush_merged_range(merged_start, merged_end);
      merged_start = start;
      merged_end = end;
    }
  }

  if (have_merged) {
    flush_merged_range(merged_start, merged_end);
  }

  return true;
}

bool MetalSharedMemory::InitializeTraceSubmitDownloads() {
  PrepareForTraceDownload();
  return trace_download_page_count() != 0;
}

void MetalSharedMemory::InitializeTraceCompleteDownloads() {
  if (buffer_) {
    const uint8_t* buffer_data =
        static_cast<const uint8_t*>(buffer_->contents());
    for (const auto& download_range : trace_download_ranges()) {
      trace_writer_.WriteMemoryRead(download_range.first, download_range.second,
                                    buffer_data + download_range.first);
    }
  }
  ReleaseTraceDownloadRanges();
}

void MetalSharedMemory::Shutdown() {
  if (buffer_) {
    buffer_->release();
    buffer_ = nullptr;
  }
  use_zero_copy_ = false;

  ShutdownCommon();  // Base class cleanup
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
