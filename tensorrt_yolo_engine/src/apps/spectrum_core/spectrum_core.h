#pragma once

#include <atomic>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cufft.h>

namespace spectrum::core {

struct Detection {
    std::string detection_id;
    std::uint64_t timestamp_ns{0};
    int bin{};
    float freq_hz{};
    float power_db{};
    float bandwidth_hz{0.0f};
    float confidence{0.0f};
    int class_id{-1};
};

struct HostSpectrumResult {
    std::vector<float> power_db;
    std::vector<Detection> detections;
};

struct FftConfig {
    int fft_size{2048};
    int batch{1};
    int averaging{1};
    bool apply_hann{true};
};

struct SpectrumView {
    float* device_power{nullptr};
    int fft_size{0};
    int batch{0};
    cudaEvent_t ready{nullptr};
};

class PinnedBufferRing {
public:
    PinnedBufferRing(std::size_t slot_count, std::size_t bytes_per_slot);
    ~PinnedBufferRing();

    PinnedBufferRing(const PinnedBufferRing&) = delete;
    PinnedBufferRing& operator=(const PinnedBufferRing&) = delete;

    std::span<std::byte> acquire_write();
    void release_write();
    std::span<const std::byte> acquire_read();
    void release_read();

    std::size_t slot_bytes() const { return bytes_per_slot_; }

private:
    struct Slot {
        std::byte* host{nullptr};
        cudaEvent_t ready{};
    };

    std::vector<Slot> slots_;
    std::size_t bytes_per_slot_{0};
    std::atomic<std::uint64_t> write_index_{0};
    std::atomic<std::uint64_t> read_index_{0};
};

class FftPlanCache {
public:
    FftPlanCache() = default;
    ~FftPlanCache();

    FftPlanCache(const FftPlanCache&) = delete;
    FftPlanCache& operator=(const FftPlanCache&) = delete;

    cufftHandle get_or_create(int fft_size, int batch, cudaStream_t stream);

private:
    struct Key {
        int fft_size{};
        int batch{};
        bool operator==(const Key& other) const {
            return fft_size == other.fft_size && batch == other.batch;
        }
    };

    struct KeyHash {
        std::size_t operator()(const Key& k) const noexcept {
            return (static_cast<std::size_t>(k.fft_size) << 32) ^ static_cast<std::size_t>(k.batch);
        }
    };

    std::unordered_map<Key, cufftHandle, KeyHash> plans_;
};

class GpuSpectrumPipeline {
public:
    GpuSpectrumPipeline(FftConfig cfg, int slot_count);
    ~GpuSpectrumPipeline();

    GpuSpectrumPipeline(const GpuSpectrumPipeline&) = delete;
    GpuSpectrumPipeline& operator=(const GpuSpectrumPipeline&) = delete;

    SpectrumView process_async(const std::complex<float>* host_iq);
    HostSpectrumResult collect(const SpectrumView& view, double sample_rate_hz,
                               float detection_threshold_db = -60.0f, int max_detections = 32);
    void wait_for(const SpectrumView& view) const;
    cudaStream_t stream() const { return stream_; }
    int fft_size() const { return cfg_.fft_size; }
    int batch() const { return cfg_.batch; }

private:
    struct Slot {
        cufftComplex* d_time{nullptr};
        cufftComplex* d_freq{nullptr};
        float* d_power{nullptr};
        cudaEvent_t ready{};
    };

    void allocate_slots(int slot_count);
    void destroy_slots();
    void build_window();
    void ensure_plan();

    FftConfig cfg_{};
    cudaStream_t stream_{};
    FftPlanCache plan_cache_;
    cufftHandle plan_{};
    std::vector<float> window_host_;
    float* window_device_{nullptr};
    std::vector<Slot> slots_;
    int next_slot_{0};
};

}  // namespace spectrum::core
