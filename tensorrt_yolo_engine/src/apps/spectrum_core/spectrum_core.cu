#include "spectrum_core.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

namespace spectrum::core {

namespace {

inline void cuda_check(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        std::ostringstream oss;
        oss << msg << ": " << cudaGetErrorString(status);
        throw std::runtime_error(oss.str());
    }
}

inline void cufft_check(cufftResult status, const char* msg) {
    if (status != CUFFT_SUCCESS) {
        std::ostringstream oss;
        oss << msg << ": cuFFT error " << status;
        throw std::runtime_error(oss.str());
    }
}

__global__ void apply_hann_kernel(cufftComplex* data, const float* window, int fft_size, int batch) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = fft_size * batch;
    if (idx >= total) return;
    const int bin = idx % fft_size;
    data[idx].x *= window[bin];
    data[idx].y *= window[bin];
}

__global__ void power_kernel(const cufftComplex* freq, float* power, int fft_size, int batch, int averaging) {
    const int bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin >= fft_size) return;
    float acc = 0.0f;
    for (int b = 0; b < batch; ++b) {
        const int idx = b * fft_size + bin;
        const float re = freq[idx].x;
        const float im = freq[idx].y;
        acc += re * re + im * im;
    }
    const float norm = fmaxf(acc / static_cast<float>(averaging), 1e-20f);
    power[bin] = 10.0f * log10f(norm);
}

}  // namespace

PinnedBufferRing::PinnedBufferRing(std::size_t slot_count, std::size_t bytes_per_slot)
    : slots_(slot_count), bytes_per_slot_(bytes_per_slot) {
    for (auto& slot : slots_) {
        cuda_check(cudaMallocHost(reinterpret_cast<void**>(&slot.host), bytes_per_slot_), "cudaMallocHost");
        cuda_check(cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming), "cudaEventCreate");
    }
}

PinnedBufferRing::~PinnedBufferRing() {
    for (auto& slot : slots_) {
        cudaEventDestroy(slot.ready);
        cudaFreeHost(slot.host);
    }
}

std::span<std::byte> PinnedBufferRing::acquire_write() {
    const auto slot_count = static_cast<std::uint64_t>(slots_.size());
    while (write_index_.load(std::memory_order_acquire) - read_index_.load(std::memory_order_acquire) >= slot_count) {
        std::this_thread::yield();
    }
    const auto idx = write_index_.load(std::memory_order_relaxed) % slot_count;
    return {slots_[idx].host, bytes_per_slot_};
}

void PinnedBufferRing::release_write() {
    write_index_.fetch_add(1, std::memory_order_release);
}

std::span<const std::byte> PinnedBufferRing::acquire_read() {
    const auto slot_count = static_cast<std::uint64_t>(slots_.size());
    while (read_index_.load(std::memory_order_acquire) >= write_index_.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    const auto idx = read_index_.load(std::memory_order_relaxed) % slot_count;
    return {slots_[idx].host, bytes_per_slot_};
}

void PinnedBufferRing::release_read() {
    read_index_.fetch_add(1, std::memory_order_release);
}

FftPlanCache::~FftPlanCache() {
    for (auto& [_, handle] : plans_) {
        cufftDestroy(handle);
    }
}

cufftHandle FftPlanCache::get_or_create(int fft_size, int batch, cudaStream_t stream) {
    Key key{fft_size, batch};
    const auto it = plans_.find(key);
    if (it != plans_.end()) {
        cufft_check(cufftSetStream(it->second, stream), "cufftSetStream");
        return it->second;
    }

    cufftHandle plan{};
    const int rank = 1;
    const int n[1] = {fft_size};
    const int istride = 1;
    const int ostride = 1;
    const int idist = fft_size;
    const int odist = fft_size;
    cufft_check(
        cufftPlanMany(&plan, rank, n, n, istride, idist, n, ostride, odist, CUFFT_C2C, batch),
        "cufftPlanMany");
    cufft_check(cufftSetStream(plan, stream), "cufftSetStream");
    plans_.emplace(key, plan);
    return plan;
}

GpuSpectrumPipeline::GpuSpectrumPipeline(FftConfig cfg, int slot_count) : cfg_(cfg) {
    cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    allocate_slots(slot_count);
    build_window();
    ensure_plan();
}

GpuSpectrumPipeline::~GpuSpectrumPipeline() {
    destroy_slots();
    if (window_device_) {
        cudaFree(window_device_);
    }
    cudaStreamDestroy(stream_);
}

void GpuSpectrumPipeline::allocate_slots(int slot_count) {
    slots_.resize(slot_count);
    const std::size_t complex_bytes = static_cast<std::size_t>(cfg_.fft_size) * cfg_.batch * sizeof(cufftComplex);
    const std::size_t power_bytes = static_cast<std::size_t>(cfg_.fft_size) * sizeof(float);
    for (auto& slot : slots_) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&slot.d_time), complex_bytes), "cudaMalloc d_time");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&slot.d_freq), complex_bytes), "cudaMalloc d_freq");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&slot.d_power), power_bytes), "cudaMalloc d_power");
        cuda_check(cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming), "cudaEventCreate");
    }
}

void GpuSpectrumPipeline::destroy_slots() {
    for (auto& slot : slots_) {
        if (slot.ready) cudaEventDestroy(slot.ready);
        if (slot.d_time) cudaFree(slot.d_time);
        if (slot.d_freq) cudaFree(slot.d_freq);
        if (slot.d_power) cudaFree(slot.d_power);
    }
}

void GpuSpectrumPipeline::build_window() {
    if (!cfg_.apply_hann) {
        return;
    }
    window_host_.resize(static_cast<std::size_t>(cfg_.fft_size));
    const float two_pi = 6.28318530717958647692f;
    for (int i = 0; i < cfg_.fft_size; ++i) {
        const float phase = two_pi * static_cast<float>(i) / static_cast<float>(cfg_.fft_size - 1);
        window_host_[i] = 0.5f * (1.0f - std::cos(phase));
    }
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&window_device_), window_host_.size() * sizeof(float)),
               "cudaMalloc window");
    cuda_check(cudaMemcpyAsync(window_device_, window_host_.data(),
                               window_host_.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream_),
               "cudaMemcpyAsync window");
}

void GpuSpectrumPipeline::ensure_plan() {
    plan_ = plan_cache_.get_or_create(cfg_.fft_size, cfg_.batch, stream_);
}

SpectrumView GpuSpectrumPipeline::process_async(const std::complex<float>* host_iq) {
    const std::size_t complex_samples = static_cast<std::size_t>(cfg_.fft_size) * cfg_.batch;
    const std::size_t complex_bytes = complex_samples * sizeof(cufftComplex);

    Slot& slot = slots_[next_slot_];
    next_slot_ = (next_slot_ + 1) % static_cast<int>(slots_.size());

    cuda_check(cudaMemcpyAsync(slot.d_time, host_iq, complex_bytes,
                               cudaMemcpyHostToDevice, stream_), "cudaMemcpyAsync H2D");

    if (cfg_.apply_hann && window_device_) {
        const int threads = 256;
        const int blocks = static_cast<int>((complex_samples + threads - 1) / threads);
        apply_hann_kernel<<<blocks, threads, 0, stream_>>>(slot.d_time, window_device_, cfg_.fft_size, cfg_.batch);
    }

    cufft_check(cufftExecC2C(plan_, slot.d_time, slot.d_freq, CUFFT_FORWARD), "cufftExecC2C");

    const int threads = 256;
    const int blocks = (cfg_.fft_size + threads - 1) / threads;
    power_kernel<<<blocks, threads, 0, stream_>>>(slot.d_freq, slot.d_power, cfg_.fft_size, cfg_.batch,
                                                  std::max(cfg_.averaging, 1));

    cuda_check(cudaEventRecord(slot.ready, stream_), "cudaEventRecord");

    return SpectrumView{
        .device_power = slot.d_power,
        .fft_size = cfg_.fft_size,
        .batch = cfg_.batch,
        .ready = slot.ready,
    };
}

void GpuSpectrumPipeline::wait_for(const SpectrumView& view) const {
    if (view.ready) {
        cudaEventSynchronize(view.ready);
    }
}

HostSpectrumResult GpuSpectrumPipeline::collect(const SpectrumView& view, double sample_rate_hz,
                                                float detection_threshold_db, int max_detections) {
    if (max_detections < 0) {
        max_detections = 0;
    }
    wait_for(view);

    HostSpectrumResult result;
    result.power_db.resize(static_cast<std::size_t>(view.fft_size));

    const std::size_t bytes = static_cast<std::size_t>(view.fft_size) * sizeof(float);
    cuda_check(cudaMemcpyAsync(result.power_db.data(), view.device_power, bytes, cudaMemcpyDeviceToHost, stream_),
               "cudaMemcpyAsync D2H power");
    cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize copy");

    if (max_detections == 0) {
        return result;
    }

    const float bin_hz = static_cast<float>(sample_rate_hz) / static_cast<float>(view.fft_size);
    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(view.fft_size);

    // Simple threshold-based peak picker (magnitude only).
    for (int bin = 1; bin + 1 < view.fft_size; ++bin) {
        const float p = result.power_db[bin];
        if (p < detection_threshold_db) continue;
        if (p < result.power_db[bin - 1] || p < result.power_db[bin + 1]) continue;  // non-peak
        candidates.emplace_back(p, bin);
    }

    const int take = std::min<int>(max_detections, static_cast<int>(candidates.size()));
    if (take > 0) {
        std::partial_sort(candidates.begin(), candidates.begin() + take, candidates.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        for (int i = 0; i < take; ++i) {
            const auto [power, bin] = candidates[i];
            result.detections.push_back(Detection{
                .detection_id = {},
                .timestamp_ns = 0,
                .bin = bin,
                .freq_hz = bin_hz * (static_cast<float>(bin) - view.fft_size * 0.5f),
                .power_db = power,
                .bandwidth_hz = bin_hz * 3.0f,
                .confidence = std::min(0.99f, 0.5f + (power / 120.0f)),
                .class_id = -1,
            });
        }
    }

    return result;
}

}  // namespace spectrum::core
