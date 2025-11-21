#include "spectrum_core.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <zmq.h>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

using spectrum::core::Detection;
using spectrum::core::FftConfig;
using spectrum::core::GpuSpectrumPipeline;

namespace {

std::atomic<bool> g_running{true};

void handle_sig(int) {
    g_running.store(false);
}

struct PipelineHolder {
    FftConfig cfg{};
    int slots{4};
    std::unique_ptr<GpuSpectrumPipeline> pipeline;
    std::vector<std::complex<float>> host_iq;
    void* iq_sock{nullptr};

    void ensure(const FftConfig& new_cfg) {
        if (!pipeline || new_cfg.fft_size != cfg.fft_size || new_cfg.batch != cfg.batch ||
            new_cfg.averaging != cfg.averaging || new_cfg.apply_hann != cfg.apply_hann) {
            cfg = new_cfg;
            pipeline = std::make_unique<GpuSpectrumPipeline>(cfg, slots);
            host_iq.resize(static_cast<std::size_t>(cfg.fft_size) * cfg.batch);
        }
    }
};

void fill_synthetic_iq(std::vector<std::complex<float>>& iq, double sample_rate_hz, double tone_hz) {
    const double w = 2.0 * M_PI * (tone_hz / sample_rate_hz);
    for (std::size_t n = 0; n < iq.size(); ++n) {
        const float re = std::cos(w * static_cast<double>(n));
        const float im = std::sin(w * static_cast<double>(n));
        iq[n] = {re, im};
    }
}

bool fill_iq_from_socket(void* sock, std::vector<std::complex<float>>& iq) {
    if (!sock) return false;

    const std::size_t wanted_bytes = iq.size() * sizeof(std::complex<float>);
    zmq_msg_t msg;
    zmq_msg_init(&msg);
    const int rc = zmq_msg_recv(&msg, sock, ZMQ_DONTWAIT);
    if (rc < 0) {
        zmq_msg_close(&msg);
        return false;
    }
    const std::size_t got = static_cast<std::size_t>(rc);
    if (got < wanted_bytes) {
        zmq_msg_close(&msg);
        return false;
    }
    std::memcpy(iq.data(), zmq_msg_data(&msg), wanted_bytes);
    zmq_msg_close(&msg);
    return true;
}

bool fill_detections_from_socket(void* sock, std::vector<spectrum::core::Detection>& detections) {
    if (!sock) return false;
    zmq_msg_t msg;
    zmq_msg_init(&msg);
    const int rc = zmq_msg_recv(&msg, sock, ZMQ_DONTWAIT);
    if (rc < 0) {
        zmq_msg_close(&msg);
        return false;
    }
    const std::string json(static_cast<char*>(zmq_msg_data(&msg)), zmq_msg_size(&msg));
    zmq_msg_close(&msg);

    rapidjson::Document doc;
    doc.Parse(json.c_str());
    if (!doc.IsArray()) {
        return false;
    }

    detections.clear();
    for (auto& d : doc.GetArray()) {
        if (!d.IsObject()) continue;
        spectrum::core::Detection det{};
        if (d.HasMember("detection_id") && d["detection_id"].IsString())
            det.detection_id = d["detection_id"].GetString();
        if (d.HasMember("timestamp_ns") && d["timestamp_ns"].IsUint64())
            det.timestamp_ns = d["timestamp_ns"].GetUint64();
        if (d.HasMember("bin") && d["bin"].IsInt())
            det.bin = d["bin"].GetInt();
        if (d.HasMember("freq_hz") && d["freq_hz"].IsNumber())
            det.freq_hz = static_cast<float>(d["freq_hz"].GetDouble());
        if (d.HasMember("power_dbm") && d["power_dbm"].IsNumber())
            det.power_db = static_cast<float>(d["power_dbm"].GetDouble());
        if (d.HasMember("power_db") && d["power_db"].IsNumber())
            det.power_db = static_cast<float>(d["power_db"].GetDouble());
        if (d.HasMember("bandwidth_hz") && d["bandwidth_hz"].IsNumber())
            det.bandwidth_hz = static_cast<float>(d["bandwidth_hz"].GetDouble());
        if (d.HasMember("confidence") && d["confidence"].IsNumber())
            det.confidence = static_cast<float>(d["confidence"].GetDouble());
        if (d.HasMember("class_id") && d["class_id"].IsInt())
            det.class_id = d["class_id"].GetInt();
        detections.push_back(det);
    }
    return !detections.empty();
}

bool parse_request(const std::string& json, int& fft_size, int& averaging, std::int64_t& center_hz,
                   std::int64_t& sample_rate_hz) {
    rapidjson::Document doc;
    doc.Parse(json.c_str());
    if (doc.HasParseError() || !doc.IsObject()) {
        return false;
    }
    fft_size = doc.HasMember("fft_size") && doc["fft_size"].IsInt() ? doc["fft_size"].GetInt() : 2048;
    averaging =
        doc.HasMember("averaging") && doc["averaging"].IsInt() ? std::max(1, doc["averaging"].GetInt()) : 10;
    center_hz =
        doc.HasMember("center_freq_hz") && doc["center_freq_hz"].IsInt64() ? doc["center_freq_hz"].GetInt64() : 0;
    sample_rate_hz = doc.HasMember("sample_rate_hz") && doc["sample_rate_hz"].IsInt64()
                         ? doc["sample_rate_hz"].GetInt64()
                         : 50'000'000;
    return true;
}

std::string build_response_header(std::int64_t center_hz, std::int64_t sample_rate_hz, int fft_size, int averaging,
                                  std::uint64_t timestamp_ns, float bin_hz,
                                  const std::vector<spectrum::core::Detection>& detections) {
    rapidjson::Document doc;
    doc.SetObject();
    auto& alloc = doc.GetAllocator();
    doc.AddMember("op", "measure", alloc);
    doc.AddMember("center_freq_hz", center_hz, alloc);
    doc.AddMember("sample_rate_hz", sample_rate_hz, alloc);
    doc.AddMember("fft_size", fft_size, alloc);
    doc.AddMember("averaging", averaging, alloc);
    doc.AddMember("timestamp_ns", rapidjson::Value().SetUint64(timestamp_ns), alloc);

    rapidjson::Value det_arr(rapidjson::kArrayType);
    for (const auto& det : detections) {
        rapidjson::Value obj(rapidjson::kObjectType);
        obj.AddMember("detection_id", rapidjson::Value().SetString(det.detection_id.c_str(), alloc), alloc);
        obj.AddMember("timestamp_ns", rapidjson::Value().SetUint64(det.timestamp_ns ? det.timestamp_ns : timestamp_ns),
                      alloc);
        obj.AddMember("bin", det.bin, alloc);
        obj.AddMember("freq_hz", det.freq_hz, alloc);  // absolute frequency
        obj.AddMember("power_dbm", det.power_db, alloc);
        obj.AddMember("bandwidth_hz",
                      rapidjson::Value().SetInt64(static_cast<std::int64_t>(det.bandwidth_hz > 0 ? det.bandwidth_hz
                                                                                                   : bin_hz * 3.0f)),
                      alloc);
        obj.AddMember("confidence", det.confidence > 0 ? det.confidence : std::min(0.99, 0.5 + (det.power_db / 120.0)),
                      alloc);
        obj.AddMember("class_id", det.class_id, alloc);
        det_arr.PushBack(obj, alloc);
    }
    doc.AddMember("detections", det_arr, alloc);

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    return std::string(buffer.GetString(), buffer.GetSize());
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sig);
    std::signal(SIGTERM, handle_sig);

    const char* endpoint_env = std::getenv("SPECTRUM_CORE_BIND");
    const std::string endpoint = endpoint_env ? endpoint_env : "tcp://*:5555";
    const char* iq_endpoint_env = std::getenv("SPECTRUM_CORE_IQ_ENDPOINT");
    const char* det_endpoint_env = std::getenv("SPECTRUM_CORE_DET_ENDPOINT");

    void* ctx = zmq_ctx_new();
    if (!ctx) {
        std::cerr << "zmq_ctx_new failed\n";
        return 1;
    }

    void* sock = zmq_socket(ctx, ZMQ_REP);
    if (!sock) {
        std::cerr << "zmq_socket failed\n";
        zmq_ctx_term(ctx);
        return 1;
    }

    if (zmq_bind(sock, endpoint.c_str()) != 0) {
        std::cerr << "zmq_bind failed on " << endpoint << ": " << zmq_strerror(errno) << "\n";
        zmq_close(sock);
        zmq_ctx_term(ctx);
        return 1;
    }

    void* iq_sock = nullptr;
    void* det_sock = nullptr;
    if (iq_endpoint_env) {
        iq_sock = zmq_socket(ctx, ZMQ_PULL);
        if (!iq_sock || zmq_connect(iq_sock, iq_endpoint_env) != 0) {
            std::cerr << "warning: failed to connect IQ PULL socket to " << iq_endpoint_env << "\n";
            if (iq_sock) {
                zmq_close(iq_sock);
                iq_sock = nullptr;
            }
        } else {
            std::cout << "connected IQ PULL at " << iq_endpoint_env << std::endl;
        }
    }
    if (det_endpoint_env) {
        det_sock = zmq_socket(ctx, ZMQ_PULL);
        if (!det_sock || zmq_connect(det_sock, det_endpoint_env) != 0) {
            std::cerr << "warning: failed to connect DET PULL socket to " << det_endpoint_env << "\n";
            if (det_sock) {
                zmq_close(det_sock);
                det_sock = nullptr;
            }
        } else {
            std::cout << "connected DET PULL at " << det_endpoint_env << std::endl;
        }
    }

    PipelineHolder holder;
    holder.iq_sock = iq_sock;
    std::cout << "spectrum_core_server listening at " << endpoint << std::endl;
    std::vector<spectrum::core::Detection> det_buffer;

    while (g_running.load()) {
        zmq_msg_t msg;
        zmq_msg_init(&msg);
        int rc = zmq_msg_recv(&msg, sock, 0);
        if (rc < 0) {
            zmq_msg_close(&msg);
            continue;
        }

        const std::string json_req(static_cast<char*>(zmq_msg_data(&msg)), zmq_msg_size(&msg));
        zmq_msg_close(&msg);

        int fft_size = 2048;
        int averaging = 10;
        std::int64_t center_hz = 0;
        std::int64_t sample_rate_hz = 50'000'000;
        if (!parse_request(json_req, fft_size, averaging, center_hz, sample_rate_hz)) {
            const std::string err = R"({"error":"invalid_request"})";
            zmq_msg_t m;
            zmq_msg_init_size(&m, err.size());
            std::memcpy(zmq_msg_data(&m), err.data(), err.size());
            zmq_msg_send(&m, sock, 0);
            zmq_msg_close(&m);
            continue;
        }

        FftConfig cfg;
        cfg.fft_size = fft_size;
        cfg.batch = 1;
        cfg.averaging = averaging;
        cfg.apply_hann = true;
        holder.ensure(cfg);

        bool have_iq = fill_iq_from_socket(holder.iq_sock, holder.host_iq);
        if (!have_iq) {
            fill_synthetic_iq(holder.host_iq, static_cast<double>(sample_rate_hz), 3.2e6);
        }
        // Get latest detections (non-blocking)
        fill_detections_from_socket(det_sock, det_buffer);
        auto view = holder.pipeline->process_async(holder.host_iq.data());
        auto result = holder.pipeline->collect(view, static_cast<double>(sample_rate_hz), -60.0f, 32);

        // Convert detections to absolute frequency
        for (auto& det : result.detections) {
            det.freq_hz += static_cast<float>(center_hz);
        }

        const float bin_hz = static_cast<float>(sample_rate_hz) / static_cast<float>(fft_size);
        const std::uint64_t now_ns =
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now().time_since_epoch())
                                           .count());

        // Merge external detections if present
        std::vector<spectrum::core::Detection> dets_out = result.detections;
        if (!det_buffer.empty()) {
            dets_out = det_buffer;
            det_buffer.clear();
        }

        const std::string header =
            build_response_header(center_hz, sample_rate_hz, fft_size, averaging, now_ns, bin_hz, dets_out);

        zmq_msg_t header_msg;
        zmq_msg_init_size(&header_msg, header.size());
        std::memcpy(zmq_msg_data(&header_msg), header.data(), header.size());

        zmq_msg_t payload_msg;
        zmq_msg_init_size(&payload_msg, result.power_db.size() * sizeof(float));
        std::memcpy(zmq_msg_data(&payload_msg), result.power_db.data(), result.power_db.size() * sizeof(float));

        zmq_msg_send(&header_msg, sock, ZMQ_SNDMORE);
        zmq_msg_send(&payload_msg, sock, 0);
        zmq_msg_close(&header_msg);
        zmq_msg_close(&payload_msg);
    }

    zmq_close(sock);
    if (iq_sock) zmq_close(iq_sock);
    if (det_sock) zmq_close(det_sock);
    zmq_ctx_term(ctx);
    return 0;
}
