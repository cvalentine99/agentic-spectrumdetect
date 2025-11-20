#ifndef INCLUDE_UHD_B210_RADIO_H
#define INCLUDE_UHD_B210_RADIO_H

//--------------------------------------------------------------------------------------------------------------------------

#include "tye_app_includes.h"

#ifdef USE_UHD_B210
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/utils/thread.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/types/tune_request.hpp>
#endif

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

class uhd_b210_radio
{
public: //==================================================================================================================

    // constructor(s) / destructor
    uhd_b210_radio( void );
   ~uhd_b210_radio( void );

    // public methods
    bool        connect_usb( void );
    bool        connect_net( std::string ipaddr, uint16_t port );
    bool        configure( uint64_t sample_rate_hz, uint64_t center_freq_hz, int32_t atten_db, double ref_level );
    bool        recv_samples( void *p_buffer, uint32_t buffer_len, int64_t *p_ns_since_epoch, bool *p_samples_dropped );
    void        disconnect( void );
    std::string get_device_name( void );
    uint64_t    get_sample_rate_hz( void );
    double      get_sample_rate_mhz( void );
    uint64_t    get_sample_rate_max_hz( void );
    uint64_t    get_bandwidth_hz( void );
    double      get_bandwidth_mhz( void );
    uint64_t    get_center_freq_hz( void );
    double      get_center_freq_mhz( void );
    uint64_t    get_center_freq_min_hz( void );
    uint64_t    get_center_freq_max_hz( void );
    bool        atten_is_auto( void );
    int32_t     get_atten_db( void );
    double      get_ref_level( void );

private: //=================================================================================================================

    // private constants
    const std::string NAME = std::string("UHD_B210_RADIO");

    // Ettus B210 specifications
    static constexpr uint64_t B210_FREQ_MIN_HZ      = 70000000ULL;      // 70 MHz
    static constexpr uint64_t B210_FREQ_MAX_HZ      = 6000000000ULL;    // 6 GHz
    static constexpr uint64_t B210_SAMPLE_RATE_MAX  = 56000000ULL;      // 56 MHz (61.44 MSPS actual)
    static constexpr double   B210_GAIN_MIN         = 0.0;              // 0 dB
    static constexpr double   B210_GAIN_MAX         = 76.0;             // 76 dB

    // private variables
    std::string this_device_name;
    std::string this_device_serial;
    std::string this_device_addr;
    uint64_t    this_sample_rate_hz;
    double      this_sample_rate_mhz;
    uint64_t    this_sample_rate_max_hz;
    uint64_t    this_bandwidth_hz;
    double      this_bandwidth_mhz;
    uint64_t    this_center_freq_hz;
    double      this_center_freq_mhz;
    uint64_t    this_center_freq_min_hz;
    uint64_t    this_center_freq_max_hz;
    int32_t     this_atten_db;           // Stored as attenuation for API compatibility
    double      this_gain_db;            // Actual UHD gain value
    double      this_ref_level;          // Reference level (for compatibility)
    bool        this_connected;
    std::string this_antenna;            // "TX/RX" or "RX2"
    size_t      this_channel;            // RX channel (0 or 1)

#ifdef USE_UHD_B210
    // UHD objects
    uhd::usrp::multi_usrp::sptr this_usrp;
    uhd::rx_streamer::sptr      this_rx_stream;
    uhd::rx_metadata_t          this_rx_metadata;
#endif

    // private helper methods
    double atten_to_gain( int32_t atten_db );
    int32_t gain_to_atten( double gain_db );
};

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_UHD_B210_RADIO_H
