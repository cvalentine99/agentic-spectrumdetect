//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "uhd_b210_radio.h"

//-- constructor(s) --------------------------------------------------------------------------------------------------------

uhd_b210_radio::uhd_b210_radio( void )
{
    // initialize
    this_device_name        = std::string("Ettus B210");
    this_device_serial      = std::string("");
    this_device_addr        = std::string("");
    this_sample_rate_hz     = 0;
    this_sample_rate_mhz    = 0.0;
    this_sample_rate_max_hz = B210_SAMPLE_RATE_MAX;
    this_bandwidth_hz       = 0;
    this_bandwidth_mhz      = 0.0;
    this_center_freq_hz     = 0;
    this_center_freq_min_hz = B210_FREQ_MIN_HZ;
    this_center_freq_max_hz = B210_FREQ_MAX_HZ;
    this_center_freq_mhz    = 0.0;
    this_atten_db           = 0;
    this_gain_db            = B210_GAIN_MAX;  // Start with max gain (min attenuation)
    this_ref_level          = 0.0;
    this_connected          = false;
    this_antenna            = "TX/RX";         // Default antenna
    this_channel            = 0;               // Default to channel 0

#ifdef USE_UHD_B210
    this_usrp       = nullptr;
    this_rx_stream  = nullptr;
#endif

    return;
}

//-- destructor ------------------------------------------------------------------------------------------------------------

uhd_b210_radio::~uhd_b210_radio( void )
{
    // clean up
    this->disconnect();

    return;
}

//-- public methods --------------------------------------------------------------------------------------------------------

// connect to USB radio
bool uhd_b210_radio::connect_usb( void )
{
#ifdef USE_UHD_B210
    try
    {
        // Create a USRP device - auto-discover B210
        std::string device_args = "type=b200";  // B200/B210 device type
        this_usrp = uhd::usrp::multi_usrp::make(device_args);

        if ( !this_usrp )
        {
            std::cerr << NAME << " :: ERROR :: Failed to create USRP device" << std::endl;
            return false;
        }

        // Get device information
        uhd::dict<std::string, std::string> usrp_info = this_usrp->get_usrp_rx_info(this_channel);
        this_device_name   = std::string("Ettus ") + usrp_info.get("mboard_id", "B210");
        this_device_serial = usrp_info.get("mboard_serial", "Unknown");

        std::cout << NAME << " :: Connected to " << this_device_name
                  << " (Serial: " << this_device_serial << ")" << std::endl;

        // Set subdevice spec (use A:A for RX0)
        this_usrp->set_rx_subdev_spec(uhd::usrp::subdev_spec_t("A:A"), 0);

        // Set default antenna
        this_usrp->set_rx_antenna(this_antenna, this_channel);

        // Set thread priority for better real-time performance
        uhd::set_thread_priority_safe();

        this_connected = true;
        return true;
    }
    catch ( uhd::exception &e )
    {
        std::cerr << NAME << " :: ERROR :: " << e.what() << std::endl;
        return false;
    }
#else
    std::cerr << NAME << " :: ERROR :: UHD support not compiled in" << std::endl;
    return false;
#endif
}

//--------------------------------------------------------------------------------------------------------------------------

// connect to network radio
bool uhd_b210_radio::connect_net( std::string ipaddr, uint16_t port )
{
#ifdef USE_UHD_B210
    try
    {
        // Create a USRP device with IP address
        std::string device_args = "type=b200,addr=" + ipaddr;
        this_usrp = uhd::usrp::multi_usrp::make(device_args);

        if ( !this_usrp )
        {
            std::cerr << NAME << " :: ERROR :: Failed to create USRP device" << std::endl;
            return false;
        }

        // Get device information
        uhd::dict<std::string, std::string> usrp_info = this_usrp->get_usrp_rx_info(this_channel);
        this_device_name   = std::string("Ettus ") + usrp_info.get("mboard_id", "B210");
        this_device_serial = usrp_info.get("mboard_serial", "Unknown");
        this_device_addr   = ipaddr;

        std::cout << NAME << " :: Connected to " << this_device_name
                  << " @ " << ipaddr << " (Serial: " << this_device_serial << ")" << std::endl;

        // Set subdevice spec
        this_usrp->set_rx_subdev_spec(uhd::usrp::subdev_spec_t("A:A"), 0);

        // Set default antenna
        this_usrp->set_rx_antenna(this_antenna, this_channel);

        // Set thread priority
        uhd::set_thread_priority_safe();

        this_connected = true;
        return true;
    }
    catch ( uhd::exception &e )
    {
        std::cerr << NAME << " :: ERROR :: " << e.what() << std::endl;
        return false;
    }
#else
    std::cerr << NAME << " :: ERROR :: UHD support not compiled in" << std::endl;
    return false;
#endif
}

//--------------------------------------------------------------------------------------------------------------------------

// configure radio
bool uhd_b210_radio::configure( uint64_t sample_rate_hz, uint64_t center_freq_hz, int32_t atten_db, double ref_level )
{
#ifdef USE_UHD_B210
    if ( !this_connected || !this_usrp )
    {
        std::cerr << NAME << " :: ERROR :: Device not connected" << std::endl;
        return false;
    }

    try
    {
        // Validate sample rate
        if ( sample_rate_hz > B210_SAMPLE_RATE_MAX )
        {
            std::cerr << NAME << " :: ERROR :: Sample rate " << sample_rate_hz
                      << " Hz exceeds maximum " << B210_SAMPLE_RATE_MAX << " Hz" << std::endl;
            return false;
        }

        // Validate frequency range
        if ( center_freq_hz < B210_FREQ_MIN_HZ || center_freq_hz > B210_FREQ_MAX_HZ )
        {
            std::cerr << NAME << " :: ERROR :: Frequency " << center_freq_hz
                      << " Hz out of range [" << B210_FREQ_MIN_HZ << ", " << B210_FREQ_MAX_HZ << "]" << std::endl;
            return false;
        }

        // Convert attenuation to gain
        // B210 uses gain (0-76 dB), Signal Hound uses attenuation (0-30 dB)
        // Mapping: high gain = low attenuation
        this_gain_db = atten_to_gain(atten_db);

        std::cout << NAME << " :: Configuring:" << std::endl;
        std::cout << "  Sample Rate: " << (sample_rate_hz / 1e6) << " MHz" << std::endl;
        std::cout << "  Center Freq: " << (center_freq_hz / 1e6) << " MHz" << std::endl;
        std::cout << "  Attenuation: " << atten_db << " dB (Gain: " << this_gain_db << " dB)" << std::endl;

        // Set sample rate
        this_usrp->set_rx_rate(sample_rate_hz, this_channel);
        double actual_rate = this_usrp->get_rx_rate(this_channel);

        if ( std::abs(actual_rate - sample_rate_hz) > 1.0 )
        {
            std::cout << NAME << " :: WARNING :: Actual sample rate " << actual_rate
                      << " differs from requested " << sample_rate_hz << std::endl;
        }

        // Set center frequency
        uhd::tune_request_t tune_request(center_freq_hz);
        uhd::tune_result_t tune_result = this_usrp->set_rx_freq(tune_request, this_channel);
        double actual_freq = this_usrp->get_rx_freq(this_channel);

        // Set gain
        this_usrp->set_rx_gain(this_gain_db, this_channel);
        double actual_gain = this_usrp->get_rx_gain(this_channel);

        // Set bandwidth (analog filter)
        this_usrp->set_rx_bandwidth(sample_rate_hz, this_channel);
        double actual_bw = this_usrp->get_rx_bandwidth(this_channel);

        // Wait for LO to lock
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Create RX streamer
        uhd::stream_args_t stream_args("fc32", "sc16");  // CPU format: complex float, OTW format: complex int16
        stream_args.channels = {this_channel};
        this_rx_stream = this_usrp->get_rx_stream(stream_args);

        // Start streaming
        uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
        stream_cmd.stream_now = true;
        this_rx_stream->issue_stream_cmd(stream_cmd);

        // Save settings
        this_sample_rate_hz  = (uint64_t)actual_rate;
        this_sample_rate_mhz = actual_rate / 1e6;
        this_bandwidth_hz    = (uint64_t)actual_bw;
        this_bandwidth_mhz   = actual_bw / 1e6;
        this_center_freq_hz  = (uint64_t)actual_freq;
        this_center_freq_mhz = actual_freq / 1e6;
        this_atten_db        = atten_db;
        this_ref_level       = ref_level;

        std::cout << NAME << " :: Configuration complete" << std::endl;
        std::cout << "  Actual Rate: " << this_sample_rate_mhz << " MHz" << std::endl;
        std::cout << "  Actual Freq: " << this_center_freq_mhz << " MHz" << std::endl;
        std::cout << "  Actual Gain: " << actual_gain << " dB" << std::endl;
        std::cout << "  Actual BW:   " << this_bandwidth_mhz << " MHz" << std::endl;

        return true;
    }
    catch ( uhd::exception &e )
    {
        std::cerr << NAME << " :: ERROR :: " << e.what() << std::endl;
        return false;
    }
#else
    std::cerr << NAME << " :: ERROR :: UHD support not compiled in" << std::endl;
    return false;
#endif
}

//--------------------------------------------------------------------------------------------------------------------------

// receive radio samples
bool uhd_b210_radio::recv_samples( void *p_buffer, uint32_t buffer_len, int64_t *p_ns_since_epoch, bool *p_samples_dropped )
{
#ifdef USE_UHD_B210
    if ( !this_connected || !this_usrp || !this_rx_stream )
    {
        std::cerr << NAME << " :: ERROR :: Device not connected or streaming not started" << std::endl;
        return false;
    }

    try
    {
        // Calculate number of samples
        // Buffer is complex<float> (8 bytes per sample)
        size_t num_samples = buffer_len / sizeof(std::complex<float>);

        // Receive samples
        // UHD will automatically convert sc16 (on-the-wire) to fc32 (in buffer)
        size_t num_rx_samps = this_rx_stream->recv(p_buffer, num_samples, this_rx_metadata, 3.0);

        if ( this_rx_metadata.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE )
        {
            if ( this_rx_metadata.error_code == uhd::rx_metadata_t::ERROR_CODE_OVERFLOW )
            {
                std::cerr << NAME << " :: WARNING :: Overflow detected" << std::endl;
                *p_samples_dropped = true;
            }
            else if ( this_rx_metadata.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT )
            {
                std::cerr << NAME << " :: ERROR :: Timeout waiting for samples" << std::endl;
                return false;
            }
            else
            {
                std::cerr << NAME << " :: ERROR :: Receive error: " << this_rx_metadata.strerror() << std::endl;
                return false;
            }
        }
        else
        {
            *p_samples_dropped = false;
        }

        // Convert timestamp to nanoseconds since epoch
        // UHD timestamp is in seconds since device start
        // Get full timestamp
        uint64_t full_secs = this_rx_metadata.time_spec.get_full_secs();
        double   frac_secs = this_rx_metadata.time_spec.get_frac_secs();

        // Convert to nanoseconds (approximate - would need system time reference for absolute time)
        *p_ns_since_epoch = (int64_t)(full_secs * 1000000000ULL + frac_secs * 1000000000.0);

        // Check if we got all requested samples
        if ( num_rx_samps != num_samples )
        {
            std::cerr << NAME << " :: WARNING :: Received " << num_rx_samps
                      << " samples, expected " << num_samples << std::endl;
        }

        return true;
    }
    catch ( uhd::exception &e )
    {
        std::cerr << NAME << " :: ERROR :: " << e.what() << std::endl;
        return false;
    }
#else
    std::cerr << NAME << " :: ERROR :: UHD support not compiled in" << std::endl;
    return false;
#endif
}

//--------------------------------------------------------------------------------------------------------------------------

// disconnect from radio
void uhd_b210_radio::disconnect( void )
{
#ifdef USE_UHD_B210
    // clean up
    if ( this_connected )
    {
        try
        {
            if ( this_rx_stream )
            {
                // Stop streaming
                uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
                this_rx_stream->issue_stream_cmd(stream_cmd);

                this_rx_stream.reset();
            }

            if ( this_usrp )
            {
                this_usrp.reset();
            }

            std::cout << NAME << " :: Disconnected" << std::endl;
        }
        catch ( uhd::exception &e )
        {
            std::cerr << NAME << " :: ERROR during disconnect :: " << e.what() << std::endl;
        }

        this_device_name   = std::string("");
        this_device_serial = std::string("");
        this_device_addr   = std::string("");
        this_connected     = false;
    }
#endif

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// getters
std::string uhd_b210_radio::get_device_name( void )        { return ( this_device_name                      ); }
uint64_t    uhd_b210_radio::get_sample_rate_hz( void )     { return ( this_sample_rate_hz                   ); }
double      uhd_b210_radio::get_sample_rate_mhz( void )    { return ( this_sample_rate_mhz                  ); }
uint64_t    uhd_b210_radio::get_sample_rate_max_hz( void ) { return ( this_sample_rate_max_hz               ); }
uint64_t    uhd_b210_radio::get_bandwidth_hz( void )       { return ( this_bandwidth_hz                     ); }
double      uhd_b210_radio::get_bandwidth_mhz( void )      { return ( this_bandwidth_mhz                    ); }
uint64_t    uhd_b210_radio::get_center_freq_hz( void )     { return ( this_center_freq_hz                   ); }
double      uhd_b210_radio::get_center_freq_mhz( void )    { return ( this_center_freq_mhz                  ); }
uint64_t    uhd_b210_radio::get_center_freq_min_hz( void ) { return ( this_center_freq_min_hz               ); }
uint64_t    uhd_b210_radio::get_center_freq_max_hz( void ) { return ( this_center_freq_max_hz               ); }
bool        uhd_b210_radio::atten_is_auto( void )          { return ( false                                 ); } // B210 doesn't have auto atten
int32_t     uhd_b210_radio::get_atten_db( void )           { return ( this_atten_db                         ); }
double      uhd_b210_radio::get_ref_level( void )          { return ( this_ref_level                        ); }

//--------------------------------------------------------------------------------------------------------------------------

// private helper methods

// Convert attenuation (Signal Hound style) to gain (UHD style)
// Attenuation: 0 dB = maximum signal, 30 dB = minimum signal
// Gain: 0 dB = minimum signal, 76 dB = maximum signal
// Mapping: atten 0 dB -> gain 76 dB, atten 30 dB -> gain 46 dB
double uhd_b210_radio::atten_to_gain( int32_t atten_db )
{
    if ( atten_db == -1 )  // Auto attenuation
    {
        return B210_GAIN_MAX / 2.0;  // Use mid-range gain (38 dB)
    }

    // Clamp attenuation to valid range
    if ( atten_db < 0 ) atten_db = 0;
    if ( atten_db > 30 ) atten_db = 30;

    // Convert: max gain - attenuation
    double gain = B210_GAIN_MAX - (double)atten_db * (B210_GAIN_MAX / 30.0);

    // Clamp to valid gain range
    if ( gain < B210_GAIN_MIN ) gain = B210_GAIN_MIN;
    if ( gain > B210_GAIN_MAX ) gain = B210_GAIN_MAX;

    return gain;
}

// Convert gain (UHD style) to attenuation (Signal Hound style)
int32_t uhd_b210_radio::gain_to_atten( double gain_db )
{
    // Clamp gain to valid range
    if ( gain_db < B210_GAIN_MIN ) gain_db = B210_GAIN_MIN;
    if ( gain_db > B210_GAIN_MAX ) gain_db = B210_GAIN_MAX;

    // Convert: (max gain - current gain) * scaling factor
    int32_t atten = (int32_t)((B210_GAIN_MAX - gain_db) * (30.0 / B210_GAIN_MAX));

    // Clamp to valid attenuation range
    if ( atten < 0 ) atten = 0;
    if ( atten > 30 ) atten = 30;

    return atten;
}

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
