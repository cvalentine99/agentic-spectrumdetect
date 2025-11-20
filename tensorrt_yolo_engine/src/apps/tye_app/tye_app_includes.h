#ifndef INCLUDE_TYE_SP_INCLUDES_H
#define INCLUDE_TYE_SP_INCLUDES_H

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include <iostream>
#include <exception>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <limits>
#include <chrono>
#include <vector>
#include <tuple>
#include <filesystem>
#include <thread>
#include <mutex>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <csignal>
#include <ctime>

#include <unistd.h>
#include <getopt.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <GL/gl.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>
#include <bsoncxx/json.hpp>
#include <mongocxx/exception/exception.hpp>
#include <mongocxx/instance.hpp>
#include <mongocxx/client.hpp>
#include <mongocxx/uri.hpp>

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

#ifdef TYE_STREAM_PROCESSOR
    #ifdef USE_SIGNALHOUND
        #include "sm_api.h"
    #endif
    #ifdef USE_UHD_B210
        #include <uhd/usrp/multi_usrp.hpp>
        #include <uhd/utils/thread.hpp>
        #include <uhd/utils/safe_main.hpp>
        #include <uhd/types/tune_request.hpp>
    #endif
#endif

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_TYE_SP_INCLUDES_H
