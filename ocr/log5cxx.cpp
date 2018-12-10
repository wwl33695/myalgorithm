#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <assert.h>
#include <iostream>
#include "log5cxx.h"
#ifdef linux
#include <glog/logging.h>
#endif

using namespace std;

void LOG5CXX_INIT(const char* v){
#ifdef linux
    // Initialize Google's logging library.
    google::InitGoogleLogging(v);
#endif
}
void LOG5CXX_DEBUG(const char* msg){
#ifdef linux
    DLOG(INFO) << msg;
    //DLOG_IF(INFO, num_cookies > 10) << "Got lots of cookies";
    //DLOG_EVERY_N(INFO, 10) << "Got the " << google::COUNTER << "th cookie";
#endif
cout << "<< DEBUG >> : " << msg << endl;
}
void LOG5CXX_INFO(const char* msg){
#ifdef linux
    LOG(INFO) << msg;
#endif
cout << "<< INFO >> : " << msg << endl;
}
void LOG5CXX_WARN(const char* msg){
#ifdef linux
    LOG(WARNING) << msg;
#endif
cerr << "<< WARN >> : " << msg << endl;
}
void LOG5CXX_ERROR(const char* msg){
#ifdef linux
    LOG(ERROR) << msg;
    //LOG_IF(ERROR, x > y) << "This should be also OK";
#endif
cerr << "<< ERROR >> : " << msg << endl;
}
void LOG5CXX_FATAL(const char* msg, int code){
#ifdef linux
    LOG(FATAL) << msg;
#endif
cerr << "<< FATAL >> : " << msg << endl;

    exit(code);    //terminal the app
}
