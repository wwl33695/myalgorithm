#ifndef _LOG5CXX_H
#define _LOG5CXX_H

// Level.ALL < Level.DEBUG < Level.INFO < Level.WARN < Level.ERROR < Level.FATAL < Level.OFF
void LOG5CXX_INIT(const char* v);
void LOG5CXX_DEBUG(const char* msg);
void LOG5CXX_INFO(const char* msg);
void LOG5CXX_WARN(const char* msg);
void LOG5CXX_ERROR(const char* msg);
void LOG5CXX_FATAL(const char* msg, int code);        // terminates the program after the message is logged.

#endif