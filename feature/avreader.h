#ifndef __AV_READER_H__
#define __AV_READER_H__

typedef int (*rgb24_data_callback)(int width, int height, const char* data);

int read_file(const char* filepath, rgb24_data_callback callback);

#endif