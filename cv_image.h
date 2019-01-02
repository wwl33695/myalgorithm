#ifndef __CV_IMAGE_H__
#define __CV_IMAGE_H__

void* cvimage_alloc(const char* infilename);

int cvimage_writefile(void* handle, const char* outfilename);

int cvimage_free(void *handle);

int cvimage_sharpen(void* handle);

int cvimage_getHistogram1DImage(void* handle, int width, int height);

int cvimage_showimage(int width, int height, const char* data);
int cvimage_showimage2(int width, int height, const char* data);

int cvimage_dialate(void* handle);

int cvimage_imageenhance_gamma(void* handle);
int cvimage_imageenhance_log(void* handle);
int cvimage_imageenhance_histbalance(void* handle);
int cvimage_imageenhance_laplace(void* handle);

int cvimage_blur_bilateral(void* handle);
int cvimage_blur_median(void* handle);
int cvimage_blur_gaussian(void* handle);
int cvimage_blur_average(void* handle);

int cvimage_drawrectangle(void* handle, int left, int right, int top, int bottom);
#endif