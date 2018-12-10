#include "cv_image.h"
#include <stdio.h>

int testcv1()
{
    void *handle = cvimage_alloc("lena.jpg");
    cvimage_dialate(handle);
    cvimage_writefile(handle, "out.jpg");

    cvimage_free(handle);

    return 0;
}

int testcv2()
{
    void *handle = cvimage_alloc("lena.jpg");
    cvimage_sharpen(handle);
    cvimage_writefile(handle, "out.jpg");

    cvimage_free(handle);

    return 0;
}

int testcv3()
{
    void *handle = cvimage_alloc("lena.jpg");
    cvimage_imageenhance_gamma(handle);
    cvimage_writefile(handle, "out.jpg");

    cvimage_free(handle);

    return 0;
}

int testcv4()
{
    void *handle = cvimage_alloc("lena.jpg");
    cvimage_imageenhance_log(handle);
    cvimage_writefile(handle, "out.jpg");

    cvimage_free(handle);

    return 0;
}

int testcv5()
{
    void *handle = cvimage_alloc("lena.jpg");
    cvimage_imageenhance_histbalance(handle);
    cvimage_writefile(handle, "out.jpg");

    cvimage_free(handle);

    return 0;
}

int testcv6()
{
    void *handle = cvimage_alloc("lena.jpg");
    cvimage_imageenhance_laplace(handle);
    cvimage_writefile(handle, "out.jpg");

    cvimage_free(handle);

    return 0;
}

int testcv7()
{
    void *handle = cvimage_alloc("lena.jpg");
    cvimage_getHistogram1DImage(handle, 640, 480);
    cvimage_writefile(handle, "out.jpg");

    cvimage_free(handle);

    return 0;
}