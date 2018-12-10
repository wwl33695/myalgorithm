#include <math.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include "fingerprint_alg.h"
#include "log5cxx.h"
#include "types.h"

using namespace std;
using namespace cv;

// depth
// enum { CV_8U=0, CV_8S=1, CV_16U=2, CV_16S=3, CV_32S=4, CV_32F=5, CV_64F=6 }
#define CV_BUILD_MATRIX_TYPE cv_build_matrix_type

static im_t* _shrink_to_88_64_(im_t *img)
{
    WARPCVMAT(img_mat, img);
    Mat img_mat_gray, img_mat_88;
    if (img_mat.channels() != 1) {
        cvtColor(img_mat, img_mat_gray, CV_BGR2GRAY);
        resize(img_mat_gray, img_mat_88, Size(8, 8));
    } else
        resize(img_mat, img_mat_88, Size(8, 8));
    
    //release_image(img);

    im_t *img_88 = (im_t*)malloc(sizeof(im_t));
    img_88->rows = 8;
    img_88->cols = 8;
    img_88->channels = 1;
    img_88->depth = CV_8U;

    char *ptr = (char*)malloc(64);
    if (!ptr) {
        char msg[512];
        sprintf(msg, "Memory malloc or memcpy failed, operate memory size is [%d]=[%d MB].", 64, 64/1024/1024);
        LOG5CXX_FATAL(msg, 210);
    }

    memcpy(ptr, img_mat_88.data, 64);
    img_88->data = ptr;

    // 256色度级到64色度级


    return img_88;
}

static im_t* _shrink_to_1616_256_(im_t *img)
{
    WARPCVMAT(img_mat, img);
    Mat img_mat_gray, img_mat_1616;
    if (img_mat.channels() != 1) {
        cvtColor(img_mat, img_mat_gray, CV_BGR2GRAY);
        resize(img_mat_gray, img_mat_1616, Size(16, 16));
    } else
        resize(img_mat, img_mat_1616, Size(16, 16));
    
    //release_image(img);

    im_t *img_1616 = (im_t*)malloc(sizeof(im_t));
    img_1616->rows = 16;
    img_1616->cols = 16;
    img_1616->channels = 1;
    img_1616->depth = CV_8U;

    char *ptr = (char*)malloc(256);
    if (!ptr) {
        char msg[512];
        sprintf(msg, "Memory malloc or memcpy failed, operate memory size is [%d]=[%d MB].", 256, 256/1024/1024);
        LOG5CXX_FATAL(msg, 210);
    }

    memcpy(ptr, img_mat_1616.data, 256);
    img_1616->data = ptr;

    // 256色度级到64色度级


    return img_1616;
}

static fp_t* _calc_fp_phash(im_t *img_88_64, fp_t *fp)
{
    WARPCVMAT(img_mat, img_88_64);
    Mat img_mat_float, img_mat_dst;

    // 1. DCT变换
    img_mat_float = Mat_<double>(img_mat);
    dct(img_mat_float, img_mat_dst);

    // 2. 求DCT系数均值
    double mean = 0, *ptr = (double*)img_mat_dst.data;
    for (int i=0; i<64; i++) {
        mean += ptr[i];
    }
    mean /= 64;

    // 3. 计算哈希值
    if (fp == NULL)
        fp = (fp_t*)malloc(sizeof(fp_t));
    fp->type = 0;
    fp->len = 64;
    for (int i=0; i<64; i++) {
        if (ptr[i] > mean)
            fp->fp[i] = 1;
        else
            fp->fp[i] = 0;
    }

    return fp;
}

static fp_t* _calc_fp_phash_16(im_t *img_1616_256, fp_t *fp)
{
    WARPCVMAT(img_mat, img_1616_256);
    Mat img_mat_float, img_mat_dst;

    // 1. DCT变换
    img_mat_float = Mat_<double>(img_mat);
    dct(img_mat_float, img_mat_dst);

    // 2. 求DCT系数均值
    double mean = 0, *ptr = (double*)img_mat_dst.data;
    for (int i=0; i<256; i++) {
        mean += ptr[i];
    }
    mean /= 256;

    // 3. 计算哈希值
    if (fp == NULL)
        fp = (fp_t*)malloc(sizeof(fp_t));
    fp->type = 0;
    fp->len = 256;
    for (int i=0; i<256; i++) {
        if (ptr[i] > mean)
            fp->fp[i] = 1;
        else
            fp->fp[i] = 0;
    }

    return fp;
}

extern "C" void calc_fingerprint_phash(fp_t *fp, im_t *img)
{
    im_t *img_88_64 = _shrink_to_88_64_(img);
    fp_t *_fp = _calc_fp_phash(img_88_64, fp);

    if (img_88_64 != NULL)
        release_image(img_88_64);
    
    if (fp == NULL) {
        free(_fp);
        _fp = NULL;
    }
    //memcpy(fp, _fp, sizeof(fp_t));

/*
	im_t *img_1616_256 = _shrink_to_1616_256_(img);
    fp_t *_fp = _calc_fp_phash_16(img_1616_256, fp);

    if (img_1616_256 != NULL)
        release_image(img_1616_256);
    
    if (fp == NULL) {
        free(_fp);
        _fp = NULL;
    }
    //memcpy(fp, _fp, sizeof(fp_t));
*/
}

extern "C" void calc_fingerprint_hash(fp_t *fp, im_t *img)
{

}

extern "C" int hanming_distance(fp_t *fp1, fp_t *fp2)
{
    if (fp1->type != fp2->type)
        return 256;

    int i, diff = 0;
    for (i=0; i<fp1->len; i++)
        if (fp1->fp[i] != fp2->fp[i])
            diff++;

    return diff;
}
