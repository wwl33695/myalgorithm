#ifndef SHOTDETECT_ALG_H
#define SHOTDETECT_ALG_H

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>
#include <string>
#include <opencv2/opencv.hpp>
#include "types.h"

#ifdef __cplusplus
extern "C"{
#endif

using namespace cv;

/* 
 * 计算图像直方图（图像划分resize_dim*resize_dim块，每块对应hist_dim个bin，所有bin值和为1，resize_dim现在为固定值2）
 */
int calc_histogram(im_t *img, float *hist_ptr, int hist_dim = 64);

/* 
 * 批量模式
 * hist_ptr长度是 length*hist_dim*resize_dim*resize_dim
 */
void calc_score_batch(float *output, int length, float *hist_ptr, int hist_dim = 64, int resize_dim = 2);

vector<float> getColorSapceHist(Mat &image);

MatND getHsvHist(Mat &image);

double ssim(Mat &i1, Mat & i2);

double face_detect(char* pklpath, char* imgpath);

std::map<int, double> face_detect_mutiple(int multiple_cnt, char* pklpath, char* filename);

int surf_detect(char* pklpath, char* imgpath);

std::map<int, int> surf_detect_mutiple(int multiple_cnt, char* pklpath, char* filename);

double face_match(char* imgpath);

int facedect_seeta(char* imgpath);

double facematch_seeta(char* imgpath, char* filename, char* labelpath);

int caffe_detect(char* pklpath, char* imgpath);

std::map<int, int> caffe_detect_mutiple(int multiple_cnt, char* pklpath, char* filename);


#ifdef __cplusplus
};
#endif

#endif
