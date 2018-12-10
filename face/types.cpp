#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include "types.h"
#include "log5cxx.h"

using namespace std;
using namespace cv;

static char pp[][1024] = {
    "#                          %d %% %s\r",
    "##                         %d %% %s\r",
    "###                        %d %% %s\r",
    "####                       %d %% %s\r",
    "#####                      %d %% %s\r",
    "######                     %d %% %s\r",
    "#######                    %d %% %s\r",
    "########                   %d %% %s\r",
    "#########                  %d %% %s\r",
    "##########                 %d %% %s\r",
    "###########                %d %% %s\r",
    "############               %d %% %s\r",
    "#############              %d %% %s\r",
    "##############             %d %% %s\r",
    "###############            %d %% %s\r",
    "################           %d %% %s\r",
    "#################          %d %% %s\r",
    "##################         %d %% %s\r",
    "###################        %d %% %s\r",
    "####################       %d %% %s\r",
    "#####################      %d %% %s\r",
    "######################     %d %% %s\r",
    "#######################    %d %% %s\r",
    "########################   %d %% %s\r",
    "#########################  %d %% %s\r"
};

extern "C" im_t* load_image(const char *path)
{
    Mat img = imread(path);
    if(img.empty()){
        // read image failed, log it and quit.
        char msg[1024];
        sprintf(msg, "Read image file failed, image data not exist or error. [%s]", path);
        LOG5CXX_FATAL(msg, 110);
    }

    im_t *t = (im_t*)malloc(sizeof(im_t));
    t->rows = img.rows;
    t->cols = img.cols;
    t->channels = img.channels();
    t->depth = img.depth();

    if(t->depth != CV_8U){
        char msg[512];
        sprintf(msg, "The image depth[%d] is not support. (only CV_8U image is valid)", t->depth);
        LOG5CXX_FATAL(msg, 110);
    }

    try{
        t->data = malloc(t->rows*t->cols*t->channels);
        memcpy(t->data, img.data, t->rows*t->cols*t->channels);
    }catch(...){
        char msg[512];
        sprintf(msg, "Memory malloc or memcpy failed, operate memory size is [%d]=[%d MB].", t->rows*t->cols*t->channels, t->rows*t->cols*t->channels/1024/1024);
        LOG5CXX_FATAL(msg, 210);
    }
    
    return t;
}

extern "C" void release_image(im_t *img)
{
    if (!img)
        return;

    if (img->data) {
        free(img->data);
        img->data = NULL;
    }
    
    if (img) {
        free(img);
        img = NULL;
    }
}

extern "C" int cv_build_matrix_type(int depth, int channels)
{
    // Determine type of the matrix
    switch (depth){
    case CV_8U:
    case CV_8S:
         return CV_8UC(channels);
         break;
    case CV_16U:
    case CV_16S:
         return CV_16UC(channels);
         break;
    case CV_32S:
    case CV_32F:
         return CV_32FC(channels);
         break;
    case CV_64F:
         return CV_64FC(channels);
         break;
    }
    return -1;
}

extern "C" void percent(int percentage)
{
    char BUF[1024];
    //sprintf(BUF, "PER: [ %d %%], MSG: [ %s ]\r", percentage, msg);

    int nn = percentage/4;
    sprintf(BUF, pp[nn], percentage, "");
    printf(BUF);

    if(percentage == 100)
        printf("\n");
    fflush(stdout);
}