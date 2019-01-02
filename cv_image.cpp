#include <iostream>  

#include <stdio.h>

#include <opencv2/opencv.hpp>  

#include "cv_image.h"

struct cmimage_tag
{
    cv::Mat origin_image;
    cv::Mat processed_image;
};

void* cvimage_alloc(const char* infilename)
{

    cv::Mat image = cv::imread(infilename, cv::IMREAD_COLOR);
    if (image.empty())
    {
        printf("打开图片失败,请检查\n");
        return NULL;
    }

    cmimage_tag *inst = new cmimage_tag;
    inst->origin_image = image;
    return inst;
}

int cvimage_writefile(void* handle, const char* outfilename)
{
    cmimage_tag *inst = (cmimage_tag*)handle;

    cv::imwrite(outfilename, inst->processed_image);

    return 0;
}

int cvimage_free(void *handle)
{
    cmimage_tag *inst = (cmimage_tag*)handle;

    delete inst;
    
    return 0;
}

//锐化的作用是加强图像的边缘和轮廓，通常也成为高通滤波器 
int cvimage_sharpen(void* handle)  
{
    cmimage_tag *inst = (cmimage_tag*)handle;

    cv::Mat kernel(3,3,CV_32F,cv::Scalar(-1)); 
    // 分配像素值
    kernel.at<float>(1,1) = 9;
    inst->processed_image.create(inst->origin_image.size(), inst->origin_image.type());  
    filter2D(inst->origin_image, inst->processed_image, inst->origin_image.depth(),kernel);

    return 0;
}  

int cvimage_getHistogram1DImage(void* handle, int width, int height)
{  
   cmimage_tag *inst = (cmimage_tag*)handle;

    int narrays = 1;  
    int channels[] = { 0 };  
    cv::InputArray mask = cv::noArray();  
    cv::Mat hist;  
    int dims = 1;  
    int histSize[] = { 256 };      
    float hranges[] = { 0.0, 255.0 };  
    const float *ranges[] = { hranges };  
    //调用 calcHist 计算直方图, 结果存放在 hist 中  
    cv::calcHist(&inst->origin_image, narrays, channels, mask, hist, dims, histSize, ranges);  
    //调用一个我自己写的简单的函数用于获取一张显示直方图数据的图片,  
    //输入参数为直方图数据 hist 和期望得到的图片的尺寸  

   cv::Size imgSize = cv::Size(width, height);
   inst->processed_image.create(imgSize, CV_8UC3);
   int Padding = 10;  
   int W = imgSize.width - 2 * Padding;  
   int H = imgSize.height - 2 * Padding;  
   double _max;  
   cv::minMaxLoc(hist, NULL, &_max);  
   double Per = (double)H / _max;  
   const cv::Point Orig(Padding, imgSize.height-Padding);  
   int bin = W / (hist.rows + 2);  
 
   //画方柱  
   for (int i = 1; i <= hist.rows; i++)  
   {  
       cv::Point pBottom(Orig.x + i * bin, Orig.y);  
       cv::Point pTop(pBottom.x, pBottom.y - Per * hist.at<float>(i-1));  
       cv::line(inst->processed_image, pBottom, pTop, cv::Scalar(255, 0, 0), bin);  
   }  
 
   //画 3 条红线标明区域  
   cv::line(inst->processed_image, cv::Point(Orig.x + bin, Orig.y - H), cv::Point(Orig.x + hist.rows *  bin, Orig.y - H), cv::Scalar(0, 0, 255), 1);  
   cv::line(inst->processed_image, cv::Point(Orig.x + bin, Orig.y), cv::Point(Orig.x + bin, Orig.y - H), cv::Scalar(0, 0, 255), 1);  
   cv::line(inst->processed_image, cv::Point(Orig.x + hist.rows * bin, Orig.y), cv::Point(Orig.x + hist.rows *bin, Orig.y - H), cv::Scalar(0, 0, 255), 1);  
//   drawArrow(histImg, Orig, Orig+Point(W, 0), 10, 30, Scalar::all(0), 2);  
//   drawArrow(histImg, Orig, Orig-Point(0, H), 10, 30, Scalar::all(0), 2);  
     
   return 0;  
}  

int cvimage_showimage(int width, int height, const char* data)
{
    IplImage *img = cvCreateImage(cvSize(width,height),8,3);  
    img->imageData =(char *)data;//把ffmpeg格式转换到opencv的bgr格式。  

//    cvCvtColor(img,imgbgr,CV_RGB2BGR);  
//    cvCvtColor(img,imgbgr,CV_BGR2HSV);  
    cvShowImage("HSV",img);  

    cvReleaseImage(&img);
    cvWaitKey(2);  
    return 0;
}

int cvimage_showimage2(int width, int height, const char* data)
{
    Mat img(height, width, CV_8UC3, (char*)data);
    imshow( "image", img );
    cvWaitKey(2);  

    return 0;
}

int cvimage_dialate(void* handle)
{
    cmimage_tag *inst = (cmimage_tag*)handle;

    //获取自定义核  
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));  

    //进行膨胀操作  
    cv::dilate(inst->origin_image, inst->processed_image, element);  

    return 0;
}

//基于拉普拉斯算子的图像增强
int cvimage_imageenhance_laplace(void* handle)
{
    cmimage_tag *inst = (cmimage_tag*)handle;

//    cv::Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);  
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
    cv::filter2D(inst->origin_image, inst->processed_image, CV_8UC3, kernel);

    return 0;
}

//基于直方图均衡化的图像增强
int cvimage_imageenhance_histbalance(void* handle)
{
    cmimage_tag *inst = (cmimage_tag*)handle;

    cv::Mat imageRGB[3];
    cv::split(inst->origin_image, imageRGB);
    for (int i = 0; i < 3; i++)
    {
        cv::equalizeHist(imageRGB[i], imageRGB[i]);
    }
    merge(imageRGB, 3, inst->processed_image);

    return 0;
}

//基于对数Log变换的图像增强
int cvimage_imageenhance_log(void* handle)
{
    cmimage_tag *inst = (cmimage_tag*)handle;

    inst->processed_image.create(inst->origin_image.size(), CV_32FC3);//inst->origin_image.type());  

    for (int i = 0; i < inst->origin_image.rows; i++)
    {
      for (int j = 0; j < inst->origin_image.cols; j++)
      {
        inst->processed_image.at<cv::Vec3f>(i, j)[0] = log(1 + inst->origin_image.at<cv::Vec3b>(i, j)[0]);
        inst->processed_image.at<cv::Vec3f>(i, j)[1] = log(1 + inst->origin_image.at<cv::Vec3b>(i, j)[1]);
        inst->processed_image.at<cv::Vec3f>(i, j)[2] = log(1 + inst->origin_image.at<cv::Vec3b>(i, j)[2]);
      }
    }
    //归一化到0~255  
    cv::normalize(inst->processed_image, inst->processed_image, 0, 255, CV_MINMAX);
    //转换成8bit图像显示  
    cv::convertScaleAbs(inst->processed_image, inst->processed_image);

    return 0;
}

//基于伽马变换的图像增强
int cvimage_imageenhance_gamma(void* handle)
{
    cmimage_tag *inst = (cmimage_tag*)handle;

    inst->processed_image.create(inst->origin_image.size(), CV_32FC3);

    for (int i = 0; i < inst->origin_image.rows; i++)
    {
      for (int j = 0; j < inst->origin_image.cols; j++)
      {
          inst->processed_image.at<cv::Vec3f>(i, j)[0] = (inst->origin_image.at<cv::Vec3b>(i, j)[0])*(inst->origin_image.at<cv::Vec3b>(i, j)[0])*(inst->origin_image.at<cv::Vec3b>(i, j)[0]);
          inst->processed_image.at<cv::Vec3f>(i, j)[1] = (inst->origin_image.at<cv::Vec3b>(i, j)[1])*(inst->origin_image.at<cv::Vec3b>(i, j)[1])*(inst->origin_image.at<cv::Vec3b>(i, j)[1]);
          inst->processed_image.at<cv::Vec3f>(i, j)[2] = (inst->origin_image.at<cv::Vec3b>(i, j)[2])*(inst->origin_image.at<cv::Vec3b>(i, j)[2])*(inst->origin_image.at<cv::Vec3b>(i, j)[2]);
      }
    }
    //归一化到0~255  
    cv::normalize(inst->processed_image, inst->processed_image, 0, 255, CV_MINMAX);
    //转换成8bit图像显示  
    cv::convertScaleAbs(inst->processed_image, inst->processed_image);

    return 0;
}
//均值滤波
int cvimage_blur_average(void* handle)
{
    cmimage_tag *inst = (cmimage_tag*)handle;

    inst->processed_image.create(inst->origin_image.size(), CV_32FC3);

    blur(inst->origin_image, inst->processed_image, cv::Size(3, 3), cv::Point(-1, -1));

    return 0;
}
//高斯滤波
int cvimage_blur_gaussian(void* handle)
{
    cmimage_tag *inst = (cmimage_tag*)handle;

    inst->processed_image.create(inst->origin_image.size(), CV_32FC3);

    GaussianBlur(inst->origin_image, inst->processed_image, cv::Size(5, 5), 5, 5);

    return 0;
}
//中值滤波
int cvimage_blur_median(void* handle)
{
    cmimage_tag *inst = (cmimage_tag*)handle;

    inst->processed_image.create(inst->origin_image.size(), CV_32FC3);
    
    medianBlur(inst->origin_image, inst->processed_image, 5);

    return 0;
}

//双边滤波--结果最清晰
int cvimage_blur_bilateral(void* handle)
{
    cmimage_tag *inst = (cmimage_tag*)handle;

    inst->processed_image.create(inst->origin_image.size(), CV_32FC3);

    bilateralFilter(inst->origin_image, inst->processed_image, 5, 100, 3);

    return 0;
}

int cvimage_drawrectangle(void* handle, int left, int right, int top, int bottom)
{
    cmimage_tag *inst = (cmimage_tag*)handle;

    cv::rectangle(inst->origin_image, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0), 2);//在result上绘制正外接矩形

    return 0;
}