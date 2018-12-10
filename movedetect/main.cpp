
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <unistd.h>

#include "avreader.h"

using namespace cv;  
using namespace std;


Mat background;//存储背景图
int detectmethod = 0;//0-运动物体检测——背景减法；1-运动物体检测——帧差法

Mat MoveDetect(Mat background, Mat frame)
{
    Mat result = frame.clone();
    //1.将background和frame转为灰度图
    Mat gray1, gray2;
    cvtColor(background, gray1, CV_BGR2GRAY);
    cvtColor(frame, gray2, CV_BGR2GRAY);
    //2.将background和frame做差
    Mat diff;
    absdiff(gray1, gray2, diff);
    imshow("diff", diff);
    //3.对差值图diff_thresh进行阈值化处理
    Mat diff_thresh;
    threshold(diff, diff_thresh, 50, 255, CV_THRESH_BINARY);
    imshow("diff_thresh", diff_thresh);
    //4.腐蚀
    Mat kernel_erode = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat kernel_dilate = getStructuringElement(MORPH_RECT, Size(15, 15));
    erode(diff_thresh, diff_thresh, kernel_erode);
    imshow("erode", diff_thresh);
    //5.膨胀
    dilate(diff_thresh, diff_thresh, kernel_dilate);
    imshow("dilate", diff_thresh);
    //6.查找轮廓并绘制轮廓
    vector<vector<Point>> contours;
    findContours(diff_thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    drawContours(result, contours, -1, Scalar(0, 0, 255), 2);//在result上绘制轮廓
    //7.查找正外接矩形
    vector<Rect> boundRect(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        boundRect[i] = boundingRect(contours[i]);
        rectangle(result, boundRect[i], Scalar(0, 255, 0), 2);//在result上绘制正外接矩形
    }
    return result;//返回result
}

int myrgb24_data_callback(int width, int height, const char* data)
{
	printf("myrgb24_data_callback width:%u, height:%u \n", width,height);

    Mat frame(height, width, CV_8UC3, (char*)data);
    if (background.empty())//将第一帧作为背景图像
    {
        background = frame.clone();
    }

    imshow("rawvideo", frame);
    Mat result = MoveDetect(background, frame);//调用MoveDetect()进行运动物体检测，返回值存入result
    imshow("result", result);

    if( detectmethod )
    {
        background = frame.clone();
    }

    cvWaitKey(40);  

	return 0;
}

int main(int argc, char* argv[])
{
	if( argc < 2 )
	{
		printf("argument error\n");
		return -1;
	}
 
	return read_file(argv[1], myrgb24_data_callback);
}  
