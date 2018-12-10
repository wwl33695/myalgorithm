
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui_c.h>

#include <iostream>
#include <stdio.h>
#include <unistd.h>

#include "avreader.h"

using namespace cv;  
using namespace std;

int SkinDetect(Mat &input_image)
{
    if(input_image.empty())
        return -1;

    /*椭圆皮肤模型*/
    Mat skinCrCbHist = Mat::zeros(Size(256, 256), CV_8UC1);
    ellipse(skinCrCbHist, Point(113, 155.6), Size(23.4, 15.2), 43.0, 0.0, 360.0, Scalar(255, 255, 255), -1);

    Mat output_mask = Mat::zeros(input_image.size(), CV_8UC1);
    Mat ycrcb_image;
    cvtColor(input_image, ycrcb_image, CV_BGR2YCrCb); //首先转换成到YCrCb空间
    for(int i = 0; i < input_image.cols; i++)   //利用椭圆皮肤模型进行皮肤检测
        for(int j = 0; j < input_image.rows; j++){
        Vec3b ycrcb = ycrcb_image.at<Vec3b>(j, i);
        if(skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0)
            output_mask.at<uchar>(j, i) = 255;
    }

    Mat output_image;
    input_image.copyTo(output_image, output_mask);

    imshow("input image", input_image);
    imshow("output mask", output_mask);
    imshow("output image", output_image);

    return 0;
}

int myrgb24_data_callback(int width, int height, const char* data)
{
	printf("myrgb24_data_callback width:%u, height:%u \n", width,height);

    Mat frame(height, width, CV_8UC3, (char*)data);
    
    SkinDetect(frame);

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
