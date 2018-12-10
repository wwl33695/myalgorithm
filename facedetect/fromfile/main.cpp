
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <unistd.h>

#include "avreader.h"

using namespace cv;  
using namespace std;

string face_cascade_name = "haarcascade_frontalface_alt.xml";
string eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

/** @函数 detectAndDisplay */
void detectAndDisplay( Mat frame )
{
  std::vector<Rect> faces;
  Mat frame_gray;

  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  //-- 多尺寸检测人脸
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

  for( int i = 0; i < faces.size(); i++ )
  {
    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

    Mat faceROI = frame_gray( faces[i] );
    std::vector<Rect> eyes;

    //-- 在每张人脸上检测双眼
    eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    for( int j = 0; j < eyes.size(); j++ )
     {
       Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
       int radius = cvRound( (eyes[j].width + eyes[i].height)*0.25 );
       circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
     }
  }
  //-- 显示结果图像
  imshow( "FaceDetect", frame );
}

/* 参数 : 输入图像、级联分类器、缩放倍数 */
void DetectAndDraw( Mat& img, CascadeClassifier& cascade, double scale = 4 )
{
    double t = 0;
    vector<Rect> faces;
    Mat gray, smallImg;
    double fx = 1 / scale;
	
    cvtColor( img, gray, COLOR_BGR2GRAY );	// 将源图像转为灰度图
 
	/* 缩放图像 */
#ifdef VERSION_2_4	
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR );
#else
	resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT );
#endif
 
    equalizeHist( smallImg, smallImg );	// 直方图均衡化，提高图像质量
 
	/* 检测目标 */
    t = (double)getTickCount();
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE,
        Size(30, 30) );
    t = (double)getTickCount() - t;
    printf( "detection time = %g ms\n", t*1000/getTickFrequency());
 
	/* 画矩形框出目标 */
    for ( size_t i = 0; i < faces.size(); i++ ) // faces.size():检测到的目标数量
    {
        Rect rectFace = faces[i];
		
        rectangle(	img, Point(rectFace.x, rectFace.y) * scale, 
					Point(rectFace.x + rectFace.width, rectFace.y + rectFace.height) * scale,
					Scalar(0, 255, 0), 2, 8);
    }
 
    imshow( "FaceDetect", img );	// 显示
}

int myrgb24_data_callback(int width, int height, const char* data)
{
	printf("myrgb24_data_callback width:%u, height:%u \n", width,height);

    Mat frame(height, width, CV_8UC3, (char*)data);

	DetectAndDraw( frame, face_cascade );

//    detectAndDisplay( frame );
    
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

   //-- 1. 加载级联分类器文件
   if( !face_cascade.load( face_cascade_name ) ){ printf("load error: face_cascade_name\n"); return -1; };
   if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("load error: eyes_cascade_name\n"); return -1; };
 

	return read_file(argv[1], myrgb24_data_callback);
}  
