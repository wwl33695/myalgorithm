
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>
#include <unistd.h>

#include "avreader.h"
#include "feature.h"

using namespace cv;  
using namespace std;

int SkinDetect(Mat &input_image)
{      
    std::string detectorType = "SIFT";
    std::string extractorType = "SIFT";
    std::string matchType = "FruteForce";
    Feature feature(detectorType, extractorType, matchType);  
      
    vector<KeyPoint> queryKeypoints;  
    feature.detectKeypoints(input_image, queryKeypoints);  
      
    Mat queryDescriptor;        
    feature.extractDescriptors(input_image, queryKeypoints, queryDescriptor);  

/*      
    vector matches;  
    feature.bestMatch(queryDescriptor, trainDescriptor, matches);  

      
    vector knnmatches;  
    feature.knnMatch(queryDescriptor, trainDescriptor, knnmatches, 2);  
      
    Mat outImage;  
    feature.saveMatches(queryImage, queryKeypoints, trainImage, trainKeypoints, matches, "../");  
*/

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
