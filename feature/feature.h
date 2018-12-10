#ifndef _FEATURE_H_ 
#define _FEATURE_H_

#include <string>
#include <vector>
//#include 

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace cv;
using namespace std;

class Feature
{
public:
	Feature();
	~Feature();

	Feature(const string& detectType, const string& extractType, const string& matchType);

public:
	
	void detectKeypoints(const Mat& image, vector<KeyPoint>& keypoints);   // 检测特征点
	void extractDescriptors(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptor);   // 提取特征向量
	void bestMatch(const Mat& queryDescriptor, Mat& trainDescriptor, vector<DMatch>& matches);  // 最近邻匹配
	void knnMatch(const Mat& queryDescriptor, Mat& trainDescriptor, std::vector<std::vector<DMatch>>& matches, int k);   // K近邻匹配

	void libBestMatch(const Mat& queryDescriptor, vector<DMatch>& matches);  // 最近邻匹配
	void libKnnMatch(const Mat& queryDescriptor, std::vector<std::vector<DMatch>>& matches, int k);   // K近邻匹配

	void addTrainDescriptor(const Mat& image);   // 检测特征点

	void saveKeypoints(const Mat& image, const vector<KeyPoint>& keypoints, const string& saveFileName = "");  // 保存特征点
	void saveMatches(const Mat& queryImage,
			 const vector<KeyPoint>& queryKeypoints,
			 const Mat& trainImage,
			 const vector<KeyPoint>& trainKeypoints,
			 const vector<DMatch>& matches,
			 const string& saveFileName = "");   // 保存匹配结果到图片中

private:
	Ptr<FeatureDetector> m_detector;
	Ptr<DescriptorExtractor> m_extractor;
	Ptr<DescriptorMatcher> m_matcher;

	string m_detectType;
	string m_extractType;
	string m_matchType;

};


#endif