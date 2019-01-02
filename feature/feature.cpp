#include "feature.h"

Feature::Feature()
{
	m_detectType = "SIFT";
	m_extractType = "SIFT";
	m_matchType = "FruteForce";
//	initModule_nonfree();
}

Feature::~Feature()
{

}


Feature::Feature(const string& detectType, const string& extractType, const string& matchType)
{
	assert(!detectType.empty());

	m_detectType = detectType;
	m_extractType = extractType;
	m_matchType = matchType;
//	initModule_nonfree(); 

	m_detector = cv::xfeatures2d::SurfFeatureDetector::create();
	m_extractor = cv::xfeatures2d::SurfDescriptorExtractor::create();
	m_matcher = DescriptorMatcher::create(m_matchType);
}


void Feature::detectKeypoints(const Mat& image, std::vector<KeyPoint>& keypoints) 
{
	assert(image.type() == CV_8UC1);

	keypoints.clear();
	m_detector->detect(image, keypoints);

}

void Feature::extractDescriptors(const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptor)
{
	assert(image.type() == CV_8UC1);

	m_extractor->compute(image, keypoints, descriptor);

}

void Feature::bestMatch(const Mat& queryDescriptor, Mat& trainDescriptor, std::vector<DMatch>& matches) 
{
	matches.clear();

	m_matcher->match(queryDescriptor, trainDescriptor, matches);
}

void Feature::knnMatch(const Mat& queryDescriptor, Mat& trainDescriptor, std::vector<std::vector<DMatch>>& matches, int k)
{
	assert(k > 0);
	matches.clear();

	m_matcher->knnMatch(queryDescriptor, trainDescriptor, matches, k);
}

void Feature::libBestMatch(const Mat& queryDescriptor, vector<DMatch>& matches)
{
	matches.clear();

	m_matcher->match(queryDescriptor, matches);
}

void Feature::libKnnMatch(const Mat& queryDescriptor, std::vector<std::vector<DMatch>>& matches, int k)
{
	assert(k > 0);
	matches.clear();

	m_matcher->knnMatch(queryDescriptor, matches, k);
}

void Feature::saveKeypoints(const Mat& image, const vector<KeyPoint>& keypoints, const string& saveFileName)
{
	assert(!saveFileName.empty());

	Mat outImage;
	cv::drawKeypoints(image, keypoints, outImage, Scalar(255,255,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

	//
	string saveKeypointsImgName = saveFileName + "_" + m_detectType + ".jpg";
	imwrite(saveKeypointsImgName, outImage);
}

void Feature::addTrainDescriptor(const Mat& trainDescriptor)
{
	m_matcher->add(std::vector<Mat>(1, trainDescriptor));
	m_matcher->train();	
}


void Feature::saveMatches(const Mat& queryImage,
							const vector<KeyPoint>& queryKeypoints,
							const Mat& trainImage,
							const vector<KeyPoint>& trainKeypoints,
							const vector<DMatch>& matches,
							const string& saveFileName)
{
	assert(!saveFileName.empty());

	Mat outImage;
	cv::drawMatches(queryImage, queryKeypoints, trainImage, trainKeypoints, matches, outImage);

	//
	string saveMatchImgName = saveFileName + "_" + m_detectType + "_" + m_extractType + "_" + m_matchType + ".jpg";
	imwrite(saveMatchImgName, outImage);
}

int Feature::getbestmatch(std::vector<cv::DMatch>& matches, float &distance)
{
	if( matches.empty() )
		return -1;

	distance = matches[0].distance;

	for( auto iter : matches )
	{
		if( iter.distance < distance )
			distance = iter.distance;
	}

	return 0;
}
