
#include <math.h>
#include <time.h>
#include <dirent.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
//#include "shotdetect_alg.h"
#include "fingerprint_alg.h"
#include "log5cxx.h"
#include "types.h"

#include "surflib.h"
#include "kmeans.h"
#include <ctime>
#include <iostream>
#include <pthread.h>

using namespace std;
using namespace cv;

pthread_mutex_t g_mutex_surf;

static char path[4096];
static const char* getFilePath()
{
	readlink("/proc/self/exe", path, MAX_PATH);
    int len = strlen(path);
    while(1){
        if(path[len-1] != '/')
            path[len-1] = '\0';
        else
            break;
        len--;
    }
	return path;
}

// 图像划分4块，根据图像行列坐标，返回所在块索引
static inline int _grid_calc_(int r, int c, int rows, int cols)
{
    const int resize_dim = 2, hr = rows/resize_dim, hc = cols/resize_dim;

    if (r < hr){
        if (c < hc)
            return 0;
        else
            return 1;
    } else {
        if (c < hc)
            return 2;
        else
            return 3;
    }

    return -1;
}
// 计算图像4块的直方图（共4个直方图）
static void _imhist_(im_t *img, float *hist_ptr, int hist_dim)
{
    const int rows = img->rows, cols = img->cols, len = rows*cols;
    int r,c,k,ind;

    unsigned char *pixel = (unsigned char*)(img->data);
    float *ptr = NULL, scale = float(hist_dim)/255.0;
    for (k=0; k<len; k++) {
        r = k/cols;
        c = k%cols;

        ind = _grid_calc_(r, c, rows, cols);
        if (ind < 0)
            continue;

        ptr = hist_ptr+ind*hist_dim;
        ind = float(pixel[k])*scale;
        if (ind < 0) ind = 0;
        if (ind >= hist_dim) ind = hist_dim - 1;

        ptr[ind] += 1;
    }

    // 直方图归一化到0-1
    double v = rows*cols/4;
    for (k=0; k<hist_dim*4; k++)
        hist_ptr[k] /= v;
}
// 计算图像直方图
extern "C" int calc_histogram(im_t *img, float *hist_ptr, int hist_dim = 64)
{
    _imhist_(img, hist_ptr, hist_dim);

    return 0;
}

// hist_ptr长度是 length*hist_dim*resize_dim*resize_dim
// resize_dim为图像分块一个方向维度的数量
extern "C" void calc_score_batch(float *output, int length, float *hist_ptr, int hist_dim = 64, int resize_dim = 2)
{
    const unsigned int dim = resize_dim*resize_dim, offset = hist_dim*dim;
    int i,j,k;

    for (k=1; k<length; k++) {
        float *hist_cur = hist_ptr + k*offset;
        float *hist_pre = hist_ptr + (k-1)*offset;

        double sum = 0, max = 0, v = 0;
        unsigned int block_offset;
        // 匹配4块图像直方图的相邻帧间1范数（绝对值和）的均值，并去除一个最大值
        for (i=0; i<dim; i++) {
            v = 0;
            block_offset = i*hist_dim;
            for (j=0; j<hist_dim; j++) 
                v += fabs(hist_cur[block_offset+j] - hist_pre[block_offset+j]);
            
            if (v > max)
                max = v;
            sum += v;
        }
        // 去除差异最大的一个直方图
        output[k] = (sum-max)/(dim-1);
    }
    output[0] = 0.0f;
}

extern "C" vector<float> getColorSapceHist(Mat &image)//RGB色彩空间的（4，4，4）模型。  
{  
    const int div = 64;  
    const int bin_num = 256 / div;  
    int nr = image.rows; // number of rows  
    int nc = image.cols; // number of columns  
    if (image.isContinuous())  {  
        // then no padded pixels    
        nc = nc*nr;  
        nr = 1;  // it is now a 1D array    
    }  
    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));// mask used to round the pixel value  
    uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0  
    int b, g, r;  
    vector<float> bin_hist;  
    int ord = 0;  
    float a[bin_num*bin_num*bin_num] = { 0 };  
    for (int j = 0; j < nr; j++) {  
        const uchar* idata = image.ptr<uchar>(j);  
        for (int i = 0; i < nc; i++) {  
            b = ((*idata++)&mask) / div;  
            g = ((*idata++)&mask) / div;  
            r = ((*idata++)&mask) / div;  
            ord = b * bin_num * bin_num + g * bin_num + r;  
            a[ord] += 1;  
        } // end of row        
    }  
    float sum = 0;  
//  cout << "a[i]2:" << endl;  
    for (int i = 0; i < div; i++)  
    {  
        sum += a[i];  
//      cout << a[i] << " ";  
    }  
    for (int i = 0; i < div; i++)//归一化  
    {  
        a[i] = a[i] / sum;  
        bin_hist.push_back(a[i]);  
    }  
    return bin_hist;  
  
}  

extern "C" MatND getHsvHist(Mat &image)//HSV
{
	Mat  hsv_base;  
	cvtColor(image, hsv_base, CV_BGR2HSV);
	
	/// 对hue通道使用30个bin,对saturatoin通道使用32个bin  
    int h_bins =8 , s_bins =8, v_bins = 8;  
    int hist_size[] = {h_bins, s_bins, v_bins}; 
	
	// hue的取值范围从0到1024, saturation取值范围从0到180  
    float h_ranges[] = { 0, 255 };  
	float s_ranges[] = { 0, 255 };  
    float v_ranges[] = { 0, 255 };  
    const float* ranges[] = { h_ranges, s_ranges, v_ranges };  
    // 使用第0和第1通道  
    int channels[] = { 0, 1, 2 };  
  
    /// 直方图  
    MatND hist_base;  
	
	/// 计算HSV图像的直方图  
    calcHist(&hsv_base, 1, channels, Mat(), hist_base, 3, hist_size, ranges, true, false);  
    //normalize(hist_base, hist_base, 0, 1, NORM_L1, -1, Mat());
	
	return hist_base;
}

extern "C" double ssim(Mat &i1, Mat & i2)
{  
    const double C1 = 6.5025, C2 = 58.5225;  
    int d = CV_32F;  
    Mat I1, I2;  
    i1.convertTo(I1, d);  
    i2.convertTo(I2, d);  
    Mat I1_2 = I1.mul(I1);  
    Mat I2_2 = I2.mul(I2);  
    Mat I1_I2 = I1.mul(I2);  
    Mat mu1, mu2;  
    GaussianBlur(I1, mu1, Size(11,11), 1.5);  
    GaussianBlur(I2, mu2, Size(11,11), 1.5);  
    Mat mu1_2 = mu1.mul(mu1);  
    Mat mu2_2 = mu2.mul(mu2);  
    Mat mu1_mu2 = mu1.mul(mu2);  
    Mat sigma1_2, sigam2_2, sigam12;  
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);  
    sigma1_2 -= mu1_2;  
  
    GaussianBlur(I2_2, sigam2_2, Size(11, 11), 1.5);  
    sigam2_2 -= mu2_2;  
  
    GaussianBlur(I1_I2, sigam12, Size(11, 11), 1.5);  
    sigam12 -= mu1_mu2;  
    Mat t1, t2, t3;  
    t1 = 2 * mu1_mu2 + C1;  
    t2 = 2 * sigam12 + C2;  
    t3 = t1.mul(t2);  
  
    t1 = mu1_2 + mu2_2 + C1;  
    t2 = sigma1_2 + sigam2_2 + C2;  
    t1 = t1.mul(t2);  
  
    Mat ssim_map;  
    divide(t3, t1, ssim_map);  
    Scalar mssim = mean(ssim_map);  
  
    double ssim = (mssim.val[0] + mssim.val[1] + mssim.val[2]) /3;  
    return ssim;  
} 


extern "C" double face_detect(char* pklpath, char* imgpath)
{
	double confidence = 0.0;
	char cmd[1024];
	memset(cmd, 0, 1024);
	
	// 人脸识别
	sprintf(cmd, "./demos/classifier.py infer %s %s | grep 'confidence' | awk '{print $4}'", pklpath, imgpath);

	//printf("%s\n", cmd);
	
	FILE* fp;
	fp = popen(cmd, "r");

	if (fp == NULL)
	{
		printf("check proc[%d] - popen is failed.----------------------------------------------------------------------------\n");
		fflush(stdout);

		_exit(-1);
		return 0;
	}

	char tmp[128];
	memset(tmp, 0, 128);
	while (NULL != fgets(tmp, 128, fp))
	{
		confidence = atof(tmp);
		memset(tmp, 0, 128);
	}

	pclose(fp);

	return confidence;
}

map<int, double> confidence_map;
typedef struct face_detect_context
{
	char *inputpath;
	char *pklpath;
} face_detect_context;

void *face_detect_callback(void *argv)
{
	face_detect_context* ctx = (face_detect_context*)argv;
	string str = ctx->inputpath;
	
	string strname = str.substr(0, str.find_last_of("_"));
	string stridr = str.substr(str.find_last_of("_") + 1, str.find_last_of(".") - 1 - str.find_last_of("_"));
	printf("[%s][%s]\n", strname.c_str(), stridr.c_str());
	double confidence = face_detect(ctx->pklpath, ctx->inputpath);
	
	long idr = atol(stridr.c_str());
	printf("[%ld]confidenc - %.3f\n", idr, confidence);
	confidence_map[idr] = confidence;
	
	char cmd[1024];
	memset(cmd, 0, 1024);
	sprintf(cmd, "rm -rf %s", ctx->inputpath);
	system(cmd);
	
}

extern "C" map<int, double> face_detect_mutiple(int multiple_cnt, char* pklpath, char* filename)
{
	char imgpath[1024];
	memset(imgpath, 0, 1024);
	sprintf(imgpath, "%simages", getFilePath());
	printf("images - %s\n", imgpath);
			
	DIR *directory_pointer;  
	struct dirent *entry;  
	if ((directory_pointer = opendir(imgpath)) == NULL)
	{  
		printf("Error open images[%s]\n", imgpath);  
		return confidence_map;  
	}
	else 
	{
		face_detect_context* ctx;
		int count = 0;
		while ((entry = readdir(directory_pointer)) != NULL)
		{
			if (entry->d_name[0] == '.') 
				continue;  
			printf("%s\n", entry->d_name);  
			
			string tmp = entry->d_name;
			if (tmp.find(filename) < 0)
				continue;
			
			char tmppath[1024];
			memset(tmppath, 0, 1024);
			sprintf(tmppath, "%simages/%s", getFilePath(), entry->d_name);
			printf("%s\n", tmppath); 
			
		    pthread_attr_t attr;
			pthread_attr_init(&attr);
			pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

			pthread_t thr;
			ctx = new face_detect_context();
			ctx->inputpath = (char*)malloc(1024);
			ctx->pklpath = (char*)malloc(1024);
			memset(ctx->inputpath, 0, 1024);
			memset(ctx->pklpath, 0, 1024);
			memcpy(ctx->inputpath, tmppath, 1024);
			memcpy(ctx->pklpath, pklpath, 1024);
			
			string str = ctx->inputpath;
			string strname = str.substr(0, str.find_last_of("_"));
			string stridr = str.substr(str.find_last_of("_") + 1, str.find_last_of(".") - 1 - str.find_last_of("_"));
			
			long idr = atol(stridr.c_str());
			confidence_map[idr] = -1.0;
			
			int ret = pthread_create(&thr, &attr, face_detect_callback, ctx);
			pthread_attr_destroy(&attr);
			
			count++;
			if (count >= multiple_cnt)
			{
				while (1)
				{
					usleep(1 * 1000000);
					int is_undef = 0;
					typedef map<int,double>::iterator IT;
					for(IT it = confidence_map.begin(); it != confidence_map.end(); it++) 
					{
						if (it->second < 0.0)
							is_undef = 1;
					}
					if (!is_undef)
						break;
				}
				count = 0;
				printf("reset calculate.\n");
				fflush(stdout);
			}
		}
		while (1)
		{
			usleep(1 * 1000000);
			int is_undef = 0;
			typedef map<int,double>::iterator IT;
			for(IT it = confidence_map.begin(); it != confidence_map.end(); it++) 
			{
				if (it->second < 0.0)
					is_undef = 1;
			}
			if (!is_undef)
				break;
		}
		printf("all calculated.\n");
		return confidence_map;
	}

}



map<char*, IpVec> surf_vec_map;
int mainStaticMatch(char* path1, char* path2, int size)
{
  int scale_height = size;
  CvSize sz;
  
  IplImage *img1 = NULL;
  IplImage *desc1 = NULL;
  pthread_mutex_lock(&g_mutex_surf);
  img1 = cvLoadImage(path1);
  //Mat mat1 = imread(path1, CV_LOAD_IMAGE_COLOR);
  //img1 = &IplImage(mat1);
  pthread_mutex_unlock(&g_mutex_surf);
  if (img1 == NULL || img1->width <= 0 || img1->height <= 0)
  {
	  if (img1)
		cvReleaseImage(&img1);
	  printf("img1 open error.\n");
	  return -1;
  }
  double scale = (double)scale_height / (double)img1->height;
  sz.width = img1->width * scale;
  sz.height = img1->height * scale;
  printf("img1 - [%d:%d]\n", sz.width, sz.height);
  desc1 = cvCreateImage(sz, img1->depth, img1->nChannels);
  cvResize(img1, desc1, CV_INTER_CUBIC);
  cvReleaseImage(&img1);
  IpVec ipts1;
  surfDetDes(desc1,ipts1,false,5,4,2,0.0004f);
  
  IpPairVec matches;
  typedef map<char*, IpVec>::iterator IT;
  if (surf_vec_map.find(path2) != surf_vec_map.end())
  {
	getMatches(ipts1, surf_vec_map[path2], matches);
	printf("Matches: %d\n", matches.size());
	cvReleaseImage(&desc1);
	return matches.size();
  }
  
  IplImage *img2 = NULL;
  IplImage *desc2 = NULL;
  pthread_mutex_lock(&g_mutex_surf);
  img2 = cvLoadImage(path2);
  //Mat mat2 = imread(path2, CV_LOAD_IMAGE_COLOR);
  //img2 = &IplImage(mat2);
  pthread_mutex_unlock(&g_mutex_surf);
  if (img2 == NULL || img2->width <= 0 || img2->height <= 0)
  {
	  if (img2)
		cvReleaseImage(&img2);
	  printf("img2 open error.\n");
	  return -1;
  }
  scale = (double)scale_height / (double)img2->height;
  sz.width = img2->width * scale;
  sz.height = img2->height * scale;
  printf("img2 - [%d:%d]\n", sz.width, sz.height);
  desc2 = cvCreateImage(sz, img2->depth, img2->nChannels);
  cvResize(img2, desc2, CV_INTER_CUBIC);                  
  cvReleaseImage(&img2);
  IpVec ipts2;
  surfDetDes(desc2,ipts2,false,5,4,2,0.0004f);
  
  surf_vec_map[path2] = ipts2;
  getMatches(ipts1,ipts2,matches);
  cvReleaseImage(&desc1);
  cvReleaseImage(&desc2);
  
/*
  for (unsigned int i = 0; i < matches.size(); ++i)
  {
    drawPoint(img1,matches[i].first);
    drawPoint(img2,matches[i].second);
  
    const int & w = img1->width;
    cvLine(img1,cvPoint(matches[i].first.x,matches[i].first.y),cvPoint(matches[i].second.x+w,matches[i].second.y), cvScalar(255,255,255),1);
    cvLine(img2,cvPoint(matches[i].first.x-w,matches[i].first.y),cvPoint(matches[i].second.x,matches[i].second.y), cvScalar(255,255,255),1);
  }
*/
  printf("Matches: %d\n", matches.size());
/*
  cvNamedWindow("1", CV_WINDOW_AUTOSIZE );
  cvNamedWindow("2", CV_WINDOW_AUTOSIZE );
  cvShowImage("1", img1);
  cvShowImage("2",img2);
  cvWaitKey(0);
*/
  return matches.size();
}



extern "C" int surf_detect(char* img1, char* img2)
{
/*
	int surf = -1;
	char cmd[1024];
	memset(cmd, 0, 1024);
	
	// surf
	sprintf(cmd, "./surfmatch %s %s 360 | grep 'Matches' | awk '{print $2}'", img1, img2);

	printf("%s\n", cmd);
	
	FILE* fp;
	fp = popen(cmd, "r");

	if (fp == NULL)
	{
		printf("check proc[%d] - popen is failed.----------------------------------------------------------------------------\n");
		fflush(stdout);

		_exit(-1);
		return surf;
	}

	char tmp[128];
	memset(tmp, 0, 128);
	while (NULL != fgets(tmp, 128, fp))
	{
		surf = atoi(tmp);
		memset(tmp, 0, 128);
	}

	pclose(fp);

	return surf;
*/

	return mainStaticMatch(img1, img2, 400);
}

map<int, int> surf_map;
map<int, string> surf_map_error;
typedef struct surf_detect_context
{
	char *inputpath1;
	char *inputpath2;
} surf_detect_context;

void *surf_detect_callback(void *argv)
{
	surf_detect_context* ctx = (surf_detect_context*)argv;
	string str = ctx->inputpath1;
	string strname = str.substr(0, str.find_last_of("_"));
	string stridr = str.substr(str.find_last_of("_") + 1, str.find_last_of(".") - 1 - str.find_last_of("_"));
	//printf("[%s][%s]\n", strname.c_str(), stridr.c_str());
	
	char imgpath[1024];
	memset(imgpath, 0, 1024);
	sprintf(imgpath, "%sdata/%s/scene", getFilePath(), ctx->inputpath2);
	//printf("scene - %s\n", imgpath);
	int surf = 0;
	int count = 0;
	
	DIR *directory_pointer;  
	struct dirent *entry;  
surf_open_dir:
	if ((directory_pointer = opendir(imgpath)) == NULL)
	{  
		printf("Error open images[%s]\n", imgpath);  
		count++;
		usleep(1.0 * 1000000);
		if (count < 3)
			goto surf_open_dir;
		else
			goto surf_end;
	}
	else
	{
		char tmppath[1024];
		while ((entry = readdir(directory_pointer)) != NULL)
		{
			if (entry->d_name[0] == '.' ||
				entry->d_type != DT_REG) 
				continue; 
			
			string tmp = entry->d_name;
			if (tmp.find("jpg") == std::string::npos &&
				tmp.find("bmp") == std::string::npos)
				continue;
			
			//printf("%s\n", entry->d_name);  
			
			memset(tmppath, 0, 1024);
			sprintf(tmppath, "%sdata/%s/scene/%s", getFilePath(), ctx->inputpath2, entry->d_name);
			//printf("%s - %s\n", ctx->inputpath1, tmppath); 
	
			int val = -1;
			int cnt = 0;
			while (1)
			{
				val = surf_detect(ctx->inputpath1, tmppath);
				if (val >= 0 || cnt >= 3)
					break;
				cnt++;
				if (val < 0)
				{
					printf("[%s] - recalculate.--------------------------------------\n",
						ctx->inputpath1);
					usleep(3.0 * 1000000);
					continue;
				}
			} 
			printf("%s - %s = %d\n", ctx->inputpath1, tmppath, val);
			if (val < 0)
			{
				long idr = atol(stridr.c_str());
				surf_map_error[idr] = string(ctx->inputpath1);	
			}
			surf = surf > val ? surf : val;
			fflush(stdout);
		}
	}

surf_end:
	long idr = atol(stridr.c_str());
	printf("[%ld]surf - %d\n", idr, surf);
	surf_map[idr] = surf;
	
	char cmd[1024];
	memset(cmd, 0, 1024);
	sprintf(cmd, "rm -rf %s", ctx->inputpath1);
	//system(cmd);
	
	closedir(directory_pointer);
	
}

extern "C" map<int, int> surf_detect_mutiple(int multiple_cnt, char* scene, char* filename)
{
	char imgpath[1024];
	memset(imgpath, 0, 1024);
	sprintf(imgpath, "%sdata/%s/images/%s", getFilePath(), scene, filename);
	printf("images - %s\n", imgpath);
	
	char cmd[1024];
	memset(cmd, 0, 1024);
	sprintf(cmd, "rm -rf %sdata/%s/images/%s/", getFilePath(), scene, filename, filename);
	printf("cmd - %s\n", cmd);
	
	DIR *directory_pointer;  
	struct dirent *entry;  
	if ((directory_pointer = opendir(imgpath)) == NULL)
	{  
		printf("Error open images[%s]\n", imgpath);  
		return surf_map;  
	}
	else 
	{
		surf_detect_context* ctx;
		int count = 0;
		char tmppath[1024];
		while ((entry = readdir(directory_pointer)) != NULL)
		{
			if (entry->d_name[0] == '.' ||
				entry->d_type != DT_REG) 
				continue;  
			printf("%s\n", entry->d_name);  
			
			string tmp = entry->d_name;
			if (tmp.find(filename) == std::string::npos)
				continue;
			
			memset(tmppath, 0, 1024);
			sprintf(tmppath, "%sdata/%s/images/%s/%s", getFilePath(), scene, filename, entry->d_name);
			printf("%s\n", tmppath); 
			
			if (access(tmppath, F_OK) == 0)
			{
			
			}
			
		    pthread_attr_t attr;
			pthread_attr_init(&attr);
			pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

			pthread_t thr;
			ctx = new surf_detect_context();
			ctx->inputpath1 = (char*)malloc(1024);
			ctx->inputpath2 = (char*)malloc(1024);
			memset(ctx->inputpath1, 0, 1024);
			memset(ctx->inputpath2, 0, 1024);
			memcpy(ctx->inputpath1, tmppath, 1024);
			memcpy(ctx->inputpath2, scene, 1024);
			
			string str = ctx->inputpath1;
			string strname = str.substr(0, str.find_last_of("_"));
			string stridr = str.substr(str.find_last_of("_") + 1, str.find_last_of(".") - 1 - str.find_last_of("_"));
			
			long idr = atol(stridr.c_str());
			surf_map[idr] = -1;
			
			int ret = pthread_create(&thr, &attr, surf_detect_callback, ctx);
			pthread_attr_destroy(&attr);
			
			count++;
			if (count >= multiple_cnt)
			{
				while (1)
				{
					usleep(0.5 * 1000000);
					int is_undef = 0;
					typedef map<int,int>::iterator IT;
					for(IT it = surf_map.begin(); it != surf_map.end(); it++) 
					{
						if (it->second < 0)
							is_undef = 1;
					}
					if (!is_undef)
						break;
				}		
				count = 0;
				printf("reset calculate.\n");
				fflush(stdout);
			}
		}
		while (1)
		{
			usleep(0.5 * 1000000);
			int is_undef = 0;
			typedef map<int,int>::iterator IT;
			for(IT it = surf_map.begin(); it != surf_map.end(); it++) 
			{
				if (it->second < 0)
					is_undef = 1;
			}
			if (!is_undef)
				break;
		}
		
		/// recalculate error
		typedef map<int, string>::iterator IT;
		for(IT it = surf_map_error.begin(); it != surf_map_error.end(); it++) 
		{
			printf("%s\n", it->second.c_str());
			
			string tmp = it->second;
			if (tmp.find(filename) == std::string::npos)
				continue;
			
		    pthread_attr_t attr;
			pthread_attr_init(&attr);
			pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

			pthread_t thr;
			ctx = new surf_detect_context();
			ctx->inputpath1 = (char*)malloc(1024);
			ctx->inputpath2 = (char*)malloc(1024);
			memset(ctx->inputpath1, 0, 1024);
			memset(ctx->inputpath2, 0, 1024);
			memcpy(ctx->inputpath1, tmp.c_str(), 1024);
			memcpy(ctx->inputpath2, scene, 1024);
			
			surf_map[it->first] = -1;
			
			int ret = pthread_create(&thr, &attr, surf_detect_callback, ctx);
			pthread_attr_destroy(&attr);
			
			count++;
			if (count >= multiple_cnt)
			{
				while (1)
				{
					usleep(0.5 * 1000000);
					int is_undef = 0;
					typedef map<int,int>::iterator IT;
					for(IT it = surf_map.begin(); it != surf_map.end(); it++) 
					{
						if (it->second < 0)
							is_undef = 1;
					}
					if (!is_undef)
						break;
				}		
				count = 0;
				printf("reset calculate.\n");
				fflush(stdout);
			}
		}
		while (1)
		{
			usleep(0.5 * 1000000);
			int is_undef = 0;
			typedef map<int,int>::iterator IT;
			for(IT it = surf_map.begin(); it != surf_map.end(); it++) 
			{
				if (it->second < 0)
					is_undef = 1;
			}
			if (!is_undef)
				break;
		}
		
		printf("all calculated.\n");
		return surf_map;
	}
	
	//system(cmd);
	closedir(directory_pointer);
}



extern "C" double face_match(char* imgpath)
{
	double confidence = 0.0;
	char cmd[1024];
	memset(cmd, 0, 1024);
	
	// 人脸识别
	sprintf(cmd, "python %sface_match.py %s | tail -n 1", getFilePath(), imgpath);

	//printf("%s\n", cmd);
	
	FILE* fp;
	fp = popen(cmd, "r");

	if (fp == NULL)
	{
		printf("check proc[%d] - popen is failed.----------------------------------------------------------------------------\n");
		fflush(stdout);

		_exit(-1);
		return 0;
	}

	char tmp[128];
	memset(tmp, 0, 128);
	while (NULL != fgets(tmp, 128, fp))
	{
		confidence = atof(tmp);
		memset(tmp, 0, 128);
	}

	pclose(fp);

	return confidence;
}


extern "C" int imgmatch_surf(char* path1, char* path2, char* path3, Rect rect1, Rect rect2, int size)
{
  int scale_height = size;
  CvSize sz;
  
  IplImage *img1 = NULL;
  IplImage *desc1 = NULL;
  pthread_mutex_lock(&g_mutex_surf);
  //img1 = cvLoadImage(path1);
  Mat mat1 = imread(path1, CV_LOAD_IMAGE_COLOR);
  if (mat1.rows <= 0 || mat1.cols <= 0)
  {
	  printf("[%s]mat1 open error.\n", path1);
	  pthread_mutex_unlock(&g_mutex_surf);
	  return -1;
  }
  Mat mat1_sc = mat1;
  if (rect1.width > 0 && rect1.height > 0)
    mat1_sc = mat1(rect1);
  imwrite(path3, mat1_sc);
  *img1 = IplImage(mat1_sc);
  pthread_mutex_unlock(&g_mutex_surf);
  if (img1 == NULL || img1->width <= 0 || img1->height <= 0)
  {
	  if (img1)
		cvReleaseImage(&img1);
	  printf("img1 open error.\n");
	  return -1;
  }
  double scale = (double)scale_height / (double)img1->height;
  sz.width = img1->width * scale;
  sz.height = img1->height * scale;
  printf("img1 - [%d:%d]\n", sz.width, sz.height);
  desc1 = cvCreateImage(sz, img1->depth, img1->nChannels);
  cvResize(img1, desc1, CV_INTER_CUBIC);
  //cvReleaseImage(&img1);
  IpVec ipts1;
  surfDetDes(desc1,ipts1,false,4,4,2,0.0001f);
  
  IpPairVec matches;
  typedef map<char*, IpVec>::iterator IT;
  if (surf_vec_map.find(path2) != surf_vec_map.end())
  {
	getMatches(ipts1, surf_vec_map[path2], matches);
	printf("Matches: %d\n", matches.size());
	cvReleaseImage(&desc1);
	return matches.size();
  }
  
  IplImage *img2 = NULL;
  IplImage *desc2 = NULL;
  pthread_mutex_lock(&g_mutex_surf);
  //img2 = cvLoadImage(path2);
  Mat mat2 = imread(path2, CV_LOAD_IMAGE_COLOR);
  if (mat2.rows <= 0 || mat2.cols <= 0)
  {
	  printf("[%s]mat2 open error.\n", path2);
	  pthread_mutex_unlock(&g_mutex_surf);
	  return -1;
  }
  Mat mat2_sc = mat2;
  if (rect2.width > 0 && rect2.height > 0)
    mat2_sc = mat2(rect2);
  *img2 = IplImage(mat2_sc);
  pthread_mutex_unlock(&g_mutex_surf);
  if (img2 == NULL || img2->width <= 0 || img2->height <= 0)
  {
	  if (img2)
		cvReleaseImage(&img2);
	  printf("img2 open error.\n");
	  return -1;
  }
  scale = (double)scale_height / (double)img2->height;
  sz.width = img2->width * scale;
  sz.height = img2->height * scale;
  printf("img2 - [%d:%d]\n", sz.width, sz.height);
  desc2 = cvCreateImage(sz, img2->depth, img2->nChannels);
  cvResize(img2, desc2, CV_INTER_CUBIC);                  
  //cvReleaseImage(&img2);
  IpVec ipts2;
  surfDetDes(desc2,ipts2,false,4,4,2,0.0001f);
  
  surf_vec_map[path2] = ipts2;
  getMatches(ipts1,ipts2,matches);
  cvReleaseImage(&desc1);
  cvReleaseImage(&desc2);
  
/*
  for (unsigned int i = 0; i < matches.size(); ++i)
  {
    drawPoint(img1,matches[i].first);
    drawPoint(img2,matches[i].second);
  
    const int & w = img1->width;
    cvLine(img1,cvPoint(matches[i].first.x,matches[i].first.y),cvPoint(matches[i].second.x+w,matches[i].second.y), cvScalar(255,255,255),1);
    cvLine(img2,cvPoint(matches[i].first.x-w,matches[i].first.y),cvPoint(matches[i].second.x,matches[i].second.y), cvScalar(255,255,255),1);
  }
*/
  printf("Matches: %d\n", matches.size());
/*
  cvNamedWindow("1", CV_WINDOW_AUTOSIZE );
  cvNamedWindow("2", CV_WINDOW_AUTOSIZE );
  cvShowImage("1", img1);
  cvShowImage("2",img2);
  cvWaitKey(0);
*/
  return matches.size();
}
