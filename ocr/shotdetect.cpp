#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <dirent.h>  
#include <opencv2/opencv.hpp>
#ifdef linux
#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <signal.h>
#include <getopt.h> 
#endif
#include "types.h"
#include "shotdetect_alg.h"
#include "fingerprint_alg.h"
#include "log5cxx.h"


#define _PHASH_FACEDETECT_			26

#define _MAX_MAPSURF_IDR_			30
#define _MAX_RESULT_CNT_			35
#define _TIME_OFFSET_				1

#define _SSIM_SCALE_DIM_			128
#define _FACEMATCH_CALC_CNT_		10
#define _FACEMATCH_CALC_INTERVAL_	30.0

#define _KEYPOINT_MIN_INTERVAL_		2.0

#define _DB_THRESH_DEFAULT_			-45.0
#define _SSIM_SIMULAR_DEFAULT_		0.7
#define _PROG_DURATION_DEFAULT_		24.0
#define _FACE_CNT_DEFAULT_			0
#define _SURF_MATCH_DEFAULT_		5

/// 字幕标题图标位置及尺寸
int _TITLE_SCENE_X_1_ = 0;
int _TITLE_SCENE_Y_1_ = 0;
int _TITLE_SCENE_W_1_ = 100;
int _TITLE_SCENE_H_1_ = 50;

int _TITLE_SCENE_X_2_ = 0;
int _TITLE_SCENE_Y_2_ = 0;
int _TITLE_SCENE_W_2_ = 100;
int _TITLE_SCENE_H_2_ = 50;

int _TITLE_SCENE_X_3_ = 0;
int _TITLE_SCENE_Y_3_ = 0;
int _TITLE_SCENE_W_3_ = 100;
int _TITLE_SCENE_H_3_ = 50;

int _TITLE_SCENE_X_4_ = 0;
int _TITLE_SCENE_Y_4_ = 0;
int _TITLE_SCENE_W_4_ = 100;
int _TITLE_SCENE_H_4_ = 50;

int g_max_mapsurf_idr = _MAX_MAPSURF_IDR_;
int g_max_result_cnt = _MAX_RESULT_CNT_;
int g_time_offset = _TIME_OFFSET_;
float g_key_interval = _KEYPOINT_MIN_INTERVAL_;
string g_program_type = "default";
float g_hist_thresh = 0.3;
int g_surf_match = 5;
float g_ssim_simular = 0.7;

using namespace std;
using namespace cv;

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

extern "C" int facedetect(Mat &mat, int *x, int *y);
extern "C" int audio_decode(const char *filename, float *dbv, unsigned int length, unsigned int fps);
extern "C" int audio_decode2(const char *filename, float *dbv, unsigned int length, unsigned int fps);

static vector<fp_t> phash_fplist;
static vector<fp_t> phash_scene_fplist;
static vector<vector<float> > hist_scene;  
static vector<MatND> hist_hsv_scene;  
static vector<Mat> mat_scene;  
static vector<Mat> mat_title;  
static vector<fp_t> phash_title_fplist;  

extern "C" pthread_mutex_t g_mutex_surf;

typedef struct MatchImg
{
	string strMatchImgUrl;
	int X;
	int Y;
	int W;
	int H;
} MatchImg;
static vector<MatchImg*> g_vec_vheads;
static vector<MatchImg*> g_vec_vtails;
static vector<Mat> mat_vheads;
static vector<Mat> mat_vtails;

vector<string> g_vec_scenes;


/// 字符串分割
static vector<string> split(string& str, const char* c)
{
    char *cstr, *p;
    vector<string> res;
    cstr = new char[str.size()+1];
    strcpy(cstr,str.c_str());
    p = strtok(cstr,c);
    while(p!=NULL)
    {
        res.push_back(p);
        p = strtok(NULL,c);
    }
    return res;
}

// 一维数组均值滤波，目前没有用到
static void mean_filter1d(float *vector, float *output, unsigned int len, unsigned int kernel_size)
{
    if (kernel_size%2 == 0)
        kernel_size += 1;
    
    double v;
    for (int nn=kernel_size/2; nn<len-kernel_size/2; nn++) {
        v = 0;
        for (int k=-int(kernel_size)/2; k<=int(kernel_size)/2; k++)
            v += vector[nn+k];
        output[nn] = v/double(kernel_size);
    }
    for (int nn=0; nn<kernel_size/2; nn++)
        output[nn] = vector[nn];
    for (int nn=len-kernel_size/2; nn<len; nn++)
        output[nn] = vector[nn];
}

typedef struct {
    int frame_idx;
    int type;         //0为无效，1为转场，2为一个人物，3为多个人物，100为静音
	int diff;
	int faces_cnt;
	int surf_point_cnt;
}sp_t;

// 获取相邻的最近的静音
static int get_near_mute(vector<sp_t> &mute_points, int pos)
{
    int len = mute_points.size();
    if (len == 0)
        return 0;
	
    for (int i=1; i<len; i++) {
		if (pos < mute_points[i].frame_idx){
            int v1 = mute_points[i].frame_idx - pos;
            int v2 = pos - mute_points[i-1].frame_idx;
            return (v1>v2)?v2:v1;
        }
    }
}

// 判断是否存在与待匹配图像帧距离小于指定范围的静音帧
static int check_near_mute(vector<sp_t> &mute_points, int pos, int range)
{
    int len = mute_points.size();
    if (len == 0)
        return 0;
	
	int pos_diff = 0;
	int pos1 = -1, pos2 = -1;
    for (int i=0; i<len; i++) 
	{
		pos_diff = pos - mute_points[i].frame_idx;
		if (pos1 < 0 && fabs(pos_diff) < range)
		{
			pos1 = i;
			continue;
		}
		if (pos1 >= 0 && pos2 < 0 && pos_diff < 0 && fabs(pos_diff) < range)
		{
			pos2 = i;
		}
		pos_diff = pos2 - pos1;
		if (pos1 >= 0 && pos2 >= 0)
			return 1;
    }
	return 0;
}


static void video_shotdetect(
	const char *path, 		// 输入视频文件路径
	const char *output, 	// 输出结果文件路径
	const char *program,	// 本次识别的任务id，如命令中五次参数，则默认使用当前时间刻度值
	const char *pklpath, 	// openface人脸识别?，模型路径
	bool with_face, 		// 是否启用人脸检测
	int face_cnt, 			// 人脸检测时的人脸个数阈值
	bool with_facematch, 	// 是否启用人脸识别，默认使用seetaface
	int surf_match, 		// 场景surf特征点匹配后，下限处最大并列长度
	bool with_voice, 		// 是否启用声音检测
	float db_thresh, 		// 声音检测后的静音分贝阈值
	float prog_duration, 	// 当前节目的时长，分钟
	float ssim_simular,		// 起始/结束时间点进行场景ssim匹配时的阈值
	bool with_less)			// 是否省略打印日志
{
	/// strFileName保存待检测视频的文件名（含扩展名）
	string strFileName = path;
	strFileName = strFileName.substr(
	strFileName.find_last_of("/") + 1,
	strFileName.length() - strFileName.find_last_of("/") - 1);
	
	
	string strProgram = "";
	
	
	/// 删除当前节目ID对应的缓存数据
	char cmd[1024];
	memset(cmd, 0, 1024);
	sprintf(cmd, "rm -rf %sdata/%s", getFilePath(), program);
	system(cmd);
	
	
	/// 检测缓存目录是否存在
	char tmppath[1024];
	memset(tmppath, 0, 1024);
	
	sprintf(tmppath, "%sdata",
		getFilePath());
	if (opendir(tmppath) == NULL)
	{
		printf("folder - data is not exists.\n");
		return;
	}
	
	
	/// 创建以当前节目ID命名的缓存目录
	sprintf(tmppath, "%sdata/%s", getFilePath(), program);
	memset(cmd, 0, 1024);
	if (opendir(tmppath) == NULL)
	{
		sprintf(cmd, "mkdir %s %s/scene %s/title %s/images", 
			tmppath, tmppath, tmppath, tmppath);
		system(cmd);
	}
	else
	{
		printf("[%s] is exists.\n", tmppath);
		int i = 1;
		/// 若已存在指定命名的目录，则添加一个后缀，重新创建
		while (1)
		{
			sprintf(tmppath, "%s_%d", i);
			if (opendir(tmppath) == NULL)
			{
				sprintf(cmd, "mkdir %s %s/scene %s/title %s/images", 
					tmppath, tmppath, tmppath);
				system(cmd);
				break;
			}
			i++;
		}
	}
	strProgram = tmppath;
	strProgram = strProgram.substr(
		strProgram.find_last_of("/") + 1, 
		strProgram.length() - strProgram.find_last_of("/") - 1);
	printf("program - %s\n", strProgram.c_str());
	
	
	/// 若用于起始/结束时间点检测的场景图片路
	/// 径缓存队列中有数据，则下载这些图片数据
	/// 至本地，保存至"data/[节目ID]/title"，
	/// 文件名称保持原URL中的文件名
	if (g_vec_vheads.size() > 0)
	{
		im_t *img1;
		Mat mat, mat_sc, rgb;
		printf("vhead size - %d\n", g_vec_vheads.size());
		
		/// 处理用于起始时间点检测的场景图片
		for (int i = 0; i < g_vec_vheads.size(); i++)
		{
			/// 获取URL
			if (g_vec_vheads[i]->strMatchImgUrl == "")
				continue;
			printf("vhead url[%d] - %s\n", i, g_vec_vheads[i]->strMatchImgUrl.c_str());
			printf("[%s]x - %d y - %d w - %d h - %d\n",
				g_vec_vheads[i]->strMatchImgUrl.c_str(), g_vec_vheads[i]->X, g_vec_vheads[i]->Y, g_vec_vheads[i]->W, g_vec_vheads[i]->H);

			/// 获取文件名
			string strImgPath = g_vec_vheads[i]->strMatchImgUrl;
			strImgPath = 
				strImgPath.substr(strImgPath.find_last_of("/") + 1, 
				strImgPath.length() - 1 - strImgPath.find_last_of("/"));
			
			/// 下载保存至本地
			memset(cmd, 0, 1024);
			memset(tmppath, 0, 1024);
			sprintf(tmppath, "%sdata/%s/title/%s",
				getFilePath(), strProgram.c_str(), strImgPath.c_str());
			sprintf(cmd, "wget -O %s %s", 
				tmppath, g_vec_vheads[i]->strMatchImgUrl.c_str());
			printf("cmd - %s\n", cmd);
			system(cmd);
			
			/// 打开下载后的本地图片文件为mat对象，
			/// 拉伸图片至指定尺寸，转换为RGB颜色模式，
			/// 保存至起始时间点检测场景图片队列
			mat = imread(tmppath, CV_LOAD_IMAGE_COLOR);
			resize(mat, mat_sc, Size(_SSIM_SCALE_DIM_, _SSIM_SCALE_DIM_));
			cvtColor(mat_sc, rgb, CV_BGR2RGB);
			mat_vheads.push_back(rgb.clone());
			printf("[%s]w - %d h - %d\n", tmppath, mat.cols, mat.rows);
		}
		printf("vtail size - %d\n", g_vec_vtails.size());
		
		/// 处理用于结束时间点检测的场景图片
		for (int i = 0; i < g_vec_vtails.size(); i++)
		{
			/// 获取URL
			if (g_vec_vtails[i]->strMatchImgUrl == "")
				continue;
			printf("vtail url - %s\n", g_vec_vtails[i]->strMatchImgUrl.c_str());  
			printf("[%s]x - %d y - %d w - %d h - %d\n",
				g_vec_vtails[i]->strMatchImgUrl.c_str(), g_vec_vtails[i]->X, g_vec_vtails[i]->Y, g_vec_vtails[i]->W, g_vec_vtails[i]->H);
			
			/// 获取文件名
			string strImgPath = g_vec_vtails[i]->strMatchImgUrl;
			strImgPath = 
				strImgPath.substr(strImgPath.find_last_of("/") + 1, 
				strImgPath.length() - 1 - strImgPath.find_last_of("/"));
			
			/// 下载保存至本地
			memset(cmd, 0, 1024);
			memset(tmppath, 0, 1024);
			sprintf(tmppath, "%sdata/%s/title/%s",
				getFilePath(), strProgram.c_str(), strImgPath.c_str());
			sprintf(cmd, "wget -O %s %s", 
				tmppath, g_vec_vtails[i]->strMatchImgUrl.c_str());
			printf("cmd - %s\n", cmd);
			system(cmd);
			
			/// 打开下载后的本地图片文件为mat对象，
			/// 拉伸图片至指定尺寸，转换为RGB颜色模式，
			/// 保存至结束时间点检测场景图片队列
			mat = imread(tmppath, CV_LOAD_IMAGE_COLOR);
			resize(mat, mat_sc, Size(_SSIM_SCALE_DIM_, _SSIM_SCALE_DIM_));
			cvtColor(mat_sc, rgb, CV_BGR2RGB);
			mat_vtails.push_back(rgb.clone());
			printf("[%s]w - %d h - %d\n", tmppath, mat.cols, mat.rows);
		}
	}
	
	
	/// 已废弃，本地直接保存用于起始/结束时间点检测的场景图片
	/*
	if (g_vec_vheads.size() > 0)
	{
		im_t *img1;
		Mat mat, mat_sc, rgb;
		printf("vhead size - %d\n", g_vec_vheads.size());
		for (int i = 0; i < g_vec_vheads.size(); i++)
		{
			if (g_vec_vheads[i]->strMatchImgUrl == "")
				continue;
			printf("vhead url[%d] - %s\n", i, g_vec_vheads[i]->strMatchImgUrl.c_str());
			printf("[%s]x - %d y - %d w - %d h - %d\n",
				g_vec_vheads[i]->strMatchImgUrl.c_str(), g_vec_vheads[i]->X, g_vec_vheads[i]->Y, g_vec_vheads[i]->W, g_vec_vheads[i]->H);

			mat = imread(g_vec_vheads[i]->strMatchImgUrl.c_str(), CV_LOAD_IMAGE_COLOR);

			resize(mat, mat_sc, Size(_SSIM_SCALE_DIM_, _SSIM_SCALE_DIM_));
			cvtColor(mat_sc, rgb, CV_BGR2RGB);
			mat_vheads.push_back(rgb.clone());
			printf("[%s]w - %d h - %d\n", g_vec_vheads[i]->strMatchImgUrl.c_str(), mat.cols, mat.rows);
		}
		printf("vtail size - %d\n", g_vec_vtails.size());
		for (int i = 0; i < g_vec_vtails.size(); i++)
		{
			if (g_vec_vtails[i]->strMatchImgUrl == "")
				continue;
			printf("vtail url - %s\n", g_vec_vtails[i]->strMatchImgUrl.c_str());  
			printf("[%s]x - %d y - %d w - %d h - %d\n",
				g_vec_vtails[i]->strMatchImgUrl.c_str(), g_vec_vtails[i]->X, g_vec_vtails[i]->Y, g_vec_vtails[i]->W, g_vec_vtails[i]->H);
			
			mat = imread(g_vec_vtails[i]->strMatchImgUrl.c_str(), CV_LOAD_IMAGE_COLOR);
			
			resize(mat, mat_sc, Size(_SSIM_SCALE_DIM_, _SSIM_SCALE_DIM_));
			cvtColor(mat_sc, rgb, CV_BGR2RGB);
			mat_vtails.push_back(rgb.clone());
			printf("[%s]w - %d h - %d\n", g_vec_vtails[i]->strMatchImgUrl.c_str(), mat.cols, mat.rows);
		}
	}
	*/
	

	/// 若用于场景surf检测的场景图片路径缓
	/// 存队列中有数据，则下载这些图片数据
	/// 至本地，保存至"data/[节目ID]/scene"，
	/// 文件名称保持原URL中的文件名
	if (g_vec_scenes.size() > 0)
	{
		printf("scene size - %d\n", g_vec_scenes.size());
		for (int i = 0; i < g_vec_scenes.size(); i++)
		{
			/// 获取URL
			if (g_vec_scenes[i] == "")
				continue;

			/// 获取文件名
			string strImgPath = g_vec_scenes[i];
			strImgPath = 
				strImgPath.substr(strImgPath.find_last_of("/") + 1, 
				strImgPath.length() - 1 - strImgPath.find_last_of("/"));
			
			/// 下载保存至本地
			memset(cmd, 0, 1024);
			memset(tmppath, 0, 1024);
			sprintf(tmppath, "%sdata/%s/scene/%s",
				getFilePath(), strProgram.c_str(), strImgPath.c_str());
			sprintf(cmd, "wget -O %s %s", 
				tmppath, g_vec_scenes[i].c_str());
			printf("cmd - %s\n", cmd);
			system(cmd);
			printf("scene url[%d] - %s\n", i, g_vec_scenes[i].c_str());
		}
	}
	
	
	/// 检测是否存在用于人脸识别的python
	/// 程序脚本（微软版）
	int is_exists_face_match = 0;
	memset(tmppath, 0, 1024);
	sprintf(tmppath, "%sface_match.py",
		getFilePath());
	if (access(tmppath, F_OK) == 0)
	{
		is_exists_face_match = 1;
		printf("[%s] is exists.\n", tmppath);
	}
	else
	{
		printf("[%s] is not exists.\n", tmppath);
	}
	
	
	/// 下载视频文件保存至本地
	/*
	memset(cmd, 0, 1024);
	memset(tmppath, 0, 1024);
	sprintf(tmppath, "%sdata/%s/%s",
		getFilePath(), strProgram.c_str(), strFileName.c_str());
	sprintf(cmd, "wget -t 5 -T 30 -O %s %s", 
		tmppath, path);
	printf("cmd - %s\n", cmd);
	system(cmd);
	*/
	
	
	g_surf_match = surf_match;
	g_ssim_simular = ssim_simular;
	
	/// 读取字幕及图标坐标位置信息
	memset(tmppath, 0, 1024);
	sprintf(tmppath, "%sdata/%s/setting.txt",
		getFilePath(), (char*)g_program_type.c_str());
	if (access(tmppath, F_OK) == 0)
	{
		char line[1024];
		FILE *fp = fopen(tmppath, "r");  
		if (fp == NULL)  
		{  
			printf("can not load file!");  
			return;  
		}  
		while (!feof(fp))  
		{
			fgets(line, 1024, fp);
			string str_line = line;
			printf("getline - %s\n", line);
			if (str_line.find("TITLE_POS1") != string::npos)
			{
				string str_val = str_line.substr(strlen("TITLE_POS1") + 1, strlen(str_line.c_str()) - strlen("TITLE_POS1") - 1);
				vector<string> _vec_vals = split(str_val, ",");
				if (_vec_vals.size() >= 4)
				{
					_TITLE_SCENE_X_1_ = atoi(_vec_vals[0].c_str());
					_TITLE_SCENE_Y_1_ = atoi(_vec_vals[1].c_str());
					_TITLE_SCENE_W_1_ = atoi(_vec_vals[2].c_str());
					_TITLE_SCENE_H_1_ = atoi(_vec_vals[3].c_str());
					printf("titleinfo1 - %d,%d,%d,%d\n",
						_TITLE_SCENE_X_1_,
						_TITLE_SCENE_Y_1_,
						_TITLE_SCENE_W_1_,
						_TITLE_SCENE_H_1_);
				}
			}
			if (str_line.find("TITLE_POS2") != string::npos)
			{
				string str_val = str_line.substr(strlen("TITLE_POS2") + 1, strlen(str_line.c_str()) - strlen("TITLE_POS2") - 1);
				vector<string> _vec_vals = split(str_val, ",");
				if (_vec_vals.size() >= 4)
				{
					_TITLE_SCENE_X_2_ = atoi(_vec_vals[0].c_str());
					_TITLE_SCENE_Y_2_ = atoi(_vec_vals[1].c_str());
					_TITLE_SCENE_W_2_ = atoi(_vec_vals[2].c_str());
					_TITLE_SCENE_H_2_ = atoi(_vec_vals[3].c_str());
					printf("titleinfo2 - %d,%d,%d,%d\n",
						_TITLE_SCENE_X_2_,
						_TITLE_SCENE_Y_2_,
						_TITLE_SCENE_W_2_,
						_TITLE_SCENE_H_2_);
				}
			}
			if (str_line.find("TITLE_POS3") != string::npos)
			{
				string str_val = str_line.substr(strlen("TITLE_POS3") + 1, strlen(str_line.c_str()) - strlen("TITLE_POS3") - 1);
				vector<string> _vec_vals = split(str_val, ",");
				if (_vec_vals.size() >= 4)
				{
					_TITLE_SCENE_X_3_ = atoi(_vec_vals[0].c_str());
					_TITLE_SCENE_Y_3_ = atoi(_vec_vals[1].c_str());
					_TITLE_SCENE_W_3_ = atoi(_vec_vals[2].c_str());
					_TITLE_SCENE_H_3_ = atoi(_vec_vals[3].c_str());
					printf("titleinfo3 - %d,%d,%d,%d\n",
						_TITLE_SCENE_X_3_,
						_TITLE_SCENE_Y_3_,
						_TITLE_SCENE_W_3_,
						_TITLE_SCENE_H_3_);
				}
			}
			if (str_line.find("TITLE_POS4") != string::npos)
			{
				string str_val = str_line.substr(strlen("TITLE_POS4") + 1, strlen(str_line.c_str()) - strlen("TITLE_POS4") - 1);
				vector<string> _vec_vals = split(str_val, ",");
				if (_vec_vals.size() >= 4)
				{
					_TITLE_SCENE_X_4_ = atoi(_vec_vals[0].c_str());
					_TITLE_SCENE_Y_4_ = atoi(_vec_vals[1].c_str());
					_TITLE_SCENE_W_4_ = atoi(_vec_vals[2].c_str());
					_TITLE_SCENE_H_4_ = atoi(_vec_vals[3].c_str());
					printf("titleinfo4 - %d,%d,%d,%d\n",
						_TITLE_SCENE_X_4_,
						_TITLE_SCENE_Y_4_,
						_TITLE_SCENE_W_4_,
						_TITLE_SCENE_H_4_);
				}
			}
			if (str_line.find("HIST_THRESH") != string::npos)
			{
				string str_val = str_line.substr(strlen("HIST_THRESH") + 1, strlen(str_line.c_str()) - strlen("HIST_THRESH") - 1);
				double _val = atof(str_val.c_str());
				printf("val[%s] - %.4f\n", (char*)str_val.c_str(), _val);
				if (_val > 0.0 && _val < 1.0)
				{
					g_hist_thresh = _val;
				}
				printf("HIST_THRESH - %.4f\n", g_hist_thresh);
			}
			if (str_line.find("SURF_MATCH") != string::npos)
			{
				string str_val = str_line.substr(strlen("SURF_MATCH") + 1, strlen(str_line.c_str()) - strlen("SURF_MATCH") - 1);
				double _val = atoi(str_val.c_str());
				printf("val[%s] - %d\n", (char*)str_val.c_str(), _val);
				if (_val > 0 && _val <= 500)
				{
					g_surf_match = _val;
				}
				printf("SURF_MATCH - %d\n", g_surf_match);
			}
			if (str_line.find("SSIM_SIMULAR") != string::npos)
			{
				string str_val = str_line.substr(strlen("SSIM_SIMULAR") + 1, strlen(str_line.c_str()) - strlen("SSIM_SIMULAR") - 1);
				double _val = atof(str_val.c_str());
				printf("val[%s] - %.4f\n", (char*)str_val.c_str(), _val);
				if (_val > 0.0 && _val < 1.0)
				{
					g_ssim_simular = _val;
				}
				printf("SSIM_SIMULAR - %.4f\n", g_ssim_simular);
			}
		}  

		fclose(fp);
	}
	
	
	/// 互斥量初始化，用于surf场景匹配
	pthread_mutex_init(&g_mutex_surf,NULL);
	
	/// 计算每一步的时间消耗
    clock_t s, e, e2; 
    char msg[4096];
    bool silence_mode = false;

	/// opencv视频处理对象
    VideoCapture capture; 
	memset(cmd, 0, 1024);
	//memset(tmppath, 0, 1024);
	//sprintf(tmppath, "%sdata/%s/%s",
	//	getFilePath(), strProgram.c_str(), strFileName.c_str());		
    capture.open(path);	// 打开待检测的视频
	/// 视频打开失败检测
    if(!capture.isOpened()) {
        if (strlen(path) < 3800)
            sprintf(msg, "Read video failed!(%s)\n", path);
        else
            sprintf(msg, "Read video failed!\n");
        LOG5CXX_FATAL(msg, 121);
    }
    unsigned int numberframes = capture.get(CV_CAP_PROP_FRAME_COUNT);	// 获取视频总帧数
    unsigned int fps = capture.get(CV_CAP_PROP_FPS);					// 获取视频帧率
    unsigned int hist_dim = 64, resize_dim = 2;     // 直方图bin数量，resize_dim是图像一个方向维度的分块数量（一幅图像计算resize_dim*resize_dim个直方图）
    const unsigned int hist_offset = hist_dim*resize_dim*resize_dim;	// 按resize_dim分块数，hist_offset为一帧图像的直方图数据个数
    const float thresh = g_hist_thresh;	// 转场识别时的直方图变化阈值

	/// 初始化保存直方图变化率的数组
    float *hist_err = (float*)calloc(numberframes*sizeof(float),1);
    if (hist_err == NULL) {
        sprintf(msg, "Memory malloc or memcpy failed, operate memory size is [%d]=[%d MB].", numberframes*sizeof(float)/1024/1024);
        LOG5CXX_FATAL(msg, 210);
    }
	
    vector<sp_t> points;	// 关键帧检测后的节点队列
	map<int, double> confidence_map;	// 人脸识别（openface方式）时，保存每个关键帧索引对应的可信度
	map<int, int> surf_map;	// surf场景匹配时，保存每个关键帧索引对应的匹配上的特征点个数
	int time_start = 0, time_end = 0;	// 起始/结束时间点检测后的起始/结束时间点
	
    s = clock();	// 开始执行计时

    // 第1步：首先进行起始/结束时间点检测，然后计算每个帧的分块灰度直方图
    percent(0);	// 标记执行进度0%
    {
		/// 创建用于保存每个帧的分块灰度直方图数据的数组
        float *hist = (float*)calloc((numberframes+1)*hist_dim*resize_dim*resize_dim*sizeof(float),1);
        if (hist == NULL) {
            sprintf(msg, "Memory malloc or memcpy failed, operate memory size is [%d]=[%d MB].", numberframes*hist_dim*resize_dim*resize_dim*sizeof(float)/1024/1024);
            LOG5CXX_FATAL(msg, 210);
        }
		
        im_t img;
        Mat frame, gray, mat_logo, mat64, rgb, rgb_title, start_title, end_title;
				
        int nn = 0;	// 当前遍历的帧的索引
		double simular_start = 0.0, simular_end = 0.0;	// 保存每帧与起始/结束场景图片进行ssim匹配后的最大值
		int diff_start = 0, diff_end = 0;
		fp_t fp_mat;
		
		/// 当存在起始/结束时间点检测的场景图片时，开始遍历视频帧进行检测
		while((mat_vheads.size() > 0 || mat_vtails.size() > 0) && capture.read(frame)) 
		{
			/// 每25帧选一帧，降低检测样本数量
			/// 待检测视频均为转码后视频，帧率25fps
			if (nn % 25 != 0)
			{
				nn++;
				percent((float(nn)/float(numberframes)*45) / 2);
				continue;
			}
			
			/// 起始时间点匹配后的最大ssim值，若大于指定阈值，
			/// 则判定起始时间点匹配成功，直接跳过35000帧（以
			/// 25fps计算，为1400秒），再匹配结束时间点，减少
			/// 检测样本数量
			if (simular_start >= g_ssim_simular && nn <= time_start + 35000)
			{
				nn++;
				percent((float(nn)/float(numberframes)*45) / 2);
				continue;
			}
			
			/// 若未检测的剩余视频时间，小于指定的节目时长，
			/// 且此时仍未检测到起始时间点，则不再继续检测，
			/// 结束此检测步骤，节省检测总时长
			if (simular_start < g_ssim_simular && numberframes - nn < prog_duration * 60.0 * fps)
				break;
			
			/// 若起始/结束时间点匹配后的最大ssim值均大于指定阈值，
			/// 则表示起始/结束时间点检测成功，结束此检测步骤
			if (simular_start >= g_ssim_simular && simular_end >= g_ssim_simular)
				break;
			
			/// 将遍历获取的视频帧，转换为RGB颜色模式
			cvtColor(frame, rgb, CV_BGR2RGB);

			/// 起始时间点匹配
			for (int i = 0; i < mat_vheads.size(); i++)
			{
				if (i > g_vec_vheads.size() - 1 && g_vec_vheads[i]->strMatchImgUrl == "")
					continue;
				/*
				printf("[%s]x - %d y - %d w - %d h - %d\n",
					g_vec_vheads[i]->strMatchImgUrl.c_str(), g_vec_vheads[i]->X, g_vec_vheads[i]->Y, g_vec_vheads[i]->W, g_vec_vheads[i]->H);
				*/
				
				/// 起始时间点场景图片在视频中匹配的位置
				Rect rect_start = 
					Rect(g_vec_vheads[i]->X, g_vec_vheads[i]->Y, g_vec_vheads[i]->W, g_vec_vheads[i]->H);
				rgb_title = rgb(rect_start);	// 截取视频帧的指定区域
				resize(rgb_title, start_title, Size(_SSIM_SCALE_DIM_, _SSIM_SCALE_DIM_));
				
				/// 计算视频帧和起始时间点场景图片的ssim值
				double simular_test = ssim(start_title, mat_vheads[i]);
				printf("[%d]start ssim simular: - %f\n", nn, simular_test);
				
				/// 保存当前ssim值和之前计算的ssim值的较大值
				if (simular_start <= simular_test)
				{
					simular_start = simular_test;
					time_start = nn;
					printf("start time - %d -----------------------------------------\n", time_start);
				}
				
				//simular_start = simular_start > simular_test ? simular_start : simular_test;
			}
			
			/// 已废弃，以下采用计算phash值进行起始时间点匹配
			/*
			img.rows = start_title.rows;
			img.cols = start_title.cols;
			img.channels = start_title.channels();
			img.depth = start_title.depth();
			img.data = start_title.data;
			calc_fingerprint_phash(&fp_mat, &img);
			for (int i = 0; i < phash_title_fplist.size(); i++)
			{
				int diff1 = hanming_distance(&fp_mat, &phash_title_fplist[i]);
				printf("[%d]diff_start - %d\n", nn, diff1);
				if (diff_start <= diff1)
				{
					diff_start = diff1;
					//time_start = nn;
					printf("start time - %d -----------------------------------------\n", time_start);
				}
				
				//diff_start = diff_start < diff1 ? diff_start : diff1;
			}
			*/
			
			/// 结束时间点匹配
			for (int i = 0; i < mat_vtails.size(); i++)
			{
				if (i > g_vec_vtails.size() - 1 && g_vec_vtails[i]->strMatchImgUrl == "")
					continue;
				/*
				printf("[%s]x - %d y - %d w - %d h - %d\n",
					g_vec_vtails[i]->strMatchImgUrl.c_str(), g_vec_vtails[i]->X, g_vec_vtails[i]->Y, g_vec_vtails[i]->W, g_vec_vtails[i]->H);
				*/
				
				/// 结束时间点场景图片在视频中匹配的位置
				Rect rect_end = 
					Rect(g_vec_vtails[i]->X, g_vec_vtails[i]->Y, g_vec_vtails[i]->W, g_vec_vtails[i]->H);
				rgb_title = rgb(rect_end);	// 截取视频帧的指定区域
				resize(rgb_title, end_title, Size(_SSIM_SCALE_DIM_, _SSIM_SCALE_DIM_));
				
				/// 计算视频帧和结束时间点场景图片的ssim值
				double simular_test = ssim(end_title, mat_vtails[i]);
				printf("[%d]end ssim simular: - %f\n", nn, simular_test);
				
				/// 保存当前ssim值和之前计算的ssim值的较大值
				if (simular_end <= simular_test)
				{
					simular_end = simular_test;
					time_end = nn;
					printf("end time - %d -----------------------------------------\n", time_end);
				}
				
				//simular_end = simular_end > simular_test ? simular_end : simular_test;
			}
			
			/// 已废弃，以下采用计算phash值进行起始时间点匹配
			/*
			img.rows = end_title.rows;
			img.cols = end_title.cols;
			img.channels = end_title.channels();
			img.depth = end_title.depth();
			img.data = end_title.data;
			calc_fingerprint_phash(&fp_mat, &img);
			for (int i = 0; i < phash_title_fplist.size(); i++)
			{
				int diff1 = hanming_distance(&fp_mat, &phash_title_fplist[i]);
				printf("[%d]diff_end - %d\n", nn, diff1);
				if (diff_end <= diff1)
				{
					diff_end = diff1;
					//time_start = nn;
					printf("end time - %d -----------------------------------------\n", time_start);
				}
				
				//diff_end = diff_end < diff1 ? diff_end : diff1;
			}
			*/

			/// 已废弃，测试使用，截取指定帧索引范围内的图像
			/*
			if (nn >= 47525 && nn <= 48000)
			{
				char outpath[1024];
				memset(outpath, 0, 1024);
				sprintf(outpath, "%s/images/%s_%d.jpg", strProgram.c_str(), strFileName.c_str(), nn);
				imwrite(outpath , rgb_title);
			}
			*/
			
            nn++;
            percent((float(nn)/float(numberframes)*45) / 2);
        }
		
		printf("start time - %d\nend time - %d\n", time_start, time_end);
		
		/// 判断已检测的起始/结束时间点的有效性
		if (time_start >= time_end ||
			time_end - time_start < prog_duration * 60 * fps)
		{
			/// 起始时间点至节目结束的时长不能小于指定的节目时长，
			/// 起始时间点匹配的最大ssim值不能小于指定的阈值
			if (time_start < numberframes && 
				numberframes - time_start > prog_duration * 60 * fps &&
				simular_start >= g_ssim_simular)
			{
				time_end = -1;
			}
			else
			{
				time_start = time_end = -1;
			}
		}
		/// 起始/结束时间点匹配的最大ssim值不能小于指定的阈值
		if (simular_start < g_ssim_simular)
			time_start = -1;
		if (simular_end < g_ssim_simular)
			simular_end = -1;
		
		printf("start time - %d\nend time - %d\n", time_start, time_end);
		
		/// 计算每帧的分块灰度直方图
		/// 遍历视频回零
		capture.set(CV_CAP_PROP_POS_FRAMES, 0);
		nn = 0;	// 索引回零
		
		/// [FIX-ME]云南卫视编目拣选时，字幕标题拣选匹配场景位置！！！
		Rect rect_logo = Rect(_TITLE_SCENE_X_1_, _TITLE_SCENE_Y_1_, _TITLE_SCENE_W_1_, _TITLE_SCENE_H_1_);
		
		/// 遍历视频帧，筛选指定字幕标题位置的转场帧
        while(capture.read(frame)) 
		{
			/// 每25*5帧选一帧，降低检测样本数量
			/// 待检测视频均为转码后视频，帧率25fps
			if (nn % (25 * 5) != 0)
			{
				nn++;
				percent(23 + (float(nn)/float(numberframes)*45) / 2);
				continue;
			}
			
			// 截取视频帧的指定区域
			mat_logo = frame(rect_logo);
			/// 将遍历的视频帧转换为灰度图
            cvtColor(mat_logo, gray, CV_BGR2GRAY);
			/// 拉伸为128像素的方阵，减少计算量
            resize(gray, mat64, Size(128, 128));

			/// 生成Image对象
            img.rows = mat64.rows;
            img.cols = mat64.cols;
            img.channels = mat64.channels();
            img.depth = mat64.depth();
            img.data = mat64.data;

			/// 计算视频帧的分块灰度直方图，保存至hist数组
            int r = calc_histogram(&img, hist+nn*hist_offset);

            nn++;
            percent(23 + (float(nn)/float(numberframes)*45) / 2);
        }
        printf("%f, %f\n", capture.get(CV_CAP_PROP_POS_FRAMES), capture.get(CV_CAP_PROP_POS_MSEC));

        /// 计算视频帧直方图差值，判定是否为转场帧
        calc_score_batch(hist_err, numberframes, hist);
        if (hist != NULL) {
            free(hist);
            hist = NULL;
        }
    }

    // 第2步：转场帧阈值筛选
    sp_t v = {0};
    int last_idx = -10;
    for (int nn=0; nn<numberframes; nn++) 
	{
		/// 灰度直方图变化率小于指定阈值时跳过，
		/// 或当前转场点索引和上一个转场点间隔小
		/// 于等于10帧时跳过（应不会出现）
        if (hist_err[nn] <= thresh || (nn-last_idx) <= 10)
            continue;
			
		/// 若已检测到有效起始/结束时间点，
		/// 则转场点索引必须在此范围内
		if (time_start >= 0 && nn < time_start)
			continue;
		if (time_end >= 0 && nn > time_end)
			continue;
		
		/// 记录有效的转场点索引、类型置为1，添加至points
        v.frame_idx = nn;
        v.type = 1;
        points.push_back(v);
        last_idx = nn;
    }
    e = clock();	// 计算结束时间
    if (!silence_mode)
        printf("TIME USED [step:1]: [%.4f s] %d points\n", double(e-s)/CLOCKS_PER_SEC, points.size());

    // 第3步：人脸检测和人脸识别
    percent(50);	// 标记执行进度50%
    int face_count = 0;
	/// 必定启用人脸检测和人脸识别
    //if (with_face) {
	if (1) {
	/// 测试时使用的本地场景surf匹配图片
/*
	fp_t fp1, fp2, fp3;
	//if (with_face)
	if (1)
	{

		char* path1 = "test3.mp4.1m.jpg";
		char* path2 = "test3.mp4.1w.jpg";
		char* path3 = "test3.mp4.2mw.jpg";
		im_t *img1 = load_image(path1);
		im_t *img2 = load_image(path2);
		im_t *img3 = load_image(path3);
		
		calc_fingerprint_phash(&fp1, img1);
		calc_fingerprint_phash(&fp2, img2);
		calc_fingerprint_phash(&fp3, img3);
		
		release_image(img1);
		release_image(img2);
		release_image(img3);

	}
*/
	
	
	///	进行过场景切换判断后，场景切换帧type为1，非场景切换帧type为0；
	///	场景切换帧经过人脸探测后，人脸数小于1和大于2的type改为0，人脸数为1的type为1，人脸数为2的type为2；
	///	在对人脸数为1或2的进行感知哈希值图像匹配，匹配成功的type改为2，匹配失败的type改为1；
	if(points.size() > 0)
	{
		/// 删除原缓存视频关键帧截图目录
		char cmd[1024];
		memset(cmd, 0, 1024);
		sprintf(cmd, "rm -rf %sdata/%s/images/%s/", getFilePath(), strProgram.c_str(), strFileName.c_str());
		system(cmd);
		memset(cmd, 0, 1024);
		sprintf(cmd, "rm -rf %sdata/%s/images/%s_keyframe/", getFilePath(), strProgram.c_str(), strFileName.c_str());
		system(cmd);
		memset(cmd, 0, 1024);
		sprintf(cmd, "rm -rf %sdata/%s/images/%s_surf/", getFilePath(), strProgram.c_str(), strFileName.c_str());
		system(cmd);
		
		/// 创建新的缓存视频关键帧截图目录
		memset(cmd, 0, 1024);
		sprintf(cmd, "mkdir %sdata/%s/images/%s/", getFilePath(), strProgram.c_str(), strFileName.c_str());
		system(cmd);
		memset(cmd, 0, 1024);
		sprintf(cmd, "mkdir %sdata/%s/images/%s_keyframe/", getFilePath(), strProgram.c_str(), strFileName.c_str());
		system(cmd);
		memset(cmd, 0, 1024);
		sprintf(cmd, "mkdir %sdata/%s/images/%s_surf/", getFilePath(), strProgram.c_str(), strFileName.c_str());
		system(cmd);
		
		/// 遍历视频回零
        capture.set(CV_CAP_PROP_POS_FRAMES, 0);
        //double st = capture.get(CV_CAP_PROP_POS_MSEC);
        Mat frame, gray, mat, rgb, left, right, center, scene_l, scene_r, scene_c;
		
		im_t img, img_color;
		fp_t fp_mat;
		int diff1 = 30, diff2 = 30, diff3 = 30;
        int nn = -1, idx = 0;
        sp_t *v = &points[idx];
        while(capture.read(frame)) 
		{
            nn++;
			
			/// 仅遍历points中有的关键帧
            if (v->frame_idx != nn)
                continue;
			
			/// 保存关键帧截图至本地
			char outpath[1024];
			memset(outpath, 0, 1024);
			sprintf(outpath, "%sdata/%s/images/%s_keyframe/%s_%d.jpg", getFilePath(), strProgram.c_str(), strFileName.c_str(), strFileName.c_str(), nn);
			imwrite(outpath , frame);
			
			/// 将类型置为1，人脸检测个数默认置为0
			v->type = 1;
			v->faces_cnt = 0;
            int fc = 0;
			
			/// 进行人脸检测，fc为检测到的人脸个数，未启用！！！
			//if (with_face == true)
				//fc = facedetect(frame, NULL, NULL);	// openface人脸检测，已废弃
				//fc = facedect_seeta(outpath);			// seetaface人脸检测

			/// 当检测到的人脸个数大于指定的阈值时，
			/// points中的对应元素类型置为2，即表示
			/// 通过此步骤的筛选，未被置为2的，在此
			/// 步骤时被淘汰，进行下一个步骤时不再
			/// 检测
			if ((fc > face_cnt && with_face == true) || 
				with_face == false)
			{
				/// 累积识别出的人脸帧个数
				face_count++;
				/// 类型置为2，保存人脸个数
				v->type = 2;
				v->faces_cnt = fc;
				
				/// 保存通过人脸检测步骤的关键帧截图至本地
				char outpath[1024];
				memset(outpath, 0, 1024);
				sprintf(outpath, "%sdata/%s/images/%s/%s_%d.jpg", getFilePath(), strProgram.c_str(), strFileName.c_str(), strFileName.c_str(), nn);
				imwrite(outpath , frame);
				
				/// 已废弃，openface人脸识别
				/*
				double confidenc = face_detect((char*)pklpath, outpath);
				if (confidenc >= g_ssim_simular)
				{
					printf("[%d:%d]simular - %.3f --------------------\n", v->frame_idx, nn, confidenc);
					v->type = 2;
					v->diff = face_cnt - 2;
				}
				*/
				
			}
			printf("[%d]face_cnt - %d\n", nn, fc);
			
            if (++idx == points.size())
                break;
            v = &points[idx];
            percent(50 + (float(idx)/float(points.size())*40) / 2);
        }
		
		/// 遍历视频回零
		capture.set(CV_CAP_PROP_POS_FRAMES, 0);
		diff1 = 30, diff2 = 30, diff3 = 30;
		nn = -1, idx = 0;
		v = &points[idx];
		
		
		/// 字幕标题拣选，并去重
		///  match title heads
		fp_t fp1;
		/// 载入用于拣选字幕标题的图标图片
		char path1[1024];
		char path2[1024];
		memset(path1, 0, 1024);
		memset(path2, 0, 1024);
		sprintf(path1, "%sdata/%s/title/title_head_1.jpg", getFilePath(), (char*)g_program_type.c_str());
		im_t *img1 = load_image(path1);
		/// 计算字幕标题图标图片的phash，未启用
		calc_fingerprint_phash(&fp1, img1);
		/// 载入用于拣选字幕标题的图标图片为mat对象
		Mat mat_title_head = imread(path1, CV_LOAD_IMAGE_COLOR);
		
		release_image(img1);

		im_t img_logo;
		Mat mat_logo, mat64;
		int diff_logo_max = 30, surf_point_cnt = 0;
		double base_test = 0.0;
		fp_t fp_logo;
		/// [FIX-ME]字幕标题开头图标区域！！！
		Rect rect_logo1 = Rect(_TITLE_SCENE_X_2_, _TITLE_SCENE_Y_2_, _TITLE_SCENE_W_2_, _TITLE_SCENE_H_2_);
		/// [FIX-ME]字幕标题整行区域！！！
		Rect rect_logo2 = Rect(_TITLE_SCENE_X_3_, _TITLE_SCENE_Y_3_, _TITLE_SCENE_W_3_, _TITLE_SCENE_H_3_);
		char outpath[1024];
		
		/// 创建新的缓存视频关键帧截图目录
		/*
		memset(cmd, 0, 1024);
		sprintf(cmd, "mkdir %sdata/%s/images/%s_phashmatch/", getFilePath(), strProgram.c_str(), strFileName.c_str());
		system(cmd);
		*/
		memset(cmd, 0, 1024);
		sprintf(cmd, "mkdir %sdata/%s/images/%s_surfmatch/", getFilePath(), strProgram.c_str(), strFileName.c_str());
		system(cmd);
		/*
		memset(cmd, 0, 1024);
		sprintf(cmd, "mkdir %sdata/%s/images/%s_colorhistmatch/", getFilePath(), strProgram.c_str(), strFileName.c_str());
		system(cmd);
		*/
		
		
		vector<float> hist, hist_title_head;
		/// 拉伸字幕标题图片为64像素方阵，减少计算量
		resize(mat_title_head, mat64, Size(64, 64));
		/// 计算字幕标题图片的彩色直方图，未启用
		hist_title_head = getColorSapceHist(mat64);
		
		while (1)//capture.read(frame)) 
		{
			/// 遍历已拣选的字幕标题位置的转场帧
			nn++;
            if (v->frame_idx != nn)
                continue;
				
			if (v->type == 2)
			{
				//mat_logo = frame(rect_logo2);
				
				/// 已废弃，计算转场帧的phash，并与字幕标题图标的phash匹配
				/*
				/// phash 
				img_logo.rows = mat_logo.rows;
				img_logo.cols = mat_logo.cols;
				img_logo.channels = mat_logo.channels();
				img_logo.depth = mat_logo.depth();
				img_logo.data = mat_logo.data;
				calc_fingerprint_phash(&fp_logo, &img_logo);
				int diff1 = hanming_distance(&fp_logo, &fp1);
				diff_logo_max = diff_logo_max < diff1 ? diff_logo_max : diff1;
				printf("[%d]diff_title_head - %d\n", nn, diff1);
				*/
				
				/// surf 
				memset(outpath, 0, 1024);
				sprintf(outpath, "%sdata/%s/images/%s/%s_%d.jpg", getFilePath(), strProgram.c_str(), strFileName.c_str(), strFileName.c_str(), nn);
				memset(path2, 0, 1024);
				sprintf(path2, "%sdata/%s/images/%s_surf/%s_%d.jpg", getFilePath(), strProgram.c_str(), strFileName.c_str(), strFileName.c_str(), nn);
				
				/// outpath为转场帧截图路径，path1为字幕标题图标图片路径，
				/// path2为转场帧按指定字幕标题图标位置截取后的图片保存路径，
				/// rect_logo1为字幕标题图标位置，即用转场帧中与字幕标题图标
				/// 图片对应位置处的图像进行surf特征点匹配，返回特征点个数
				surf_point_cnt = imgmatch_surf(outpath, path1, path2, rect_logo1, Rect(0, 0, 0, 0), 128);
				printf("[%d]surf_title_head - %d\n", nn, surf_point_cnt);
				
				/// 已废弃，颜色直方图匹配转场帧与指定字幕标题图标
				/*
				/// colorhist
				resize(mat_logo, mat64, Size(64, 64));
				hist = getColorSapceHist(mat64);
				base_test = compareHist(hist, hist_title_head, CV_COMP_CORREL);
				printf("[%d]hist_title_head - %.4f\n", nn, base_test);
				*/
				
				/// 已废弃，计算转场帧的phash，并与字幕标题图标的phash匹配
				/*
				v->type = 1;
				if (diff1 <= 26)
				{
					v->type = 2;
					v->diff = diff1;
					
					memset(cmd, 0, 1024);
					sprintf(cmd, "cp -rf %sdata/%s/images/%s/%s_%d.jpg %sdata/%s/images/%s_phashmatch/%s_%d.jpg", 
						getFilePath(), program, strFileName.c_str(), strFileName.c_str(), v->frame_idx,
						getFilePath(), program, strFileName.c_str(), strFileName.c_str(), v->frame_idx);
					system(cmd);
				}
				*/
				
				/// 转场帧类型置为1
				v->type = 1;
				
				/// 当转场帧与字幕标题图标匹配的特征点个数大于等于25时，
				/// 此转场帧被判定为含有字幕标题，且将此转场帧拷贝至指定
				/// 目录，类型置为2，即通过此步骤筛选，保存匹配的特征点个数
				if (surf_point_cnt >= g_surf_match)
				{
					v->type = 2;
					v->surf_point_cnt = surf_point_cnt;
					
					memset(cmd, 0, 1024);
					sprintf(cmd, "cp -rf %sdata/%s/images/%s/%s_%d.jpg %sdata/%s/images/%s_surfmatch/%s_%d.jpg", 
						getFilePath(), program, strFileName.c_str(), strFileName.c_str(), v->frame_idx,
						getFilePath(), program, strFileName.c_str(), strFileName.c_str(), v->frame_idx);
					system(cmd);
				}
				
				/// 已废弃，颜色直方图匹配转场帧与指定字幕标题图标
				/*
				v->type = 1;
				if (base_test >= 0.65)
				{
					//v->type = 2;
					//v->diff = diff1;
					
					memset(cmd, 0, 1024);
					sprintf(cmd, "cp -rf %sdata/%s/images/%s/%s_%d.jpg %sdata/%s/images/%s_colorhistmatch/%s_%d.jpg", 
						getFilePath(), program, strFileName.c_str(), strFileName.c_str(), v->frame_idx,
						getFilePath(), program, strFileName.c_str(), strFileName.c_str(), v->frame_idx);
					system(cmd);
				}
				*/
			}
			if (++idx == points.size())
                break;
            v = &points[idx];

		}
		
		/// 字幕标题去重
		/// [FIX-ME]字幕标题整行区域，用于字幕标题帧去重！！！
		Rect rect_logo3 = Rect(_TITLE_SCENE_X_4_, _TITLE_SCENE_Y_4_, _TITLE_SCENE_W_4_, _TITLE_SCENE_H_4_);
		int points_len = points.size();
		sp_t *v_cur, *v_next;
		double simular_test = 0.0;
		Mat mat_cur, mat_next, mat_cur_rect, mat_next_rect, mat_cur_sc, mat_next_sc;
		/*
		for (int nn = 0; nn < points_len; )
		{
		*/
		
		/// 遍历视频回零
		capture.set(CV_CAP_PROP_POS_FRAMES, 0);
		nn = -1, idx = 0;	// 索引回零
		v = &points[idx];
		
		while (1) 
		{
			if (nn <= v->frame_idx)
			{
				/// 逐个遍历视频帧
				capture.read(frame);
				nn++;
				printf("nn - %d\tidx - %d\n", nn, v->frame_idx);
				
				/// 未找到待检测关键帧时，继续遍历
				if (v->frame_idx != nn)
					continue;
			}
			else
			{
				printf("nn - %d\tidx - %d\n", nn, v->frame_idx);
			}
			
			/// 仅处理通过上一步骤筛选的帧，
			/// 即含有字幕标题的转场帧
			if (v->type == 2)
			{
				/// 遍历已拣选的关键帧
				for (int j = 0; j < points_len; )
				{
					v_cur = &points[j];	// 当前遍历的关键帧
					
					/// 当查找到类型2的关键帧时，
					if (v->frame_idx == v_cur->frame_idx && v_cur->type == 2)
					{
						/*
						memset(outpath, 0, 1024);
						sprintf(outpath, "%sdata/%s/images/%s/%s_%d.jpg", 
							getFilePath(), strProgram.c_str(), strFileName.c_str(), strFileName.c_str(), v_cur->frame_idx);
						mat_cur = imread(outpath, CV_LOAD_IMAGE_COLOR);
						*/
						
						/// frame为类型2的当前关键帧图像
						mat_cur = frame;
						mat_cur_rect = mat_cur(rect_logo3);	// 截取字幕标题行
						resize(mat_cur_rect, mat_cur_sc, Size(_SSIM_SCALE_DIM_, _SSIM_SCALE_DIM_));	// 拉伸为128像素的方阵，减少计算量
						
						/// 从当前关键帧处继续向后遍历关键帧，
						/// 找到下一个类型2的关键帧，计算当前
						/// 关键帧和下一个关键帧的ssim，若ssim
						/// 值大于0.7，则表示当前关键帧与下一个
						/// 关键帧相似度很高，将下一个关键帧类型
						/// 置为1，即去重，若相似度未大于0.7，则
						/// 记录已遍历的关键帧位置，向后继续遍历
						for (int i = j + 1; i < points_len; i++)
						{
							v_next = &points[i];
							while (capture.read(frame))
							{
								nn++;
								if (nn == v_next->frame_idx)
									break;
							}
							if (nn == v_next->frame_idx && v_next->type == 2)
							{
								/*
								memset(outpath, 0, 1024);
								sprintf(outpath, "%sdata/%s/images/%s/%s_%d.jpg", 
									getFilePath(), strProgram.c_str(), strFileName.c_str(), strFileName.c_str(), v_next->frame_idx);
								mat_next = imread(outpath, CV_LOAD_IMAGE_COLOR);
								*/
								
								mat_next = frame;
								mat_next_rect = mat_next(rect_logo3);
								resize(mat_next_rect, mat_next_sc, Size(_SSIM_SCALE_DIM_, _SSIM_SCALE_DIM_));
						
								simular_test = ssim(mat_cur_sc, mat_next_sc);
								printf("[%d]cur - [%d]next - simular_test - %.4f\n", v_cur->frame_idx, v_next->frame_idx, simular_test);
								fflush(stdout);
								if (simular_test >= g_ssim_simular)
								{
									v_next->type = 1;
									printf("[%d]skip - [%d]\n", v_cur->frame_idx, v_next->frame_idx);
									fflush(stdout);
								}
								else
								{
									//j = i;
									printf("before idx - %d\ti - %d\n", idx, i);
									idx = i - 1;
									printf("current idx - %d\ti - %d\n", idx, i);
									fflush(stdout);
									break;
								}
							}
						}
						break;
					}
					else
					{
						j++;
					}
				}
			}
			++idx;
			if (idx == points.size())
                break;
            v = &points[idx];
			
		}
			
		/*
		memset(cmd, 0, 1024);
		sprintf(cmd, "rm -rf %sdata/%s/images/%s/", getFilePath(), strProgram.c_str(), strFileName.c_str());
		system(cmd);
		memset(cmd, 0, 1024);
		sprintf(cmd, "rm -rf %sdata/%s/images/%s_keyframe/", getFilePath(), strProgram.c_str(), strFileName.c_str());
		system(cmd);
		memset(cmd, 0, 1024);
		sprintf(cmd, "rm -rf %sdata/%s/images/%s_surf/", getFilePath(), strProgram.c_str(), strFileName.c_str());
		system(cmd);
		*/
	}
    }
	
    capture.release();
    e2 = clock();
    if (!silence_mode)
        printf("TIME USED [step:2]: [%.4f s] %d face frame\n", double(e2-e)/CLOCKS_PER_SEC, face_count);
		
		
    // 第4步：声音筛选，未启用
    percent(90);	// 标记执行进度90%
	
    int voice_dec = 0, voice_inc = 0;
    vector<sp_t> mute_points;
    //if (with_voice) {
	if (0) {
		/*
        // 原来版本
        int audio_len = audio_decode(path, hist_err, numberframes, fps);    
        hist_err[0] = hist_err[1];

        // 此处复用hist_err变量
        // 均值滤波，窗口大小为fps*2/3（约0.6秒）
        int kernel_size = fps*2.0/3.0;
        float *audio_db = (float*)malloc(numberframes*sizeof(float));
        if (audio_db == NULL) {
            sprintf(msg, "Memory malloc or memcpy failed, operate memory size is [%d]=[%d MB].", numberframes*sizeof(float)/1024/1024);
            LOG5CXX_FATAL(msg, 210);
        }
        mean_filter1d(hist_err, audio_db, audio_len, kernel_size);   // 一维均值滤波

        sp_t v;
        int step = fps;
        int min_idx; float min_value;
        int last_idx = -1;
        for (int nn=0; nn<audio_len; nn+=step) {
            min_idx = 0; min_value = 0;
            for (int k=0; k<step; k++) {
                if (min_value > audio_db[nn+k]) {
                    min_idx = nn+k;
                    min_value = audio_db[nn+k];
                }
            }

            if (min_value < db_thresh && (min_idx-last_idx) > step) {
                v.frame_idx = min_idx;
                v.type = 100;
                points.push_back(v);
                mute_points.push_back(v);
                last_idx = min_idx;
                voice_inc++;
            }
        }

        if (audio_db != NULL) {
            free(audio_db);
            audio_db = NULL;
        }
		*/
		
        int ret = audio_decode2(path, hist_err, numberframes, fps); 
        if (ret == 0) {
            sp_t v;
            for (int nn=0; nn<numberframes-1; nn++) {
                if (hist_err[nn] == 0)
                    continue;

                if (hist_err[nn-1] != 0 && hist_err[nn+1] != 0)
                    continue;
                v.frame_idx = nn;
                v.type = 100;
                points.push_back(v);
                mute_points.push_back(v);
                voice_inc++;
            }
        }
    }
    if (hist_err != NULL) {
        free(hist_err);
        hist_err = NULL;
    }
    e = clock();
    if (!silence_mode)
        printf("TIME USED [step:3]: [%.4f s] %d mute frame\n", double(e-e2)/CLOCKS_PER_SEC, voice_inc/2);

    percent(98);	// 标记执行进度98%

    // 第5步：结果写入文件
    FILE *fp_out = NULL;  
    if (strcmp(output, "") != 0) 
	{
        fp_out = fopen(output, "w");
        if (fp_out == NULL) {
            sprintf(msg, "open file failed!\n");
            LOG5CXX_FATAL(msg, 121);
        }
    }
	string strOutput = output;
	strOutput = strOutput.substr(
		0, 
		strOutput.find_last_of("/") + 1);
	printf("output - %s\n", strOutput.c_str());
	
    char buf[64];
    unsigned int points_len = points.size();
	
	int pre_type = -1;
	int diff = 30;
	
	/// 若已检测到有效起始时间点，写入结果集
	if (time_start >= 0)
	{
		/// 写入结果集时，当前关键帧要偏移指定数量的帧索引
		/// 调整当前帧索引，保证web页面显示时的正确定位
		if (time_start + g_time_offset > numberframes)
			g_time_offset = numberframes - time_start;
		printf("[%d]%.3f - %d\n", time_start + g_time_offset, double(time_start + g_time_offset)/double(fps), g_surf_match - 2);
		sprintf(buf, "%d %.3f %d;\n", time_start + g_time_offset, double(time_start + g_time_offset)/double(fps), 2 - 1);
		fwrite(buf, strlen(buf), 1, fp_out);
	}
	for (int nn=0; nn<points_len; nn++) 
	{
		sp_t *v = &points[nn];

		/// points中类型为2的可通过
		if (v->type == 0 || v->type == 100)
			continue;
		if (with_less && v->type == 1)
			continue;
		diff = v->diff;
		/*
		if (diff > face_cnt)
		{
			if (check_near_mute(mute_points, v->frame_idx, fps))
			{
				v->diff = v->diff - 2;
				diff = v->diff;
			}
		}
		*/
		
		/// 筛选关键帧，当两个关键帧间隔时间小于指定时间时，
		/// 仅保留首个关键帧，去除后面的一个或多个关键帧，
		/// 即将类型置为1
		for (int i = 1; i < points_len - nn; i++)
		{
			if (nn + i < points_len)
			{
				sp_t *v_next = &points[nn + i];
				double vtime = double(v->frame_idx)/double(fps);
				double v_nexttime = double(v_next->frame_idx)/double(fps);
				if (abs(v_nexttime - vtime) <= g_key_interval)
				{
					if (v->diff >= v_next->diff)
					{
						v_next->type = 1;
						printf("skip - [%d]%.3f - %d\n", v_next->frame_idx, double(v_next->frame_idx)/double(fps), v_next->diff);
					}
					else
					{
						v->type = 1;
						printf("skip - [%d]%.3f - %d\n", v->frame_idx, double(v->frame_idx)/double(fps), v->diff);
					}
				}
				else
				{
					break;
				}
			}
		}
		
		/// 再次过滤类型为1的关键帧
		if (with_less && v->type == 1)
			continue;
		
//		if (diff <= face_cnt)
//		{
			/// 写入结果集时，当前关键帧要偏移指定数量的帧索引
			/// 调整当前帧索引，保证web页面显示时的正确定位
			if (v->frame_idx + g_time_offset > numberframes)
				g_time_offset = numberframes - v->frame_idx;
			printf("[%d]%.3f - %d\n", v->frame_idx + g_time_offset, double(v->frame_idx + g_time_offset)/double(fps), v->diff);
			sprintf(buf, "%d %.3f %d;\n", v->frame_idx + g_time_offset, double(v->frame_idx + g_time_offset)/double(fps), v->type-1);
			fwrite(buf, strlen(buf), 1, fp_out);
			
			memset(cmd, 0, 1024);
			sprintf(cmd, "cp -rf %sdata/%s/images/%s/%s_%d.jpg %s/%s_%d.jpg", 
				getFilePath(), program, strFileName.c_str(), strFileName.c_str(), v->frame_idx,
				strOutput.c_str(), strFileName.c_str(), v->frame_idx);
			system(cmd);
//		}
		pre_type = v->type;
	}
	/// 若已检测到有效结束时间点，写入结果集
	if (time_end >= 0)
	{
		/// 写入结果集时，当前关键帧要偏移指定数量的帧索引
		/// 调整当前帧索引，保证web页面显示时的正确定位
		if (time_end + g_time_offset > numberframes)
			g_time_offset = numberframes - time_end;
		printf("[%d]%.3f - %d\n", time_end + g_time_offset, double(time_end + g_time_offset)/double(fps), g_surf_match - 2);
		sprintf(buf, "%d %.3f %d;\n", time_end + g_time_offset, double(time_end + g_time_offset)/double(fps), 2 - 1);
		fwrite(buf, strlen(buf), 1, fp_out);
	}
    if (fp_out) {
        fclose(fp_out);
        fp_out = NULL;
    }

    e = clock();
    percent(100);

    if (!silence_mode)
        printf("TIME USED [total]: [%.4f s]\n", double(e-s)/CLOCKS_PER_SEC);
	
	memset(cmd, 0, 1024);
	sprintf(cmd, "rm -rf %sdata/%s", getFilePath(), strProgram.c_str());
	system(cmd);
}

static void video_format_test(const char *path, const char *output)
{
    char msg[1024];

    VideoCapture capture;  
    capture.open(path);
    if(!capture.isOpened()) {
        sprintf(msg, "Not supported format.\n", path);
        LOG5CXX_FATAL(msg, 121);
    }

    unsigned int numberframes = capture.get(CV_CAP_PROP_FRAME_COUNT);
    unsigned int fps = capture.get(CV_CAP_PROP_FPS);

    if (numberframes == 0 || fps == 0) {
        sprintf(msg, "Not supported format.\n", path);
        LOG5CXX_FATAL(msg, 121);
    }
    printf("total frames: %d, fps: %d\n", numberframes, fps);

    /*{
        Mat frame;
        int c = 0;
        double st = capture.get(CV_CAP_PROP_POS_MSEC);
        printf("%.3f  ", st);
        while (capture.read(frame)) {
            double st = capture.get(CV_CAP_PROP_POS_MSEC);
            printf("%.3f  ", st);
            if (c > 100)
                break;
            c++;
        }
    }*/

    Mat frame;
    capture.set(CV_CAP_PROP_POS_FRAMES, numberframes/2);
    if (!capture.read(frame)) {
        sprintf(msg, "Not supported format.\n", path);
        LOG5CXX_FATAL(msg, 121);
    }

    imwrite(output, frame);
}

static char *optstring = "a:A:b:B:c:C:d:D:e:E:f:F:g:G:H:I:j:J:k:t";
static struct option long_options[] = {         //  no_argument--0,required_argument--1,optional_argument--2
    {"video",           1, NULL, 'a'},
    {"output",          1, NULL, 'A'},
    {"with_face",       0, NULL, 'b'},
    {"with_voice",      0, NULL, 'B'},
    {"db_thresh",       1, NULL, 'c'},
    {"less",            0, NULL, 'C'},
	{"face_cnt",       	1, NULL, 'd'},
	{"surf_match",      1, NULL, 'D'},
	{"vheads",       	1, NULL, 'e'},
	{"vtails",       	1, NULL, 'E'},
	{"scene",       	1, NULL, 'f'},
	{"pkl",       		1, NULL, 'F'},
	{"prog_duration",   1, NULL, 'g'},
	{"with_facematch",  0, NULL, 'G'},
	{"program",       	1, NULL, 'H'},
	{"ssim_simular",    1, NULL, 'i'},	
	{"max_mapsurf_idr",	1, NULL, 'I'},
	{"max_result_cnt",	1, NULL, 'j'},
	{"time_offset",     1, NULL, 'J'},
	{"key_interval",    1, NULL, 'k'},
	{"program_type",    1, NULL, 'K'},
    {"test",            0, NULL, 't'},
    {0, 0, 0, 0}  
};
int cmd_shotdetect(int argc, char **argv)
{
    const int max_path = MAX_PATH;

    char _video[max_path], _output[max_path], _pkl[max_path], \
	_program[max_path], _vheads[max_path], _vtails[max_path], \
	_scene[max_path], _program_type[max_path];
    strcpy(_video, ""); strcpy(_output, ""); strcpy(_pkl, ""); 
	strcpy(_program, ""); strcpy(_vheads, ""); strcpy(_vtails, ""); 
	strcpy(_scene, ""); strcpy(_program_type, "");
    bool with_face = false, with_voice = false, with_less = true, just_test = false, with_facematch = false;
    float db_thresh = _DB_THRESH_DEFAULT_;
	float ssim_simular = _SSIM_SIMULAR_DEFAULT_;
	float prog_duration = _PROG_DURATION_DEFAULT_;
	int face_cnt = _FACE_CNT_DEFAULT_;
	int surf_match = _SURF_MATCH_DEFAULT_;
	int sx, sy, sw, sh, ex, ey, ew, eh;
    int opt, option_index = 0;
    while ((opt = getopt_long(argc, argv, optstring, long_options, &option_index)) != -1) 
	{

        switch(opt)
		{
            case 'a':{ strcpy(_video, optarg); break; }
            case 'A':{ strcpy(_output, optarg); break; }
            case 'b':{ with_face = true; break; }
            case 'B':{ with_voice = true; break; }
            case 'c':{ db_thresh = atof(optarg); break; }
            case 'C':{ with_less = true; break; }
			case 'd':{ face_cnt = atoi(optarg); break; }
			case 'D':{ surf_match = atoi(optarg); break; }
			case 'e':{ strcpy(_vheads, optarg); break; }
			case 'E':{ strcpy(_vtails, optarg); break; }
			case 'f':{ strcpy(_scene, optarg); break; }
			case 'F':{ strcpy(_pkl, optarg); break; }
			case 'g':{ prog_duration = atof(optarg); break; }
			case 'G':{ with_facematch = true; break; }
			case 'H':{ strcpy(_program, optarg); break; }
			case 'i':{ ssim_simular = atof(optarg); break; }
			case 'I':{ g_max_mapsurf_idr = atoi(optarg); break; }
			case 'j':{ g_max_result_cnt = atoi(optarg); break; }
			case 'J':{ g_time_offset = atoi(optarg); break; }
			case 'k':{ g_key_interval = atof(optarg); break; }
			case 'K':{ strcpy(_program_type, optarg); break; }
            case 't':{ just_test = true; break; }
        }
    }

	/// --test
    if (just_test) 
	{
        video_format_test(_video, _output);
        return 0;
    }

    /// --video and --output
    if (strcmp(_video, "") != 0 && 
		strcmp(_output, "") != 0)
	{
		/// output
	    FILE *fp_out = NULL;
		if (strcmp(_output, "") != 0)
		{
			fp_out = fopen(_output, "w");
			if (fp_out == NULL)
			{
				printf("open file failed - %s!\n", _output);
				return -1;
			}
		}
		if (fp_out)
		{
			fclose(fp_out);
			fp_out = NULL;
		}
		
		/// db_thresh
        if (db_thresh >= -1)
            db_thresh = -45;
			
		/// prog_duration
        if (prog_duration <= 1)
            prog_duration = 1;
			
		/// ssim_simular
        if (ssim_simular > 1.0)
            ssim_simular = 1.0;
		if (ssim_simular < 0.0)
            ssim_simular = 0.0;
		
		/// face_cnt
        if (face_cnt < 0)
            face_cnt = 0;
			
		/// g_max_mapsurf_idr
        if (g_max_mapsurf_idr < 0)
            g_max_mapsurf_idr = 0;
			
		/// g_max_result_cnt
        if (g_max_result_cnt < g_max_mapsurf_idr + 1)
            g_max_result_cnt = g_max_mapsurf_idr + 1;
			
		/// g_time_offset
        if (g_time_offset < 0)
            g_time_offset = 0;
			
		/// g_key_interval
        if (g_key_interval < 0)
            g_key_interval = 0;
			
		/// pkl
		if (strcmp(_pkl, "") != 0)
        {
			
		}
		
		/// vheads
		if (strcmp(_vheads, "") != 0)
        {
			string str = _vheads;
			vector<string> _vec_vheads = split(str, ";");
			MatchImg* mimg;
			for (int i = 0; i < _vec_vheads.size(); i++)
			{
				printf("vhead - %s\n", _vec_vheads[i].c_str());
				vector<string> _vec_vhead = split(_vec_vheads[i], ",");
				mimg = new MatchImg();
				for (int j = 0; j < _vec_vhead.size(); j++)
				{
					printf("%s\t", _vec_vhead[j].c_str());
					if (j == 0)//_vec_vhead[j].find("url=") != std::string::npos)
					{
						string strUrl = _vec_vhead[j];
						/*
						strUrl = 
							strUrl.substr(strUrl.find_first_of("url=") + 4,
							strUrl.length() - 4 - strUrl.find_first_of("url="));
						*/
						if (strUrl.find("\"") != std::string::npos)
							strUrl = strUrl.substr(1, strUrl.length() - 2);
						mimg->strMatchImgUrl = strUrl;
						printf("url - %s\n", mimg->strMatchImgUrl.c_str());
					}
					else if (j == 1)//_vec_vhead[j].find("x=") != std::string::npos)
					{
						string strX = _vec_vhead[j];
						/*
						strX = 
							strX.substr(strX.find_first_of("x=") + 2,
							strX.length() - 2 - strX.find_first_of("x="));
						*/
						mimg->X = atoi(strX.c_str());
						printf("x - %d\n", mimg->X);
					}
					else if (j == 2)//_vec_vhead[j].find("y=") != std::string::npos)
					{
						string strY = _vec_vhead[j];
						/*
						strY = 
							strY.substr(strY.find_first_of("y=") + 2,
							strY.length() - 2 - strY.find_first_of("y="));
						*/
						mimg->Y = atoi(strY.c_str());
						printf("y - %d\n", mimg->Y);
					}
					else if (j == 3)//_vec_vhead[j].find("w=") != std::string::npos)
					{
						string strW = _vec_vhead[j];
						/*
						strW = 
							strW.substr(strW.find_first_of("w=") + 2,
							strW.length() - 2 - strW.find_first_of("w="));
						*/
						mimg->W = atoi(strW.c_str());
						printf("w - %d\n", mimg->W);
					}
					else if (j == 4)//_vec_vhead[j].find("h=") != std::string::npos)
					{
						string strH = _vec_vhead[j];
						/*
						strH = 
							strH.substr(strH.find_first_of("h=") + 2,
							strH.length() - 2 - strH.find_first_of("h="));
						*/
						mimg->H = atoi(strH.c_str());
						printf("h - %d\n", mimg->H);
					}
				}
				g_vec_vheads.push_back(mimg);
				printf("\n");
			}
		}
		
		/// vtails
		if (strcmp(_vtails, "") != 0)
        {
			string str = _vtails;
			vector<string> _vec_vtails = split(str, ";");
			MatchImg* mimg;
			for (int i = 0; i < _vec_vtails.size(); i++)
			{
				printf("vtail - %s\n", _vec_vtails[i].c_str());
				vector<string> _vec_vtail = split(_vec_vtails[i], ",");
				mimg = new MatchImg();
				for (int j = 0; j < _vec_vtail.size(); j++)
				{
					printf("%s\t", _vec_vtail[j].c_str());
					if (j == 0)//_vec_vtail[j].find("url=") != std::string::npos)
					{
						string strUrl = _vec_vtail[j];
						/*
						strUrl = 
							strUrl.substr(strUrl.find_first_of("url=") + 4,
							strUrl.length() - 4 - strUrl.find_first_of("url="));
						*/
						if (strUrl.find("\"") != std::string::npos)
							strUrl = strUrl.substr(1, strUrl.length() - 2);
						mimg->strMatchImgUrl = strUrl;
						printf("url - %s\n", mimg->strMatchImgUrl.c_str());
					}
					else if (j == 1)//_vec_vtail[j].find("x=") != std::string::npos)
					{
						string strX = _vec_vtail[j];
						/*
						strX = 
							strX.substr(strX.find_first_of("x=") + 2,
							strX.length() - 2 - strX.find_first_of("x="));
						*/
						mimg->X = atoi(strX.c_str());
						printf("x - %d\n", mimg->X);
					}
					else if (j == 2)//_vec_vtail[j].find("y=") != std::string::npos)
					{
						string strY = _vec_vtail[j];
						/*
						strY = 
							strY.substr(strY.find_first_of("y=") + 2,
							strY.length() - 2 - strY.find_first_of("y="));
						*/
						mimg->Y = atoi(strY.c_str());
						printf("y - %d\n", mimg->Y);
					}
					else if (j == 3)//_vec_vtail[j].find("w=") != std::string::npos)
					{
						string strW = _vec_vtail[j];
						/*
						strW = 
							strW.substr(strW.find_first_of("w=") + 2,
							strW.length() - 2 - strW.find_first_of("w="));
						*/
						mimg->W = atoi(strW.c_str());
						printf("w - %d\n", mimg->W);
					}
					else if (j == 4)//_vec_vtail[j].find("h=") != std::string::npos)
					{
						string strH = _vec_vtail[j];
						/*
						strH = 
							strH.substr(strH.find_first_of("h=") + 2,
							strH.length() - 2 - strH.find_first_of("h="));
						*/
						mimg->H = atoi(strH.c_str());
						printf("h - %d\n", mimg->H);
					}
				}
				g_vec_vtails.push_back(mimg);
				printf("\n");
			}
		}
		
		/// scene
		if (strcmp(_scene, "") != 0)
        {
			string str = _scene;
			vector<string> _vec_scene = split(str, ",");
			for (int i = 0; i < _vec_scene.size(); i++)
			{
				printf("scene - %s\n", _vec_scene[i].c_str());
				string strUrl = _vec_scene[i];
				if (strUrl.find("\"") != std::string::npos)
					strUrl = strUrl.substr(1, strUrl.length() - 2);
				printf("url - %s\n", strUrl.c_str());
				
				g_vec_scenes.push_back(strUrl);
			}
		}
		
		/// program
		if (strcmp(_program, "") == 0)
		{
			//sprintf(_program, "%d", time(NULL));
			struct timeval tv;
			gettimeofday(&tv, NULL);
			sprintf(_program, "%ld", tv.tv_sec * 1000000 + tv.tv_usec);
		}
		
		/// program_type
		if (strcmp(_program_type, "") != 0)
		{
			g_program_type = _program_type;
		}
		
		printf("program - %s\n", _program);
		printf("video - %s\n", _video);
		printf("output - %s\n", _output);
		printf("prog_duration - %f\n", prog_duration);
		printf("with_face - %d\n", with_face);
		printf("with_facematch - %d\n", with_facematch);
		printf("with_voice - %d\n", with_voice);
		printf("db_thresh - %f\n", db_thresh);
		printf("with_less - %d\n", with_less);
		printf("face_cnt - %d\n", face_cnt);
		printf("surf_match - %d\n", surf_match);
		printf("pkl - %s\n", _pkl);
		printf("vheads - %s\n", _vheads);
		printf("vtails - %s\n", _vtails);
		printf("scene - %s\n", _scene);
		printf("ssim_simular - %f\n", ssim_simular);
		printf("max_mapsurf_idr - %d\n", g_max_mapsurf_idr);
		printf("max_result_cnt - %d\n", g_max_result_cnt);
		printf("time_offset - %d\n", g_time_offset);
		printf("key_interval - %f\n", g_key_interval);
		printf("test - %s\n", just_test);

        video_shotdetect(
			_video, _output, 
			_program, 
			_pkl, 
			with_face, face_cnt, 
			with_facematch, surf_match, 
			with_voice, db_thresh, 
			prog_duration, 
			ssim_simular, 
			with_less);
		printf("[imchar]-[exec done.]\n");
        return 0;
    }

    LOG5CXX_FATAL("fatal: argument error.", 1);
    return 0;
}

void cmd_shotdetect_usage()
{   
    printf("\
shotdetect:\n\
 -a/--video <file>  Video file full path\n\
 -A/--output <file> Output result file path\n\
 -b/--with_face     Detect face frame\n\
 -B/--with_voice    Detect mutex frame\n\
 -c/--db_thresh <float> Db value for mutex, between (-infinite, 0), default is -45\n\
 -C/--less          Less the result\n\
 -d/--face_cnt <int> Face detect count for one of face frame\n\
 -D/--surf_match <int> Surf match points count for one of face frame\n\
 -e/--vheads <string> Start title - image url / image rect.\n\
 -E/--vtails <string> End title - image url / image rect.\n\
 -f/--scene <string> Surf match - image url.\n\
 -F/--pkl <file>    The file of face dectect model.\n\
 -g/--prog_duration <float> The duration of the program file.\n\
 -G/--with_facematch Microsoft face detect.\n\
 -H/--program		The name of task.\n\
 -i/--ssim_simular 	The simular with SSIM.\n\
 -I/--max_mapsurf_idr The max idr of surf's map.\n\
 -j/--max_result_cnt The max size of result.\n\
 -J/--time_offset The time offset of result.\n\
 -k/--key_interval The interval of key points.\n\
 -K/--program_type	The type of task.\n\
 -t/--test          Test the video\n\
");
}
