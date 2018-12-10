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
#include "mycurl.h"

#define _PHASH_FACEDETECT_			26

#define _MAX_MAPSURF_IDR_			30
#define _MAX_RESULT_CNT_			35
#define _TIME_OFFSET_				0

#define _SSIM_SCALE_DIM_			128
#define _FACEMATCH_CALC_CNT_		10
#define _FACEMATCH_CALC_INTERVAL_	30.0

#define _KEYPOINT_MIN_INTERVAL_		2.0

#define _HIST_THRESH_DEFAULT_		0.3
#define _DB_THRESH_DEFAULT_			-45.0
#define _SSIM_SIMULAR_DEFAULT_		0.7
#define _PROG_DURATION_DEFAULT_		24.0
#define _FACE_CNT_DEFAULT_			2
#define _SURF_MATCH_DEFAULT_		5


int g_max_mapsurf_idr = _MAX_MAPSURF_IDR_;
int g_max_result_cnt = _MAX_RESULT_CNT_;
int g_time_offset = _TIME_OFFSET_;
float g_key_interval = _KEYPOINT_MIN_INTERVAL_;
string g_program_type = "default";
float g_hist_thresh = 0.3;
int g_surf_match = 5;
float g_ssim_simular = 0.7;
int g_low_hist_cnt = 25;
float g_low_hist_thresh = 0.1;
float g_low_hist_thresh_top = 2.0;
string g_res_upload_url = "";
int g_res_width = 100, g_res_height = 56;
string g_program = "";
float g_delay_scenecut = 1.0;
string g_scene_type = "caffe"; // "surf" - surf_detect_mutiple or "caffe" - caffe_detect_mutiple
int g_shotimage = 0;

typedef struct HistRange
{
	int X;
	int Y;
	float W_scale;
	float H_scale;
} HistRange;

HistRange g_histrange;

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
vector<string> g_vec_seeta_label_list;


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

/// 关键帧检测相关信息
typedef struct {
    int frame_idx;
    int type;         //0为无效，1为转场，2为一个人物，3为多个人物，100为静音
	float diff;
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

/// 图像关键帧识别
static int video_shotdetect(
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
	int ret = 0;
	
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
		return -1;
	}
	
	
	/// 创建以当前节目ID命名的缓存目录
	sprintf(tmppath, "%sdata/%s", getFilePath(), program);
	memset(cmd, 0, 1024);
	if (opendir(tmppath) == NULL)
	{
		sprintf(cmd, "mkdir %s %s/scene %s/title %s/images %s/seeta %s/seeta/label_faces", 
			tmppath, tmppath, tmppath, tmppath, tmppath, tmppath);
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
				sprintf(cmd, "mkdir %s %s/scene %s/title %s/images %s/seeta %s/seeta/label_faces", 
					tmppath, tmppath, tmppath, tmppath, tmppath, tmppath);
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
			sprintf(cmd, "wget -t 5 -T 30 -O %s %s", 
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
			sprintf(cmd, "wget -t 5 -T 30 -O %s %s", 
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
			sprintf(cmd, "wget -t 5 -T 30 -O %s %s", 
				tmppath, g_vec_scenes[i].c_str());
			printf("cmd - %s\n", cmd);
			system(cmd);
			printf("scene url[%d] - %s\n", i, g_vec_scenes[i].c_str());
		}
	}
	
	
	/// 若用于人脸识别的人脸模型图片路径缓存
	/// 队列中有数据，则下载这些图片数据至本
	/// 地，保存至"data/seeta/[program_type]/label_faces"
	/// 文件名称保持原URL中的文件名
	if (g_vec_seeta_label_list.size() > 0)
	{
		/// 人脸人脸模型图片路径缓存队列中有
		/// 数据，需要下载
		printf("seeta_label_list size - %d\n", g_vec_seeta_label_list.size());
		for (int i = 0; i < g_vec_seeta_label_list.size(); i++)
		{
			/// 获取URL
			if (g_vec_seeta_label_list[i] == "")
				continue;

			/// 获取文件名
			string strImgPath = g_vec_seeta_label_list[i];
			strImgPath = 
				strImgPath.substr(strImgPath.find_last_of("/") + 1, 
				strImgPath.length() - 1 - strImgPath.find_last_of("/"));
			
			/// 下载保存至本地
			memset(cmd, 0, 1024);
			memset(tmppath, 0, 1024);
			sprintf(tmppath, "%sdata/%s/seeta/label_faces/%s",
				getFilePath(), strProgram.c_str(), strImgPath.c_str());
			sprintf(cmd, "wget -t 5 -T 30 -O %s %s", 
				tmppath, g_vec_seeta_label_list[i].c_str());
			printf("cmd - %s\n", cmd);
			system(cmd);
			printf("seeta_label_list url[%d] - %s\n", i, g_vec_seeta_label_list[i].c_str());
		}
		
		/// 下载图片文件完毕，生成label_file_list.txt
		memset(cmd, 0, 1024);
		memset(tmppath, 0, 1024);
		sprintf(tmppath, "%sdata/%s/seeta/label_faces/",
			getFilePath(), strProgram.c_str());
		sprintf(cmd, "cd %s && sh /usr/local/bin/data/seeta/default/label_faces/create_txt.sh", 
			tmppath);
		printf("cmd - %s\n", cmd);
		system(cmd);
	}
	
	
	/// 下载视频文件保存至本地
	
	
	g_surf_match = surf_match;
	g_ssim_simular = ssim_simular;
	
	/// 读取参数配置信息
	memset(tmppath, 0, 1024);
	sprintf(tmppath, "%sdata/seeta/%s/setting.txt",
		getFilePath(), (char*)g_program_type.c_str());
	if (access(tmppath, F_OK) == 0)
	{
		char line[1024];
		FILE *fp = fopen(tmppath, "r");  
		if (fp == NULL)  
		{  
			printf("can not load file!");  
			return -1;  
		}  
		while (!feof(fp))  
		{
			fgets(line, 1024, fp);
			string str_line = line;
			printf("getline - %s\n", line);
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
			if (str_line.find("KEY_INTERVAL") != string::npos)
			{
				string str_val = str_line.substr(strlen("KEY_INTERVAL") + 1, strlen(str_line.c_str()) - strlen("KEY_INTERVAL") - 1);
				double _val = atof(str_val.c_str());
				printf("val[%s] - %.4f\n", (char*)str_val.c_str(), _val);
				if (_val > 0.0 && _val < 60.0)
				{
					g_key_interval = _val;
				}
				printf("KEY_INTERVAL - %.4f\n", g_key_interval);
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
			if (str_line.find("LOW_HIST_THRESH") != string::npos)
			{
				string str_val = str_line.substr(strlen("LOW_HIST_THRESH") + 1, strlen(str_line.c_str()) - strlen("LOW_HIST_THRESH") - 1);
				double _val = atof(str_val.c_str());
				printf("val[%s] - %.4f\n", (char*)str_val.c_str(), _val);
				if (_val > 0.0 && _val < 1.0)
				{
					g_low_hist_thresh = _val;
				}
				printf("LOW_HIST_THRESH - %.4f\n", g_low_hist_thresh);
			}
			if (str_line.find("LOW_HIST_THRESH_TOP") != string::npos)
			{
				string str_val = str_line.substr(strlen("LOW_HIST_THRESH_TOP") + 1, strlen(str_line.c_str()) - strlen("LOW_HIST_THRESH_TOP") - 1);
				double _val = atof(str_val.c_str());
				printf("val[%s] - %.4f\n", (char*)str_val.c_str(), _val);
				if (_val > 0.0 && _val < 1.0)
				{
					g_low_hist_thresh_top = _val;
				}
				printf("LOW_HIST_THRESH_TOP - %.4f\n", g_low_hist_thresh_top);
			}
			if (str_line.find("LOW_HIST_CNT") != string::npos)
			{
				string str_val = str_line.substr(strlen("LOW_HIST_CNT") + 1, strlen(str_line.c_str()) - strlen("LOW_HIST_CNT") - 1);
				double _val = atoi(str_val.c_str());
				printf("val[%s] - %d\n", (char*)str_val.c_str(), _val);
				if (_val > 0 && _val <= 500)
				{
					g_low_hist_cnt = _val;
				}
				printf("LOW_HIST_CNT - %d\n", g_low_hist_cnt);
			}
			if (str_line.find("DELAY_SCENECUT") != string::npos)
			{
				string str_val = str_line.substr(strlen("DELAY_SCENECUT") + 1, strlen(str_line.c_str()) - strlen("DELAY_SCENECUT") - 1);
				double _val = atof(str_val.c_str());
				printf("val[%s] - %.4f\n", (char*)str_val.c_str(), _val);
				if (_val > 0.0 && _val <= 30.0)
				{
					g_delay_scenecut = _val;
				}
				printf("DELAY_SCENECUT - %.4f\n", g_delay_scenecut);
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
	unsigned int width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	unsigned int height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    unsigned int hist_dim = 64, resize_dim = 2;     // 直方图bin数量，resize_dim是图像一个方向维度的分块数量（一幅图像计算resize_dim*resize_dim个直方图）
    const unsigned int hist_offset = hist_dim*resize_dim*resize_dim;	// 按resize_dim分块数，hist_offset为一帧图像的直方图数据个数
    float thresh = g_hist_thresh;	// 转场识别时的直方图变化阈值

	/// 初始化保存直方图变化率的数组
    float *hist_err = (float*)calloc(numberframes*sizeof(float),1);
    if (hist_err == NULL) {
        sprintf(msg, "Memory malloc or memcpy failed, operate memory size is [%d]=[%d MB].", numberframes*sizeof(float)/1024/1024);
        LOG5CXX_FATAL(msg, 210);
    }
	
    vector<sp_t> points;	// 关键帧检测后的节点队列
	map<int, double> confidence_map;	// 人脸识别（openface方式）时，保存每个关键帧索引对应的可信度
	map<int, int> surf_map;	// surf场景匹配时，保存每个关键帧索引对应的匹配上的特征点个数
	map<int, int> caffe_map;	// caffe场景匹配时，保存每个关键帧索引对应的类别
	int time_start = 0, time_end = 0;	// 起始/结束时间点检测后的起始/结束时间点
	
    s = clock();	// 开始执行计时

    // 第1步：首先进行起始/结束时间点检测，然后计算每个帧的分块灰度直方图
	printf("\n\ncalculate hist step\n\n");
    percent(0);	// 标记执行进度0%
    {
		/// 创建用于保存每个帧的分块灰度直方图数据的数组
        float *hist = (float*)calloc((numberframes+1)*hist_dim*resize_dim*resize_dim*sizeof(float),1);
        if (hist == NULL) {
            sprintf(msg, "Memory malloc or memcpy failed, operate memory size is [%d]=[%d MB].", numberframes*hist_dim*resize_dim*resize_dim*sizeof(float)/1024/1024);
            LOG5CXX_FATAL(msg, 210);
        }
		
        im_t img;
        Mat frame, gray, mat_logo, mat64, rgb, rgb_title, start_title, end_title, mat_sc;
				
        int nn = 0;	// 当前遍历的帧的索引
		int hist_cnt = 0;	// 当前计算直方图的帧数，默认每秒计算一次
		double simular_start = 0.0, simular_end = 0.0;	// 保存每帧与起始/结束场景图片进行ssim匹配后的最大值
		int diff_start = 0, diff_end = 0;
		fp_t fp_mat;
		
		/// 当存在起始/结束时间点检测的场景图片时，开始遍历视频帧进行检测
		while((mat_vheads.size() > 0 || mat_vtails.size() > 0) && capture.read(frame)) 
		{
			/// 每25帧选一帧，降低检测样本数量
			/// 待检测视频均为转码后视频，帧率25fps
			if (nn % 10 != 0)
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
				resize(rgb_title, start_title, Size(_SSIM_SCALE_DIM_, _SSIM_SCALE_DIM_));	// 拉伸视频帧截取图像
				
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
			
			/// 结束时间点匹配
			for (int i = 0; i < mat_vtails.size(); i++)
			{
				if (i > g_vec_vtails.size() - 1 && g_vec_vtails[i]->strMatchImgUrl == "")
					continue;
				
				/// 结束时间点场景图片在视频中匹配的位置
				Rect rect_end = 
					Rect(g_vec_vtails[i]->X, g_vec_vtails[i]->Y, g_vec_vtails[i]->W, g_vec_vtails[i]->H);
				rgb_title = rgb(rect_end);	// 截取视频帧的指定区域
				resize(rgb_title, end_title, Size(_SSIM_SCALE_DIM_, _SSIM_SCALE_DIM_));	// 拉伸视频帧截取图像
				
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
			
            nn++;
            percent((float(nn)/float(numberframes)*45) / 2);
        }
		
		printf("start time - %d\nend time - %d\n", time_start, time_end);
		
		/// 判断已检测的起始/结束时间点的有效性
		if (time_start >= time_end ||
			time_end - time_start < prog_duration * 60 * fps)	// 结束时间点减起始时间点不能小于指定的节目时长
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
			time_end = -1;
		
		printf("start time - %d\nend time - %d\n", time_start, time_end);
		
		system("date");
		/// 计算每帧的分块灰度直方图
		/// 遍历视频回零
		capture.set(CV_CAP_PROP_POS_FRAMES, 0);
		nn = 0;	// 索引回零
		
		/// 判断转场识别区域的坐标有效性
		int x = 0, y = 0;
		float w_scale = 1.0, h_scale = 1.0;
		if ((g_histrange.X >= 0 && g_histrange.X < width) &&
			(g_histrange.Y >= 0 && g_histrange.Y < height))
		{
			x = g_histrange.X;
			y = g_histrange.Y;
			float ws = width * g_histrange.W_scale + x;
			float hs = height * g_histrange.H_scale + y;
			if ((ws > 0 && ws <= width) &&
				(hs > 0 && hs <= height))
			{
				w_scale = g_histrange.W_scale;
				h_scale = g_histrange.H_scale;
			}
			else
			{
				x = 0;
				y = 0;
				w_scale = 1.0;
				h_scale = 1.0;
			}
		}
		else
		{
			x = 0;
			y = 0;
			w_scale = 1.0;
			h_scale = 1.0;
		}
		printf("hist_range X - %d, Y - %d, W_scale - %f, H_scale - %f\n", 
			x, y, w_scale, h_scale);

        while(capture.read(frame)) 
		{	
			/// 裁剪视频帧，防止画面下方出现字幕等情况时，触发转场识别
			/// 因此，此处仅取原始画面的上2/3，宽度不变
			Rect rect_crop = Rect(x, y, frame.cols * w_scale, frame.rows * h_scale);
			mat_sc = frame(rect_crop);
			/// 将遍历的视频帧转换为灰度图
            cvtColor(mat_sc, gray, CV_BGR2GRAY);
			/// 拉伸为64像素的方阵，减少计算量
            resize(gray, mat64, Size(64, 64));

			/// 生成Image对象
            img.rows = mat64.rows;
            img.cols = mat64.cols;
            img.channels = mat64.channels();
            img.depth = mat64.depth();
            img.data = mat64.data;

            /// 计算视频帧的分块灰度直方图，保存至hist数组
			int r = calc_histogram(&img, hist+nn*hist_offset);

            nn++;
			hist_cnt++;
            percent(23 + (float(nn)/float(numberframes)*45) / 2);
        }
        printf("%f, %f\n", capture.get(CV_CAP_PROP_POS_FRAMES), capture.get(CV_CAP_PROP_POS_MSEC));

        /// 计算视频帧直方图差值，判定是否为转场帧
		calc_score_batch(hist_err, numberframes, hist);
		/// 释放hist数组
        if (hist != NULL) 
		{
            free(hist);
            hist = NULL;
        }
    }

    // 第2步：转场帧阈值筛选
	printf("\n\ngetpoints step\n\n");
    sp_t v = {0};
    int last_idx = -10;
    for (int nn=0, hist_idr = 0; nn<numberframes; nn++) 
	{		
//		printf("[%d]hist_diff - %.4f\tlast_idx - %d\n", nn, hist_err[nn], last_idx);
		
		/// 灰度直方图变化率小于指定阈值时，跳过
		hist_idr++;
		if (hist_err[nn] <= thresh)// || (nn-last_idx) <= 10)	// 当前转场点与上一个转场点间隔必须大于等于10个帧，未启用
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

    // 第3步：人脸检测和人脸识别
	printf("\n\nface detect step, %d points\n\n", points.size());
    percent(50);	// 标记执行进度50%
    int face_count = 0;
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
			
			/// 进行人脸检测，fc为检测到的人脸个数
			if (with_face == true)
				fc = facedect_seeta(outpath);			// seetaface人脸检测

			/// 当检测到的人脸个数大于指定的阈值时，
			/// points中的对应元素类型置为2，即表示
			/// 通过此步骤的筛选，未被置为2的，在此
			/// 步骤时被淘汰，进行下一个步骤时不再
			/// 检测
			if ((fc > 0 && fc <= face_cnt && with_face == true) || 
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

				printf("face detected[%s]\n", outpath);
			}
//			printf("[%d]face_cnt - %d\n", nn, fc);
			
            if (++idx == points.size())
                break;
            v = &points[idx];
            percent(50 + (float(idx)/float(points.size())*40) / 2);
        }		
		printf("after face detect, face_cnt - %d\n", face_count);
		face_count = 0;

		if (g_scene_type == "surf")
		{
			/// 并行执行surf场景特征点匹配检测，
			/// 并根据匹配的特征点数量进行冒泡
			/// 排序
			/// bubble sort map of surf
			typedef map<int,int>::iterator IT;
			/// 并行执行surf场景特征点匹配检测，返回map，key为关键帧索引，value为匹配的特征点个数
			surf_map = surf_detect_mutiple(1, (char*)strProgram.c_str(), (char*)strFileName.c_str());
			
			/// 初始化一个大小为map长度的数组，保存排序后的关键帧索引
			//int* surf_key_des = (int*)malloc(sizeof(int*) * surf_map.size());
			int surf_key_des[surf_map.size()];
			
			/// 打印原始排序的关键帧特征点信息日志，按帧索引排序
			int i, j, tmp;
			i = 0;
			for(IT it = surf_map.begin(); it != surf_map.end(); it++) 
			{
				printf("map[%d] - surf = %d\n", it->first, surf_map[it->first]);
				
				surf_key_des[i] = it->first;
				i++;
			}
			
			/// 根据匹配的特征点数量进行冒泡排序，保存关键
			/// 帧索引顺序，按特征点数量由高到低，降序排列
			int is_changed = 1;
			for (i = 0; i < surf_map.size(); ++i) 
			{  
				is_changed = 0;  
				for (j = 0; j < surf_map.size() - 1 - i; ++j) 
				{  
					if (surf_map[surf_key_des[j]] < surf_map[surf_key_des[j + 1]]) 
					{  
						is_changed = 1;  
						tmp = surf_key_des[j];  
						surf_key_des[j] = surf_key_des[j + 1];
						surf_key_des[j + 1] = tmp;  
					}  
				}  
				if (is_changed == 0) 
					break;  
			}
			
			/// 查找排序中第30个（g_max_mapsurf_idr的值）位置处的关键帧索引，
			/// 同时查找map中的特征点匹配数量，若排序中后面存在大于指定surf
			/// 阈值减1的关键帧，则保留其索引，经此步骤处理后，可筛选出全部
			/// surf特征点匹配数量大于指定阈值减1的关键帧，即为下一步人脸识
			/// 别的执行范围
			int max_idr = g_max_mapsurf_idr - 1;
			int max_surf = -1;
			int result_cnt = 0;
			int result_max_cnt = g_max_result_cnt;
			for (i = 0; i < surf_map.size(); i++)
			{
				printf("sort - %d map[%d] - surf = %d\n", i, surf_key_des[i], surf_map[surf_key_des[i]]);
				if (surf_map.size() - 1 >= max_idr && 
					surf_map.size() - 1 >= i)// && i < result_max_cnt)	// result_max_cnt用于限制返回的max_idr的最大值，未启用
				{
					if (i == max_idr)
						max_surf = surf_map[surf_key_des[i]];
					else if (i > max_idr)
					{
						if (max_surf >= g_surf_match - 1 && surf_map[surf_key_des[i]] >= g_surf_match - 1)
						{
							max_idr = i;
						}
					}
				}
			}
			
			/// 执行人脸识别，仅对上一步骤中筛选后的关键帧进行识别，
			/// 若筛选后的关键帧总数量小于默认的30（起始筛选下限），
			/// 则置max_idr为关键帧总数量，即识别范围为全部关键帧
			if (surf_map.size() - 1 < max_idr)
			{
				max_idr = surf_map.size() - 1;
			}
			if (max_idr >= 0 && surf_map.size() - 1 >= max_idr)
			{
				printf("max - %d map[%d] - surf = %d\n", max_idr, surf_key_des[max_idr], surf_map[surf_key_des[max_idr]]);
				
				// surf - vpoints
				/// 最大结果集上限，未启用
				result_max_cnt = g_max_result_cnt;
				/// 若已检测到有效起始/结束时间点，
				/// 则分别要占用一个结果，即结果集
				/// 上限值分别减去1，未启用
				if (time_start >= 0)
					result_max_cnt -= 1;
				if (time_end >= 0)
					result_max_cnt -= 1;
				int face_match_cnt = 0;
				for (i = 0; i < points.size(); i++)
				{
					v = &points[i];
					if (v->type == 2)
					{
						v->type = 1;
						for (j = 0; j <= max_idr /*&& result_cnt <= result_max_cnt*/; j++)
						{
							if (v->frame_idx == surf_key_des[j]
								&& surf_map[surf_key_des[j]] >= g_surf_match - 1
								)
							{
								/// 先将类型置为1
								v->type = 1;
								//v->diff = surf_map[surf_key_des[j]];
								v->diff = 0;
								
								/// match face seeta 
								char outpath[1024];
								memset(outpath, 0, 1024);
								sprintf(outpath, "%sdata/%s/images/%s/%s_%d.jpg", 
									getFilePath(), program, strFileName.c_str(), strFileName.c_str(), v->frame_idx);
								char labelpath[1024];
								memset(labelpath, 0, 1024);
								if (g_vec_seeta_label_list.size() == 0)
								{	
									sprintf(labelpath, "/usr/local/bin/data/seeta/%s", 
										(char*)g_program_type.c_str());
								}
								else if (g_vec_seeta_label_list.size() > 0)
								{
									sprintf(labelpath, "%sdata/%s/seeta", 
										getFilePath(), strProgram.c_str());
								}
								double conf = facematch_seeta(outpath, (char*)strFileName.c_str(), labelpath);
								printf("[%s]sim - %f\n", outpath, conf);
								face_match_cnt++;
								if (conf > 0)
								{
									/// 通过人脸识别的关键帧类型置为2，即通过此步骤筛选
									v->type = 2;
									face_count++;
									v->diff = conf;
									printf("scene matched[%s]conf - %f\n", outpath, conf);
								}

								break;
							}
						}
					}
					percent(70 + (float(i)/float(points.size())*40) / 2);
				}
			}
			
		}
		else if (g_scene_type == "caffe")
		{
			/// 并行执行caffe场景匹配，
			/// 并根据匹配的类别筛选关键帧
			typedef map<int,int>::iterator IT;
			/// 并行执行caffe场景匹配，返回map，key为关键帧索引，value为匹配的类别
			caffe_map = caffe_detect_mutiple(4, (char*)strProgram.c_str(), (char*)strFileName.c_str());
			
			/// 初始化一个大小为map长度的数组，保存排序后的关键帧索引
			int caffe_key_des[caffe_map.size()];
			
			/// 打印原始排序的关键帧特征点信息日志，按帧索引排序
			int i, j, tmp;
			i = 0;
			int scene_cnt = 0;
			for(IT it = caffe_map.begin(); it != caffe_map.end(); it++) 
			{
				printf("map[%d] - caffe = %d\n", it->first, caffe_map[it->first]);
				caffe_key_des[i] = it->first;
				if (caffe_map[it->first] == 2)
					scene_cnt++;
				i++;
			}
			
			printf("scene_cnt - %d\n", scene_cnt);
			/// 根据场景匹配的结果，选择类型为2（类型2为演播室场景）的进行人脸识别
			int face_match_cnt = 0;
			for (i = 0; i < points.size(); i++)
			{
				v = &points[i];
				if (v->type == 2)
				{
					v->type = 1;
					for (j = 0; j < caffe_map.size(); j++)
					{
						if (v->frame_idx == caffe_key_des[j]
							&& caffe_map[caffe_key_des[j]] == 2)
						{
							/// 先将类型置为1
							v->type = 1;
							//v->diff = caffe_map[caffe_key_des[j]];
							v->diff = 0;
							
							/// match face seeta 
							char outpath[1024];
							memset(outpath, 0, 1024);
							sprintf(outpath, "%sdata/%s/images/%s/%s_%d.jpg", 
								getFilePath(), program, strFileName.c_str(), strFileName.c_str(), v->frame_idx);
							char labelpath[1024];
							memset(labelpath, 0, 1024);
							if (g_vec_seeta_label_list.size() == 0)
							{	
								sprintf(labelpath, "/usr/local/bin/data/seeta/%s", 
									(char*)g_program_type.c_str());
							}
							else if (g_vec_seeta_label_list.size() > 0)
							{
								sprintf(labelpath, "%sdata/%s/seeta", 
									getFilePath(), strProgram.c_str());
							}
							double conf = facematch_seeta(outpath, (char*)strFileName.c_str(), labelpath);
							printf("[%s]sim - %f\n", outpath, conf);
							face_match_cnt++;
							if (conf > 0)
							{
								/// 通过人脸识别的关键帧类型置为2，即通过此步骤筛选
								v->type = 2;
								face_count++;
								v->diff = conf;
								printf("scene matched[%s]conf - %f\n", outpath, conf);
							}

							break;
						}
					}
				}
			}
		}
		
		/// 遍历视频回零
		capture.set(CV_CAP_PROP_POS_FRAMES, 0);
		diff1 = 30, diff2 = 30, diff3 = 30;
		nn = -1, idx = 0;
		v = &points[idx];
    }
	printf("after scene match, face_cnt - %d\n", face_count);
	face_count = 0;
		
    if (hist_err != NULL) {
        free(hist_err);
        hist_err = NULL;
    }

	char buf[64];
    unsigned int points_len = points.size();
	int pre_type = -1;
	int diff = 30;
	
	/// 第5步：筛选关键帧，当两个关键帧间隔时间小于指定时间时，
	/// 仅保留首个关键帧，去除后面的一个或多个关键帧，即将类型置为1
	printf("\n\ndelte frame that too close step, %d points\n\n", points.size());
	for (int nn=0; nn<points_len; nn++)
	{
		sp_t *v = &points[nn];

		/// points中类型为2的可通过
		if (v->type == 0 || v->type == 100)
			continue;
		if (with_less && v->type == 1)
			continue;
		diff = v->diff;

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
						printf("skip - [%d]%.3f - %f\n", v_next->frame_idx, double(v_next->frame_idx)/double(fps), v_next->diff);
					}
					else
					{
						v->type = 1;
						printf("skip - [%d]%.3f - %f\n", v->frame_idx, double(v->frame_idx)/double(fps), v->diff);
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

		printf("[%d]%.3f - %f\n", v->frame_idx, double(v->frame_idx)/double(fps), v->diff);
		pre_type = v->type;
	}
	
	
	/// 第6步：去除从演播室切出的淡出淡入帧
	/// 遍历视频回零
	printf("\n\ndelete shadow in/out frames step, %d points\n\n", points.size());
	capture.set(CV_CAP_PROP_POS_FRAMES, 0);
	Mat frame;
	int nn = -1, idx = 0;
	sp_t* _v = &points[idx];
	while (capture.read(frame)) 
	{
		nn++;
		
		/// 仅遍历points中有的关键帧
		if ((int)(_v->frame_idx + g_delay_scenecut * (double)fps) != nn)
			continue;

		/// 保存关键帧截图至本地
		if (_v->type == 2)
		{
			char outpath[1024];
			memset(outpath, 0, 1024);
			sprintf(outpath, "%sdata/%s/images/%s_surf/%s_%d.jpg", getFilePath(), strProgram.c_str(), strFileName.c_str(), strFileName.c_str(), nn);
			printf("imwrite - %s\n", outpath);
			imwrite(outpath, frame);
		}
		
		if (++idx == points.size())
			break;
		_v = &points[idx];
		//percent(90 + (float(idx)/float(points.size())*6));
	}
	for (int i=0; i<points_len; i++)
	{
		sp_t* _v = &points[i];
		if (_v->type == 1)
			continue;
		
		_v->type = 1;

		/// match face seeta 
		char outpath1[1024];
		memset(outpath1, 0, 1024);
		sprintf(outpath1, "%sdata/%s/images/%s_surf/%s_%d.jpg", 
			getFilePath(), strProgram.c_str(), strFileName.c_str(), strFileName.c_str(), (int)(_v->frame_idx + g_delay_scenecut * (double)fps));
		printf("imread - %s\n", outpath1);
		
		char labelpath[1024];
		memset(labelpath, 0, 1024);
		if (g_vec_seeta_label_list.size() == 0)
		{	
			sprintf(labelpath, "/usr/local/bin/data/seeta/%s", 
				(char*)g_program_type.c_str());
		}
		else if (g_vec_seeta_label_list.size() > 0)
		{
			sprintf(labelpath, "%sdata/%s/seeta", 
				getFilePath(), strProgram.c_str());
		}
		double conf = facematch_seeta(outpath1, (char*)strFileName.c_str(), labelpath);
		printf("[%s]sim1 - %f\n", outpath1, conf);
		if (conf > 0)
		{
			/// 通过人脸识别的关键帧类型置为2，即通过此步骤筛选
			_v->type = 2;
			face_count++;
			printf("face matched[%s]conf - %f\n", outpath1, conf);
		}
	}
	printf("after face match, face_cnt - %d\n", face_count);

	/// 第7步：截取结果关键帧图像，上传关键帧截图
	printf("\n\nupload image step, %d points\n\n", points.size());
	capture.set(CV_CAP_PROP_POS_FRAMES, 0);
	Mat mat_sc;
	ret = 0;
	int error = 0;
	nn = -1, idx = 0;
	_v = &points[idx];
	
	memset(cmd, 0, 1024);
	sprintf(cmd, "mkdir %sdata/%s/images/%s_result/", getFilePath(), strProgram.c_str(), strFileName.c_str());
	system(cmd);
	
	if (g_res_upload_url == "")
		printf("upload is null.\n");
	SendFile send(g_res_upload_url);
	send.InitSendFile();
	while (capture.read(frame) && g_res_upload_url != "") 
	{
		nn++;
		
		/// 仅遍历points中有的关键帧
		if (_v->frame_idx != nn &&
			(time_start != nn && time_end != nn))
			continue;

		/// 保存关键帧截图至本地
		if (_v->type == 2 ||
			(time_start == nn || time_end == nn))
		{
			if (width > 0 && height > 0)
			{
				g_res_height = ((double)height / (double)width) * (double)g_res_width;
			}
			
			resize(frame, mat_sc, Size(g_res_width, g_res_height));
			char outpath[1024];
			memset(outpath, 0, 1024);
			sprintf(outpath, "%sdata/%s/images/%s_result/task%s_%d.jpg", getFilePath(), strProgram.c_str(), strFileName.c_str(), g_program.c_str(), nn);
			imwrite(outpath, mat_sc);
			
			string fileName = outpath;
			fileName = fileName.substr(fileName.find_last_of('/') + 1);
			
			/// 上传关键帧截图
			ret = send.MySend(outpath, (char*)fileName.c_str());
			if (!ret)
			{
				printf("[%s]finished upload - %s\n", outpath, g_res_upload_url.c_str());
			}
			else
			{
				printf("[%s]upload error - %s\n", outpath, g_res_upload_url.c_str());
				error++;
			}
		}
		
		if (++idx == points.size())
			break;
		_v = &points[idx];
		percent(90 + (float(idx)/float(points.size())*6));
	}
	if (error > 0)
		ret = -1;
	
	/// 第8步：删除当前缓存视频关键帧截图目录
	if (!g_shotimage)
	{
		memset(cmd, 0, 1024);
		sprintf(cmd, "rm -rf %sdata/%s/images/%s_result/", getFilePath(), strProgram.c_str(), strFileName.c_str());
		system(cmd);
		memset(cmd, 0, 1024);
		sprintf(cmd, "rm -rf %sdata/%s/images/%s/", getFilePath(), strProgram.c_str(), strFileName.c_str());
		system(cmd);
		memset(cmd, 0, 1024);
		sprintf(cmd, "rm -rf %sdata/%s/images/%s_keyframe/", getFilePath(), strProgram.c_str(), strFileName.c_str());
		system(cmd);
		memset(cmd, 0, 1024);
		sprintf(cmd, "rm -rf %sdata/%s/images/%s_surf/", getFilePath(), strProgram.c_str(), strFileName.c_str());
		system(cmd);
	}
	
	/// 删除当前关键帧进行人脸识别的缓存目录
	if (g_vec_seeta_label_list.size() == 0)
	{
		//find data/tmp/ -type d | grep -v tmp | grep -v label_faces | grep -v test_face_recognizer | xargs rm -rf
		memset(cmd, 0, 1024);
		sprintf(cmd, "rm -rf /usr/local/bin/data/seeta/%s/%s", (char*)g_program_type.c_str(), (char*)strFileName.c_str());
		system(cmd);
	}
	else if (g_vec_seeta_label_list.size() > 0)
	{
		memset(cmd, 0, 1024);
		sprintf(cmd, "rm -rf %sdata/%s/seeta/%s", 
			getFilePath(), (char*)strProgram.c_str(), (char*)strFileName.c_str());
		system(cmd);
	}
	
	capture.release();
	
    /// 第9步：结果写入文件
	printf("\n\nwrite to output file, %d points\n\n", points.size());
    FILE *fp_out = NULL;  
    if (strcmp(output, "") != 0) 
	{
        fp_out = fopen(output, "w");
        if (fp_out == NULL) {
            sprintf(msg, "open file failed!\n");
            LOG5CXX_FATAL(msg, 121);
        }
    }
	
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

		if (with_less && v->type == 1)
			continue;
				
//		if (diff <= face_cnt)
//		{
			/// 写入结果集时，当前关键帧要偏移指定数量的帧索引
			/// 调整当前帧索引，保证web页面显示时的正确定位
			if (v->frame_idx + g_time_offset > numberframes)
				g_time_offset = numberframes - v->frame_idx;
			printf("[%d]%.3f - %f\n", v->frame_idx + g_time_offset, double(v->frame_idx + g_time_offset)/double(fps), v->diff);
			sprintf(buf, "%d %.3f %d;\n", v->frame_idx + g_time_offset, double(v->frame_idx + g_time_offset)/double(fps), v->type-1);
			fwrite(buf, strlen(buf), 1, fp_out);
//		}
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
	
	if (!g_shotimage)
	{
		memset(cmd, 0, 1024);
		sprintf(cmd, "rm -rf %sdata/%s", getFilePath(), strProgram.c_str());
		system(cmd);
	}
	
	return ret;
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

static char *optstring = "a:A:b:B:c:C:d:D:e:E:f:F:g:G:H:I:j:J:k:K:t:T:l:L:m:M:n:N:o";
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
	{"seeta_label_list",1, NULL, 'T'},
	{"res_upload_url",	1, NULL, 'l'},
	{"res_size",		1, NULL, 'L'},
	{"hist_thresh",		1, NULL, 'm'},
	{"delay_scenecut",	1, NULL, 'M'},
	{"scene_type",		1, NULL, 'n'},
	{"shotimage",  		0, NULL, 'N'},
	{"hist_range",      1, NULL, 'o'},
    {0, 0, 0, 0}  
};
int cmd_shotdetect(int argc, char **argv)
{
    const int max_path = MAX_PATH;

    char _video[max_path], _output[max_path], _pkl[max_path], \
	_program[max_path], _vheads[max_path], _vtails[max_path], \
	_scene[max_path], _seeta_label_list[max_path], _program_type[max_path], \
	_res_upload_url[max_path], _res_size[max_path], _scene_type[max_path], \
	_hist_range[max_path];
    strcpy(_video, ""); strcpy(_output, ""); strcpy(_pkl, ""); 
	strcpy(_program, ""); strcpy(_vheads, ""); strcpy(_vtails, ""); 
	strcpy(_scene, ""); strcpy(_seeta_label_list, ""); strcpy(_program_type, "");
	strcpy(_res_upload_url, ""); strcpy(_res_size, ""); strcpy(_scene_type, ""); 
	strcpy(_hist_range, ""); 
    bool with_face = true, with_voice = false, with_less = true, \
	just_test = false, with_facematch = true, shotimage = false;
    float db_thresh = _DB_THRESH_DEFAULT_;
	float hist_thresh = _HIST_THRESH_DEFAULT_;
	float ssim_simular = _SSIM_SIMULAR_DEFAULT_;
	float prog_duration = _PROG_DURATION_DEFAULT_;
	float delay_scenecut = 0.0;
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
			case 'T':{ strcpy(_seeta_label_list, optarg); break; }
			case 'l':{ strcpy(_res_upload_url, optarg); break; }
			case 'L':{ strcpy(_res_size, optarg); break; }
			case 'm':{ hist_thresh = atof(optarg); break; }
			case 'M':{ delay_scenecut = atof(optarg); break; }
			case 'n':{ strcpy(_scene_type, optarg); break; }
			case 'N':{ shotimage = true; break; }
			case 'o':{ strcpy(_hist_range, optarg); break; }
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
		else
		{
			g_program = _program;
		}
		
		/// program_type
		if (strcmp(_program_type, "") != 0)
		{
			g_program_type = _program_type;
		}
		
		/// seeta_label_list
		if (strcmp(_seeta_label_list, "") != 0)
        {
			string str = _seeta_label_list;
			vector<string> _vec_seeta_label_list = split(str, ",");
			for (int i = 0; i < _vec_seeta_label_list.size(); i++)
			{
				printf("seeta_label_list - %s\n", _vec_seeta_label_list[i].c_str());
				string strUrl = _vec_seeta_label_list[i];
				if (strUrl.find("\"") != std::string::npos)
					strUrl = strUrl.substr(1, strUrl.length() - 2);
				printf("url - %s\n", strUrl.c_str());
				
				g_vec_seeta_label_list.push_back(strUrl);
			}
		}
		
		/// res_upload_url
		if (strcmp(_res_upload_url, "") != 0)
		{
			g_res_upload_url = _res_upload_url;
		}
		
		/// res_size
		if (strcmp(_res_size, "") != 0)
		{
			string str = _res_size;
			vector<string> _vec_res_size = split(str, ",");
			if (_vec_res_size.size() == 2)
			{
				g_res_width = atoi(_vec_res_size[0].c_str());
				g_res_height = atoi(_vec_res_size[1].c_str());
			}
		}
		
		/// hist_thresh
		if (hist_thresh > 0.0)
		{
			g_hist_thresh = hist_thresh;
		}
		
		/// delay_scenecut
		if (delay_scenecut > 0.0)
		{
			g_delay_scenecut = delay_scenecut;
		}
		
		/// _scene_type
		if (strcmp(_scene_type, "") != 0)
		{
			g_scene_type = _scene_type;
		}
		
		if (shotimage == true)
			g_shotimage = 1;
			
		/// hist_range
		g_histrange.X = 0;
		g_histrange.Y = 0;
		g_histrange.W_scale = 1.0;
		g_histrange.H_scale = 1.0;
		printf("hist_range(default) X - %d, Y - %d, W_scale - %f, H_scale - %f\n", 
			g_histrange.X, g_histrange.Y, g_histrange.W_scale, g_histrange.H_scale);
		if (strcmp(_hist_range, "") != 0)
        {
			printf("hist_range - %s\n", _hist_range);
			string str = _hist_range;
			vector<string> _vec_hist_range = split(str, ",");
			for (int i = 0; i < _vec_hist_range.size(); i++)
			{
				printf("hist_range - %s\n", _vec_hist_range[i].c_str());
				if (i == 0)
				{
					string strX = _vec_hist_range[i];
					g_histrange.X = atoi(strX.c_str());
					printf("X - %d\n", g_histrange.X);
				}
				else if (i == 1)
				{
					string strY = _vec_hist_range[i];
					g_histrange.Y = atoi(strY.c_str());
					printf("Y - %d\n", g_histrange.Y);
				}
				else if (i == 2)
				{
					string strW = _vec_hist_range[i];
					g_histrange.W_scale = atof(strW.c_str());
					printf("W_scale - %f\n", g_histrange.W_scale);
				}
				else if (i == 3)
				{
					string strH = _vec_hist_range[i];
					g_histrange.H_scale = atof(strH.c_str());
					printf("H_scale - %f\n", g_histrange.H_scale);
				}
				printf("\n");
			}
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
		printf("scene_type - %s\n", _scene_type);
		printf("surf_match - %d\n", surf_match);
		printf("pkl - %s\n", _pkl);
		printf("vheads - %s\n", _vheads);
		printf("vtails - %s\n", _vtails);
		printf("scene - %s\n", _scene);
		printf("seeta_label_list - %s\n", _seeta_label_list);
		printf("ssim_simular - %f\n", ssim_simular);
		printf("max_mapsurf_idr - %d\n", g_max_mapsurf_idr);
		printf("max_result_cnt - %d\n", g_max_result_cnt);
		printf("time_offset - %d\n", g_time_offset);
		printf("key_interval - %f\n", g_key_interval);
		printf("test - %s\n", just_test);
		printf("res_upload_url - %s\n", _res_upload_url);
		printf("shotimage - %d\n", g_shotimage);

        int ret = video_shotdetect(
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
        return ret;
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
 -T/--seeta_label_list <string> Seeta label list - image url.\n\
 -l/--res_upload_url	The result's callback url.\n\
 -L/--res_size	The result's size.\n\
 -m/--hist_thresh	The hist's value.\n\
 -M/--delay_scenecut	The delay of scenecut.\n\
 -n/--scene_type	The type of scene detection.\n\
 -N/--shotimage	Whether save the shot image.\n\
 -o/--hist_range <string> The rect of hist range.\n\
");
}
