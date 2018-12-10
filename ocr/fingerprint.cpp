#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <opencv2/opencv.hpp>
#ifdef linux
#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <signal.h>
#include <getopt.h> 
#endif
#include "types.h"
#include "fingerprint_alg.h"
#include "log5cxx.h"

using namespace std;
using namespace cv;

// 二进制位权值
static const unsigned int _bit_[8] = {128,64,32,16,8,4,2,1};

// 单字节数的二进制表示中有几个1
static const unsigned int _bit0_[256] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8};

#define _bit_mul_(idx,offset) (_bit_[idx]*(unsigned int)(bytes[offset+idx]))

// 字符串转字节   "0" "0" "0" "0" "0" "0" "1" "0" ==> 0x00000010
static inline void _byte_to_bits_(const char *bytes, unsigned char *bits, unsigned int len) 
{
    assert(len%8 == 0);

    unsigned int v;
    for (int i=0; i<len; i+=8) {
        int j = i/8;
        v = _bit_mul_(0,i) + _bit_mul_(1,i) + _bit_mul_(2,i) + _bit_mul_(3,i) +  _bit_mul_(4,i) + _bit_mul_(5,i) + _bit_mul_(6,i) + _bit_mul_(7,i);
        bits[j] = v;

        /*for (int k=0; k<8; k++)
            printf("%d", bytes[i+k]);
        printf("--> %x(%d)\n", (unsigned char)v, j);*/
    }
}

/* 
 * 计算图片指纹
 */
static void calc_image_fingerprint(const char *path, const char *output, bool shrink)
{
    im_t *img = load_image(path);
    
    fp_t fp;
    calc_fingerprint_phash(&fp, img);

    release_image(img);

    if (strcmp(output, "") != 0) {
        FILE *fp_out = NULL;
        fp_out = fopen(output, "w");
        if (fp_out == NULL)
            return;

        if (!shrink) {
            char buf[8];
            for (int i=0; i<fp.len; i++) {
                sprintf(buf, "%d ", fp.fp[i]);
                fwrite(buf, strlen(buf), 1, fp_out);
            }
        } else {
            unsigned char bits[8];
            _byte_to_bits_(fp.fp, bits, 64);
            fwrite(bits, 8, 1, fp_out);
        }

        fclose(fp_out);
    } else {
        printf("fingerprint: ");
        for (int i=0; i<fp.len; i++)
            printf("%d ", fp.fp[i]);
        printf("\n");
    }
}

/* 
 * 计算视频指纹
 */
static void calc_video_fingerprint(const char *path, const char *output, bool shrink)
{
    clock_t s, e;
    FILE *fp_out = NULL;    
    if (strcmp(output, "") != 0) {
        fp_out = fopen(output, "w");
        if (fp_out == NULL)
            return;
    }

    VideoCapture capture;  
    capture.open(path);
    if(!capture.isOpened()) {
        printf("read video failed!(%s)\n", path); 
        return;
    }

    s = clock();
    im_t img; fp_t fp;
    Mat frame, gray;
    while(capture.read(frame)) {  
        cvtColor(frame, gray, CV_BGR2GRAY);

        img.rows = gray.rows;
        img.cols = gray.cols;
        img.channels = gray.channels();
        img.depth = gray.depth();
        img.data = gray.data;
        //img.data = malloc(img.rows*img.cols*img.channels);
        //memcpy(img.data, gray.data, img.rows*img.cols*img.channels);
        //imwrite("/home/niu/video_search/1.jpg", gray);

        calc_fingerprint_phash(&fp, &img);
    
        //free(img.data);

        if (fp_out) {
            if (!shrink) {
                char buf[8];
                for (int i=0; i<fp.len; i++) {
                    sprintf(buf, "%d ", fp.fp[i]);
                    fwrite(buf, strlen(buf), 1, fp_out);
                }
                fwrite("\n", 1, 1, fp_out);
            } else {
                // 使用字节方式
                unsigned char bits[8];
                _byte_to_bits_(fp.fp, bits, 64);
                fwrite(bits, 8, 1, fp_out);
            }
        }
    }

    if (fp_out) {
        fclose(fp_out);
        fp_out = NULL;
    }
    e = clock();
    printf("TIME USED: [%.4f s]\n", double(e-s)/CLOCKS_PER_SEC);
}

/* 
 * 对比指纹文件
 */
static void compare_image(const char *path1, const char *path2)
{
    im_t *img1 = load_image(path1);
    im_t *img2 = load_image(path2);
    
    fp_t fp1, fp2;
    calc_fingerprint_phash(&fp1, img1);
    calc_fingerprint_phash(&fp2, img2);

    release_image(img1);
    release_image(img2);

    int diff = hanming_distance(&fp1, &fp2);
    printf("diff: %d (bigger than 5 could be treat different)\n", diff);
}


// 使用字节方式，在对比2个64×1向量时，即4个字节做异或（xor）运算，然后分别查表相加，共4次异或4此查表4此求和运算。
static inline unsigned int _feature_distance_(const char *f1, const char *f2, int bytes)
{
    unsigned char r;
    int idx;
    unsigned int err_sum = 0;
    for (int i=0; i<bytes; i++) {
        r = f1[i]^f2[i];
        idx = r;
        err_sum += _bit0_[idx];
    }
    return err_sum;
}

/* 
 * 在指定指纹文件中搜索
 * 视频搜索策略：从待搜索样本中找出3个64×1向量数据，先搜索这3帧数据是否能匹配，再从加入更多数据进行精确匹配，最终得出匹配结果。
 */
static void search_target(const char *path, const char *list, const char *output, int target_length_user)
{
    // 1. 获取目标/搜索列表特征文件长度，计算有多少特特征，得出需要匹配多少次
    // 2. 读取目标特征到内存中，搜索特征分段读取（一段为4M）
    // 3. 循环匹配，得出匹配度，并写入内存或存储

    clock_t s, e;
    char msg[4096];

    // 1.
    FILE *fp_target = NULL, *fp_list = NULL;
    fp_target = fopen(path, "r");
    fp_list = fopen(list, "r");

    if (fp_target == NULL || fp_list == NULL) {
        LOG5CXX_FATAL("file open error", 210);
    }

    struct stat st_target, st_list;
    stat(path, &st_target);
    stat(list, &st_list);
    unsigned int target_len = st_target.st_size;
    unsigned int list_len = st_list.st_size;
    if (target_len%8 != 0 || list_len%8 != 0) {
        LOG5CXX_FATAL("feature file length not equal to n*8", 10);
    }
    if (target_len > target_length_user*8)      // 如果特征比用户输入的要长，则截断至用户要求长度
        target_len = target_length_user*8;
    unsigned int target_feature_len = target_len/8;
    unsigned int list_feature_len = list_len/8;
    unsigned int feature_calc_len = list_feature_len - target_feature_len + 1;
    if (list_feature_len*8 > 1*1024*1024*1024) {
        LOG5CXX_FATAL("search feature length bigger than 1GB", 10);
    }

    // 2.
    char *target_feature = (char*)malloc(target_feature_len*8); // 开辟目标特征空间
    if (target_feature == NULL) {
        sprintf(msg, "Memory malloc or memcpy failed, operate memory size is [%d]=[%d MB].", target_feature_len*8, target_feature_len*8/1024/1024);
        LOG5CXX_FATAL(msg, 10);
    }
    char *list_feature = (char*)malloc(list_feature_len*8);
    if (target_feature == NULL) {
        sprintf(msg, "Memory malloc or memcpy failed, operate memory size is [%d]=[%d MB].", list_feature_len*8, list_feature_len*8/1024/1024);
        LOG5CXX_FATAL(msg, 10);
    }

    char buffer[8];
    size_t byteread, offset = 0;
    while (!feof(fp_target)) {
        byteread = fread(buffer, 1, 8, fp_target);
        memcpy(target_feature+offset, buffer, byteread);
        offset += byteread;

        if (offset > target_feature_len*8)
            break;
    }
    fclose(fp_target);
    offset = 0;
    while (!feof(fp_list)) {
        byteread = fread(buffer, 1, 8, fp_list);
        memcpy(list_feature+offset, buffer, byteread);
        offset += byteread;
    }
    fclose(fp_list);


    // 3.
    s = clock();
    unsigned char *distance = (unsigned char*)malloc(feature_calc_len);
    int feature_offset = 8;
    char *list_feature_ptr;
    for (int i=0; i<feature_calc_len; i++) {
        list_feature_ptr = list_feature + i*feature_offset;

        int dd = _feature_distance_(target_feature, list_feature_ptr, feature_offset*target_feature_len);
        dd = float(dd)/float(target_feature_len);
        distance[i] = dd;
    }
    if (target_feature) {
        free(target_feature);
        target_feature = NULL;
    }
    if (list_feature) {
        free(list_feature);
        list_feature = NULL;
    }
    e = clock();
    printf("TIME USED: [%.4f s]\n", double(e-s)/CLOCKS_PER_SEC);

    // 4.
    FILE *fp_output = fopen(output, "w+");
    if (fp_output == NULL) {
        LOG5CXX_FATAL("file open error", 210);
    }
    //size_t bytes_write = fwrite()
    for (int i=0; i<feature_calc_len; i++) {
        if (i == feature_calc_len-1)
            fprintf(fp_output, "%d", distance[i]);
        else
            fprintf(fp_output, "%d, ", distance[i]);
    }
    fclose(fp_output);
}


static char *optstring = "a:A:b:B:c:Cd:D:e:";
static struct option long_options[] = {         //  no_argument--0,required_argument--1,optional_argument--2
    {"image",           1, NULL, 'a'},
    {"video",           1, NULL, 'A'},
    {"output",          1, NULL, 'b'},
    {"compare_1",       1, NULL, 'B'},
    {"compare_2",       1, NULL, 'c'},
    {"shrink_output",   0, NULL, 'C'},
    {"search",          1, NULL, 'd'},
    {"target",          1, NULL, 'D'},
    {"target_len",      1, NULL, 'e'},
    {0, 0, 0, 0}  
};
int cmd_fingerprint(int argc, char **argv)
{
    const int max_path = MAX_PATH;

    char _image[max_path], _video[max_path], _output[max_path], _compare1[max_path], _compare2[max_path], _search[max_path], _target[max_path];
    strcpy(_image, ""); strcpy(_video, ""); strcpy(_output, ""); strcpy(_compare1, ""); strcpy(_compare2, ""); strcpy(_search, ""); strcpy(_target, "");
    bool shrink = false;
    int target_len = -1;

    int opt, option_index = 0;
    while ((opt = getopt_long(argc, argv, optstring, long_options, &option_index)) != -1) {

        switch(opt){
            case 'a':{ strcpy(_image, optarg); break; }
            case 'A':{ strcpy(_video, optarg); break; }
            case 'b':{ strcpy(_output, optarg); break; }
            case 'B':{ strcpy(_compare1, optarg); break; }
            case 'c':{ strcpy(_compare2, optarg); break; }
            case 'C':{ shrink = true; break; }
            case 'd':{ strcpy(_search, optarg); break; }
            case 'D':{ strcpy(_target, optarg); break; }
            case 'e':{ target_len = atoi(_search); break; }
        }
    }

    // 计算图像指纹
    if (strcmp(_image, "") != 0) {
        calc_image_fingerprint(_image, _output, shrink);
        return 0;
    }

    // 计算视频指纹
    if (strcmp(_video, "") != 0) {
        calc_video_fingerprint(_video, _output, shrink);
        return 0;
    }

    // 对比两个图像
    if (strcmp(_compare1, "") != 0) {
        compare_image(_compare1, _compare2);
        return 0;
    }

    // 在指纹文件中搜索
    if (strcmp(_search, "") != 0) {
        search_target(_target, _search, _output, target_len);
        return 0;
    }

    LOG5CXX_FATAL("fatal: argument error.", 1);
    return 0;
}

void cmd_fingerprint_usage()
{   
    printf("\
fingerprint:\n\
 -a/--image <file>  Image file full path\n\
 -A/--video <file>  Video file full path\n\
 -b/--output <file> Output result file path\n\
 -B/--compare_1 <file> Compare A image\n\
 -c/--compare_2 <file> Compare B image\n\
 -C/--shrink_output Output file in shrink model\n\
 -d/--search <file> Search files\n\
 -D/--target <file> Search target\n\
 -e/--target_len <int> Search front 'target_len' of target\n\
");
}