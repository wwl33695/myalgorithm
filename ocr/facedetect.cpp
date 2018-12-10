#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "types.h"
#include "log5cxx.h"

using namespace std;
using namespace cv;
    
const static Scalar colors[] =  { CV_RGB(0,0,255),
    CV_RGB(0,128,255),
    CV_RGB(0,255,255),
    CV_RGB(0,255,0),
    CV_RGB(255,128,0),
    CV_RGB(255,255,0),
    CV_RGB(255,0,0),
    CV_RGB(255,0,255)};
static int _detect_(Mat &img, CascadeClassifier &cascade, double scale)
{
    int i = 0;
    vector<Rect> faces;
    scale = 240.0/img.rows;
    if (scale > 1.0)
        scale = 1.0;
    Mat gray, smallImg(cvRound(img.rows*scale), cvRound(img.cols*scale), CV_8UC1);

    cvtColor(img, gray, CV_BGR2GRAY);
    resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
    //equalizeHist(smallImg, smallImg);

    cascade.detectMultiScale(smallImg, faces,
        1.1, 2, 0
        //|CV_HAAR_FIND_BIGGEST_OBJECT
        //|CV_HAAR_DO_ROUGH_SEARCH
        |CV_HAAR_SCALE_IMAGE
        ,
        Size(30, 30));
    
    return faces.size();


    int face_count = 0;
    for(vector<Rect>::const_iterator r = faces.begin(); r!=faces.end(); r++, i++) {
        Point center;
        Scalar color = colors[i%8];
        int radius;

        double aspect_ratio = (double)r->width/r->height;
        if (0.75 < aspect_ratio && aspect_ratio < 1.3) {
            center.x = cvRound((r->x + r->width*0.5)*scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);
            radius = cvRound((r->width + r->height)*0.25*scale);
            circle(img, center, radius, color, 3, 8, 0);
        }
        else {
            rectangle(img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                       cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                       color, 3, 8, 0);
        }
        face_count += 1;
    }
    return face_count;
}

static CascadeClassifier cascade;
static bool flag = false;
//static std::string cascade_train_content = "";
static char pwd[4096];
static const char* getFilterPath(){
#ifdef WIN32
    getcwd(pwd, MAX_PATH);
#else
    readlink("/proc/self/exe", pwd, MAX_PATH);
    int len = strlen(pwd);
    while(1){
        if(pwd[len-1] != '/')
            pwd[len-1] = '\0';
        else
            break;
        len--;
    }
#endif
    //printf("The current directory is: %s", buffer);
    return pwd;
}

extern "C" int facedetect(Mat &mat, int *x, int *y)
{
    if (!flag) {
        const char *cur_pwd = getFilterPath();
        string cascadeName = string(cur_pwd) + "/fd.xml"; //"/home/niu/imchar/Source/fd.xml";
            
        char msg[2048];
        try {
            //FileStorage fs(cascade_train_content, FileStorage::READ | FileStorage::MEMORY);
            //if (!cascade.read(fs.getFirstTopLevelNode())) {

            if (!cascade.load(cascadeName)) {
                sprintf(msg, "Load cascade error(%s).", cascadeName.c_str());
                LOG5CXX_INFO(msg);
                return 0;
            }
        } catch (...) {
            sprintf(msg, "Load cascade exception(%s).", cascadeName.c_str());
            LOG5CXX_INFO(msg);
            return 0;
        }
        flag = true;
    }
    int face_count = _detect_(mat, cascade, 1.0);

    return face_count;
}
