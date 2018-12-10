#ifndef FFCLIPS_H_
#define FFCLIPS_H_

#define __STDC_CONSTANT_MACROS

#include <stdint.h>
#include <pthread.h>
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
#include <libavutil/avutil.h>
#include <libavutil/mem.h>
#include <libavutil/mathematics.h>
}


#include <list>
using namespace std;

typedef struct{
	list<AVPacket>* lspacket;
	int64_t firsttm;
	AVCodecParserContext* pparser;
}streamcontext;

typedef struct{
	char inputfile[512];
	char outputfile[512];
	char cuttype[10];
	int clipcount;
	int64_t clipblock; //time or byte
	int64_t filesize;
	int clipindex;
	AVFormatContext* pinformatctx;
	AVFormatContext* poutformatctx;
	streamcontext** ppstreamsctx;
	int defstreamidx;
	int64_t curtm;
	int64_t lasttm;
	int64_t clipstarttm;
	int64_t defstreamst;
	int mintime;// second
	volatile int quit_signal;
	volatile int pprogress;
	volatile int iseof;
    pthread_mutex_t mutexevent;
}clipscontext;

void clips_log1(void* ptr, int level, const char* lformat, va_list vl);

void clips_log2(int level,const char* lformat,...);

int clips_init(clipscontext* pcontext);

void clips_uninit(clipscontext* pcontext);

int clips_cutsize(clipscontext* pcontext);

int clips_cuttime(clipscontext* pcontext);

int clips_virtualcut(clipscontext* pcontext);

#endif
