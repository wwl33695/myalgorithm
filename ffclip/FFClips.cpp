#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <sys/stat.h>  
#include "FFClips.h"

#define MIN_DTS ((int64_t)(0x8000000000000000))
#define NDEBUG

AVRational AV_COMMON_BASE={1,AV_TIME_BASE};

inline int pthread_mutex_lock_experid(pthread_mutex_t* lock, int milesecond){
	if(pthread_mutex_trylock(lock) == 0)
		return 0;
	usleep(milesecond*1000);
	return pthread_mutex_trylock(lock);
}

static int rectifycliptime(clipscontext* pcontext){
	if(!pcontext->clipblock){
		AVFormatContext* inc=pcontext->pinformatctx;
		if(inc->duration>0 && pcontext->clipcount){
			pcontext->clipblock=inc->duration/pcontext->clipcount;
			return 0;
		}else{
			clips_log2(AV_LOG_ERROR,"clip size invalid\n");
			return -1;
		}
	}else{
		AVRational cliptb={1,1};
		pcontext->clipblock=av_rescale_q(pcontext->clipblock,cliptb,AV_COMMON_BASE);
		return 0;
	}
	return -1;
}

static int streams_init(clipscontext* pcontext){
	AVFormatContext* inc=pcontext->pinformatctx;
	streamcontext** ppstreams=(streamcontext**)malloc(inc->nb_streams*sizeof(streamcontext*));
	int defidx=-1;
	for(int n=0;n<inc->nb_streams;n++){
		streamcontext* pctx=(streamcontext*)malloc(sizeof(streamcontext));
		pctx->firsttm=MIN_DTS;
		pctx->lspacket=new list<AVPacket>();
		pctx->pparser=NULL;
		//pctx->pparser=av_parser_init(inc->streams[n]->codec->codec_id);
		ppstreams[n]=pctx;
		if(inc->streams[n]->codec->codec_type==AVMEDIA_TYPE_VIDEO && defidx<0){
			defidx=n;
		}
	}
	pcontext->defstreamidx=defidx;
	pcontext->ppstreamsctx=ppstreams;
	return 0;
}

static void streams_uinit(clipscontext* pcontext){
	if(pcontext->ppstreamsctx){
		for(int n=0;n<pcontext->pinformatctx->nb_streams;n++){
			streamcontext* pstreamctx=pcontext->ppstreamsctx[n];
			for(list<AVPacket>::iterator ite=pstreamctx->lspacket->begin();ite!=pstreamctx->lspacket->end();++ite){
				AVPacket& packet=*ite;
				av_free_packet(&packet);
			}
			delete pstreamctx->lspacket;
			if(pstreamctx->pparser)
				av_parser_close(pstreamctx->pparser);
			free(pstreamctx);
		}
		free(pcontext->ppstreamsctx);
		pcontext->ppstreamsctx=NULL;
	}
}

int read_packet_from_stream(clipscontext* pcontext,int idx,AVPacket* avpacket){
	if(idx<0)
		return 0;
	int ret=0;
	streamcontext* pstreamctx=pcontext->ppstreamsctx[idx];
	if(pstreamctx->lspacket->empty()){
		return 0;
	}else{
		AVPacket tmpacket;
		av_init_packet(&tmpacket);
		//tmpacket=*(pstreamctx->lspacket->begin());
		*avpacket=*(pstreamctx->lspacket->begin());
		pstreamctx->lspacket->pop_front();
		/*if (av_parser_change(pstreamctx->pparser, pcontext->poutformatctx->streams[tmpacket.stream_index]->codec, &(avpacket->data), &(avpacket->size), tmpacket.data, tmpacket.size, tmpacket.flags & AV_PKT_FLAG_KEY)) {
			avpacket->buf = av_buffer_create(avpacket->data, avpacket->size, av_buffer_default_free, NULL, 0);
			if (!avpacket->buf)
				exit(1);
		}else{
			*avpacket=tmpacket;
		}*/
		ret=1;
	}
	if(!pstreamctx->lspacket->empty()){
		AVPacket packet=*(pstreamctx->lspacket->begin());
		if(packet.dts>=0)
			pstreamctx->firsttm=av_rescale_q(packet.dts,pcontext->pinformatctx->streams[packet.stream_index]->time_base,AV_COMMON_BASE);
	}
	return ret;
}

int write_packet_into_stream(clipscontext* pcontext,AVPacket* avpacket){
	streamcontext* pstreamctx=pcontext->ppstreamsctx[avpacket->stream_index];
	if(pstreamctx->lspacket->empty())
		pstreamctx->firsttm=av_rescale_q(avpacket->dts,pcontext->pinformatctx->streams[avpacket->stream_index]->time_base,AV_COMMON_BASE);
	pstreamctx->lspacket->push_back(*avpacket);
/*	for(pcontext->)*/
	return 1;
}

int choose_ouput_packet(clipscontext* pcontext,AVPacket* avpacket){
	int idx=-1; //av_rescale_q(ost->st->cur_dts, ost->st->time_base,AV_TIME_BASE_Q);
	int64_t tmptm=0x7FFFFFFFFFFFFFFF;
	int ret=0;
	for(int n=0;n<pcontext->pinformatctx->nb_streams;n++){
		streamcontext* pstreamctx=pcontext->ppstreamsctx[n];
		 if(pstreamctx->firsttm<tmptm &&  !pstreamctx->lspacket->empty()){
			tmptm=pstreamctx->firsttm;
			idx=n;
		}
	}
	if(read_packet_from_stream(pcontext,idx,avpacket)){
		if(tmptm>pcontext->curtm){
			pcontext->curtm=tmptm;
			if(pcontext->clipstarttm==MIN_DTS)
				pcontext->clipstarttm=tmptm;
		}
		ret=1;
	}
	return ret;
}

int choose_ouput_packet_nondefault(clipscontext* pcontext,AVPacket* avpacket){
	int idx=-1;
	int64_t tmptm=0x7FFFFFFFFFFFFFFF;
	int ret=0;

	for(int n=0;n<pcontext->pinformatctx->nb_streams;n++){
		if(pcontext->defstreamidx==n)
			continue;
		streamcontext* pstreamctx=pcontext->ppstreamsctx[n];
		if(pstreamctx->firsttm<=pcontext->curtm && !pstreamctx->lspacket->empty()){
			idx=n;
			tmptm=pstreamctx->firsttm;
			break;
		}
	}
	if(read_packet_from_stream(pcontext,idx,avpacket)){
		if(tmptm>pcontext->curtm)
			pcontext->curtm=tmptm;
		ret=1;
	}
	
	return ret;
}

static int write_packet_ouput(clipscontext* pcontext,AVPacket* avpacket){
	if(avpacket->dts<=pcontext->poutformatctx->streams[avpacket->stream_index]->cur_dts){
		avpacket->pts=AV_NOPTS_VALUE;
		avpacket->dts =pcontext->poutformatctx->streams[avpacket->stream_index]->cur_dts+avpacket->duration;
	}
	if(av_interleaved_write_frame(pcontext->poutformatctx,avpacket)<0){
		clips_log2(AV_LOG_ERROR,"write output packet failed\n");
		return -1;
	}
	return 0;
}

void* input_thread(void* pparam){
	clipscontext* pcontext=(clipscontext*)pparam;
	AVPacket avpacket;
	while(av_read_frame(pcontext->pinformatctx,&avpacket)>=0 && !(pcontext->quit_signal)){
		//write_packet_ouput(pcontext,&avpacket);
		while(!write_packet_into_stream(pcontext,&avpacket) && !(pcontext->quit_signal)){
			sleep(10);
		}
		av_init_packet(&avpacket);
	}
	pcontext->iseof=1;
	return 0;
}

static inline char* __tcsrchr(char* ptr, char v){
	int i=0,len=strlen(ptr);
	while(i<len){
		if(ptr[i]==v)
			return ptr+i;
		i++;
	}
	return NULL;
}
static void appendclipname(clipscontext* pcontext,char* clipsfile){
	char tmpfile[512]={0};
	strcpy(tmpfile,pcontext->outputfile);
	char* pdot=(char*)__tcsrchr(tmpfile,'.');
	if(pdot==NULL){
		const char* psname=__tcsrchr(pcontext->inputfile,'\\');
		const char* pext=__tcsrchr(pcontext->inputfile,'.');
		sprintf(clipsfile,"%s\\%s%d%s",tmpfile,psname+1,pcontext->clipindex,pext+1);
	}else{
		*pdot='\0';
		const char* pext=__tcsrchr(pcontext->outputfile,'.');
		sprintf(clipsfile,"%s%d%s",tmpfile,pcontext->clipindex,pext);
	}
}

static void closeoutput(AVFormatContext** ppoutformatctx){
	if(ppoutformatctx && *ppoutformatctx){
		if(!((*ppoutformatctx)->oformat->flags & AVFMT_NOFILE)){
			if(av_write_trailer(*ppoutformatctx)<0){
				clips_log2(AV_LOG_ERROR,"write output tailer failed\n");
			}
			avio_close((*ppoutformatctx)->pb);
		}
		avformat_free_context(*ppoutformatctx);
		*ppoutformatctx=NULL;
	}
}

static int createnewoutput(clipscontext* pcontext){
	char newfile[512]={0};
	appendclipname(pcontext,newfile);
	if(avformat_alloc_output_context2(&(pcontext->poutformatctx),NULL,pcontext->pinformatctx->iformat->name,newfile)<0){
		clips_log2(AV_LOG_ERROR,"create new ouput %s failed\n",newfile);
		return -1;
	}
	pcontext->poutformatctx->oformat->video_codec=pcontext->pinformatctx->video_codec_id;
	pcontext->poutformatctx->oformat->audio_codec=pcontext->pinformatctx->audio_codec_id;
	AVFormatContext* inc=pcontext->pinformatctx;
	//pcontext->poutformatctx->bit_rate=inc->bit_rate;
	if(inc->packet_size)
		pcontext->poutformatctx->packet_size=inc->packet_size;
	for(int n=0;n<inc->nb_streams;n++){
		AVCodecContext *icodec = inc->streams[n]->codec;
		AVStream* pStream=avformat_new_stream(pcontext->poutformatctx,icodec->codec);
		avcodec_copy_context(pStream->codec,icodec);
		pStream->time_base=inc->streams[n]->time_base;
		/*if(pcontext->curtm==MIN_DTS){
			pStream->start_time=inc->streams[n]->start_time;
		}else{
			pStream->start_time=av_rescale_q(pcontext->curtm,AV_COMMON_BASE,inc->streams[n]->time_base);
		}*/
		if(pStream==NULL){
			clips_log2(AV_LOG_ERROR,"create new stream failed\n");
			return -1;
		}
		AVCodecContext* ocodec=pStream->codec;
		/*if (!pStream->frame_rate.num)
			pStream->frame_rate = inc->streams[n]->frame_rate;
		if(pStream->frame_rate.num)
			codec->time_base = av_inv_q(pStream->frame_rate);*/
		av_reduce(&ocodec->time_base.num, &ocodec->time_base.den,
			ocodec->time_base.num, ocodec->time_base.den, INT_MAX);
		//ocodec->bits_per_raw_sample    = icodec->bits_per_raw_sample;
		//ocodec->chroma_sample_location = icodec->chroma_sample_location;
		//ocodec->codec_id   = icodec->codec_id;
		//ocodec->codec_type = icodec->codec_type;

		//if (!ocodec->codec_tag) {
		//	/*unsigned int codec_tag;
		//	if (!inc->oformat->codec_tag ||
		//		av_codec_get_id (inc->oformat->codec_tag, icodec->codec_tag) == ocodec->codec_id ||
		//		!av_codec_get_tag2(inc->oformat->codec_tag, icodec->codec_id, &codec_tag))*/
		//	ocodec->codec_tag = icodec->codec_tag;
		//}
		//ocodec->bit_rate       = icodec->bit_rate;
		//ocodec->rc_max_rate    = icodec->rc_max_rate;
		//ocodec->rc_buffer_size = icodec->rc_buffer_size;
		//ocodec->field_order    = icodec->field_order;
		//if(icodec->extradata){
		//	ocodec->extradata =(uint8_t*) malloc(icodec->extradata_size);
		//	memcpy(ocodec->extradata, icodec->extradata, icodec->extradata_size);
		//	ocodec->extradata_size= icodec->extradata_size;
		//}
		//ocodec->bits_per_coded_sample  = icodec->bits_per_coded_sample;
		//ocodec->time_base =icodec->time_base;
		//ocodec->profile=icodec->profile;
		//ocodec->level=icodec->level;
		switch (ocodec->codec_type) {
			case AVMEDIA_TYPE_AUDIO:
				/*ocodec->channel_layout     = icodec->channel_layout;
				ocodec->sample_rate        = icodec->sample_rate;
				ocodec->channels           = icodec->channels;
				ocodec->frame_size         = icodec->frame_size;
				ocodec->audio_service_type = icodec->audio_service_type;
				ocodec->block_align        = icodec->block_align;*/
				if((ocodec->block_align == 1 || ocodec->block_align == 1152 || ocodec->block_align == 576) && ocodec->codec_id == AV_CODEC_ID_MP3)
					ocodec->block_align= 0;
				if(ocodec->codec_id == AV_CODEC_ID_AC3)
					ocodec->block_align= 0;
				if (pcontext->poutformatctx->oformat->flags & AVFMT_GLOBALHEADER){
					ocodec->flags |= CODEC_FLAG_GLOBAL_HEADER;
				}
				break;
			case AVMEDIA_TYPE_VIDEO:
				/*ocodec->pix_fmt            = icodec->pix_fmt;
				ocodec->width              = icodec->width;
				ocodec->height             = icodec->height;
				ocodec->bit_rate=icodec->bit_rate;
				ocodec->has_b_frames       = icodec->has_b_frames;*/
				if (pcontext->poutformatctx->oformat->flags & AVFMT_GLOBALHEADER){
					pStream->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;
				}
				if(ocodec->ticks_per_frame>1){
					ocodec->time_base.den /= ocodec->ticks_per_frame;
				}
				//ocodec->rc_buffer_size=0;
				pStream->avg_frame_rate =  inc->streams[n]->avg_frame_rate;
				ocodec->sample_aspect_ratio=pStream->sample_aspect_ratio;
				if (pStream->sample_aspect_ratio.num) { // overridden by the -aspect cli option
					AVRational arcodec={ ocodec->height, ocodec->width };
					ocodec->sample_aspect_ratio   =
						pStream->sample_aspect_ratio =
						av_mul_q(pStream->sample_aspect_ratio,arcodec);
				} else if (!ocodec->sample_aspect_ratio.num) {
					AVRational aratio={0, 1};
					ocodec->sample_aspect_ratio =
						pStream->sample_aspect_ratio =
						inc->streams[n]->sample_aspect_ratio.num ? inc->streams[n]->sample_aspect_ratio :
						inc->streams[n]->codec->sample_aspect_ratio.num ?
						inc->streams[n]->codec->sample_aspect_ratio : aratio;
				}
				pStream->avg_frame_rate = inc->streams[n]->avg_frame_rate;
				break;
			case AVMEDIA_TYPE_SUBTITLE:
				ocodec->width  = icodec->width;
				ocodec->height = icodec->height;
				break;
			default: break;
		}
	}
	if(!(pcontext->poutformatctx->oformat->flags & AVFMT_NOFILE)){
		if(avio_open(&(pcontext->poutformatctx->pb),newfile,AVIO_FLAG_WRITE)<0){
			clips_log2(AV_LOG_ERROR,"open file: %s failed\n",newfile);
			return -1;
		}
	}
	//av_dump_format(m_pOutFormat,0,m_pOutFile,1);
	if(avformat_write_header(pcontext->poutformatctx,NULL)<0){
		clips_log2(AV_LOG_ERROR,"write header failed\n");
		return -1;
	}
	//pcontext->clipindex;
	return 0;
}

void clips_log1(void* ptr, int level, const char* lformat, va_list vl)
{
#ifdef NDEBUG
	if(AV_LOG_DEBUG==level)
		return;
#endif
	const int logLen=1500;
	char logInfo[logLen]={0};
	char *pstrLevel="";
	switch(level)
	{
		case AV_LOG_FATAL:
		case AV_LOG_ERROR:
		case AV_LOG_PANIC:pstrLevel="error:";break;
		case AV_LOG_WARNING:pstrLevel="warning:";break;
		case AV_LOG_DEBUG:pstrLevel="debug:";break;
		default:pstrLevel="info:";break;
	}
	strcpy(logInfo,pstrLevel);
	int leftLen=strlen(logInfo);
	//vsprintf_s(logInfo+leftLen,logLen-leftLen,lformat,vl);
	vsprintf(logInfo+leftLen,lformat,vl);
	printf("%s",logInfo);
	//OutputDebugString(logInfo);
}

void clips_log2(int level,const char* lformat,...)
{
#ifdef NDEBUG
	if(AV_LOG_DEBUG==level)
		return;
#endif
	va_list argp;
	va_start(argp,lformat);
	clips_log1(NULL,level,lformat,argp);
	va_end(argp);
}

static unsigned long get_file_size(const char *path)
{  
    unsigned long filesize = -1;      
    struct stat statbuff;  
    if(stat(path, &statbuff) < 0){  
        return filesize;  
    }else{  
        filesize = statbuff.st_size;  
    }  
    return filesize;  
}  
int clips_init(clipscontext* pcontext)
{
	av_register_all();
	avcodec_register_all();
	avformat_network_init();
	//av_log_set_callback(clips_log1);
	pcontext->clipindex=0;
	pcontext->pinformatctx=NULL;
	pcontext->poutformatctx=NULL;
	pcontext->iseof=0;
	pcontext->curtm=MIN_DTS;
	pcontext->clipstarttm=MIN_DTS;
	pcontext->quit_signal=0;
    pthread_mutex_init(&pcontext->mutexevent, NULL);
	//pcontext->mutexevent=CreateMutex(NULL,FALSE,NULL);
	if(avformat_open_input(&(pcontext->pinformatctx),pcontext->inputfile,NULL,NULL)<0)
	{
		clips_log2(AV_LOG_ERROR,"open input %s failed\n",pcontext->inputfile);
		return -1;
	}
	pcontext->pinformatctx->max_analyze_duration*=2;
	pcontext->pinformatctx->probesize*=2;
	if(avformat_find_stream_info(pcontext->pinformatctx,NULL)<0)
	{
		clips_log2(AV_LOG_ERROR,"no media content in %s\n",pcontext->inputfile);
		return -1;
	}
	if(rectifycliptime(pcontext)<0)
	{
		return -1;
	}
	/*HANDLE hFile=CreateFile(pcontext->inputfile,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
	if(hFile==INVALID_HANDLE_VALUE){
		clips_log2(AV_LOG_WARNING,"get file size %s failed(%d) and don't print progress\n",pcontext->inputfile,GetLastError());
	}else{
		LARGE_INTEGER fileSize;
		GetFileSizeEx(hFile,&fileSize);
		pcontext->filesize=fileSize.QuadPart;
		CloseHandle(hFile);
	}*/

	return streams_init(pcontext);
}

void clips_uninit(clipscontext* pcontext)
{
	if(pcontext)
		return;
	streams_uinit(pcontext);
	if(pcontext->pinformatctx)
		avformat_close_input(&(pcontext->pinformatctx));
	closeoutput(&(pcontext->poutformatctx));
	//CloseHandle(pcontext->mutexevent);
    pthread_mutex_destroy(&pcontext->mutexevent);
	avformat_network_deinit();
}

int clips_cutsize(clipscontext* pcontext)
{
	AVRational cliptb={1,1000};
	pcontext->clipblock=av_rescale_q(pcontext->clipblock,cliptb,AV_COMMON_BASE);
	if(createnewoutput(pcontext)<0)
		return -1;
	/*HANDLE ht=CreateThread(NULL,0,&input_thread,(LPVOID)pcontext,0,NULL);
	if(ht==NULL){
		clips_log2(AV_LOG_ERROR,"create input thread failed\n");
		return -1;
	}
	CloseHandle(ht);*/
    pthread_attr_t attr; 
    pthread_attr_init(&attr); 
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    int err = pthread_create(NULL, &attr, input_thread, (void*)pcontext);
    if(err != 0)
    {
		clips_log2(AV_LOG_ERROR,"create input thread failed\n");
		return -1;
    }

	while(!(pcontext->quit_signal))
	{
		AVPacket avpacket;
		if(choose_ouput_packet(pcontext,&avpacket))
		{
			if(avpacket.stream_index==pcontext->defstreamidx 
				&& (avpacket.flags & AV_PKT_FLAG_KEY) 
				&& avpacket.pos>=0)
			{
					int64_t lldif=pcontext->curtm-pcontext->clipstarttm;
					if(lldif>=pcontext->clipblock 
						|| (lldif * 100 / pcontext->clipblock)>=95)
					{
						AVPacket tmppacket;
						while(choose_ouput_packet_nondefault(pcontext,&tmppacket))
						{
							write_packet_ouput(pcontext,&tmppacket);
							if(tmppacket.pos>0)
							{
								pcontext->pprogress=tmppacket.pos * 100 / pcontext->filesize;
								clips_log2(AV_LOG_INFO,"progress: %d\n",pcontext->pprogress);
							}
							av_free_packet(&tmppacket);
							av_init_packet(&tmppacket);
						}
						closeoutput(&(pcontext->poutformatctx));
						pcontext->clipindex++;
						if(createnewoutput(pcontext)<0)
							return -1;
						pcontext->clipstarttm=pcontext->curtm;
					}
			}
			
			write_packet_ouput(pcontext,&avpacket);
			if(avpacket.pos>0)
			{
				pcontext->pprogress=avpacket.pos * 100 / pcontext->filesize;
				clips_log2(AV_LOG_INFO,"progress: %d\n",pcontext->pprogress);
			}
			av_free_packet(&avpacket);
			av_init_packet(&avpacket);
		}else{
			if(pcontext->iseof)
				break;
			else
				sleep(10);
		}
	}
	pcontext->pprogress=100;
	clips_log2(AV_LOG_INFO,"progress: %d\n",pcontext->pprogress);
	return 0;
}

int clips_cuttime(clipscontext* pcontext)
{
	if(createnewoutput(pcontext)<0)
		return -1;
	/*HANDLE ht=CreateThread(NULL,0,&input_thread,(LPVOID)pcontext,0,NULL);
	if(ht==NULL){
		clips_log2(AV_LOG_ERROR,"create input thread failed\n");
		return -1;
	}
	CloseHandle(ht);*/

    pthread_attr_t attr; 
    pthread_attr_init(&attr); 
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    int err = pthread_create(NULL, &attr, input_thread, (void*)pcontext);
    if(err != 0)
    {
		clips_log2(AV_LOG_ERROR,"create input thread failed\n");
		return -1;
    }

	AVPacket avpacket;
	while(!(pcontext->quit_signal))
	{
		if(choose_ouput_packet(pcontext,&avpacket))
		{
			//if(avpacket.stream_index==pcontext->defstreamidx){
			//	printf("choose video\n");
			//}else
			//	printf("choose audio\n");
			if(avpacket.stream_index==pcontext->defstreamidx 
				&& (avpacket.flags & AV_PKT_FLAG_KEY) 
				&& avpacket.pos>=0)
			{
				int64_t lldif=pcontext->curtm-pcontext->clipstarttm;
				if(lldif>=pcontext->clipblock /*|| (lldif * 100 / pcontext->clipblock)>=95*/)
				{
					AVPacket tmppacket;
					while(choose_ouput_packet_nondefault(pcontext,&tmppacket))
					{
						write_packet_ouput(pcontext,&tmppacket);
						if(tmppacket.pos>0 && pcontext->filesize)
						{
							int tmp=tmppacket.pos * 100 / pcontext->filesize;
							if(tmp>pcontext->pprogress)
							{
								pcontext->pprogress=tmp;
								clips_log2(AV_LOG_INFO,"progress: %d\n",pcontext->pprogress);
							}
						}
						av_free_packet(&tmppacket);
						av_init_packet(&tmppacket);
					}
					clips_log2(AV_LOG_INFO,"clip: %s\n",pcontext->poutformatctx->filename);
					closeoutput(&(pcontext->poutformatctx));
					pcontext->clipindex++;
					if(createnewoutput(pcontext)<0)
						return -1;
					pcontext->clipstarttm=pcontext->curtm;
				}
			}
			write_packet_ouput(pcontext,&avpacket);
			if(avpacket.pos>0 && pcontext->filesize)
			{
				int tmp=avpacket.pos * 100 / pcontext->filesize;
				if(tmp>pcontext->pprogress)
				{
					pcontext->pprogress=tmp;
					clips_log2(AV_LOG_INFO,"progress: %d\n",pcontext->pprogress);
				}
			}
			av_free_packet(&avpacket);
			av_init_packet(&avpacket);
		}
		else
		{
			if(pcontext->iseof)
			{
				while(choose_ouput_packet(pcontext,&avpacket))
				{
					write_packet_ouput(pcontext,&avpacket);
					av_free_packet(&avpacket);
				}
				clips_log2(AV_LOG_INFO,"clip: %s\n",pcontext->poutformatctx->filename);
				closeoutput(&(pcontext->poutformatctx));
				break;
			}
			else
				sleep(10);
		}
	}
	pcontext->pprogress=100;
	//clips_log2(AV_LOG_INFO,"progress: %d\n",pcontext->pprogress);
	return 0;
}

static void ouput_virtualclip(clipscontext* pcontext)
{
	static AVRational ms_time_base={1,1000};
	//char strss[20]={0};
	//char strt[20]={0};	
	char strss[20]="00:00:00.000";
	char strt[20]="00:00:00.000";
//	clips_log2(AV_LOG_INFO,"lasttm: %lld-%lld \n",pcontext->lasttm, pcontext->clipstarttm);
	if(!pcontext->iseof)
	{
		int64_t tmpduration=av_rescale_q(pcontext->lasttm - pcontext->clipstarttm, pcontext->pinformatctx->streams[pcontext->defstreamidx]->time_base,ms_time_base);
		int msec=tmpduration%1000;
		int sec=tmpduration/1000;
		int min=sec/60;
		int hour=min/60;
		min=min%60;
		sec=sec%60;
		sprintf(strt,"%02d:%02d:%02d.%03d",hour,min,sec,msec);
	}
	
	if(pcontext->clipstarttm > pcontext->pinformatctx->streams[pcontext->defstreamidx]->start_time)
	{
		int64_t tmpstart=av_rescale_q(pcontext->clipstarttm/*-pcontext->defstreamst*/
			,pcontext->pinformatctx->streams[pcontext->defstreamidx]->time_base,ms_time_base);
		int msec=tmpstart%1000;
		int sec=tmpstart/1000;
		int min=sec/60;
		int hour=min/60;
		min=min%60;
		sec=sec%60;
		sprintf(strss,"%02d:%02d:%02d.%03d",hour,min,sec,msec);
	}
	clips_log2(AV_LOG_INFO,"virtualclip: %s-%s\n",strss,strt);
}

int clips_virtualcut(clipscontext* pcontext)
{
	pcontext->clipblock=av_rescale_q(pcontext->clipblock,AV_COMMON_BASE
		,pcontext->pinformatctx->streams[pcontext->defstreamidx]->time_base);

	AVRational minRa={1,1};
	int64_t mintm=av_rescale_q(pcontext->mintime,minRa
		,pcontext->pinformatctx->streams[pcontext->defstreamidx]->time_base);
	if(mintm>pcontext->clipblock)
	{
		clips_log2(AV_LOG_DEBUG,"mintime>cliptime: %lld > %lld\n",mintm,pcontext->clipblock);
		pcontext->clipblock=mintm;
	}
	
	int64_t lduration=av_rescale_q(pcontext->pinformatctx->duration,AV_COMMON_BASE
		,pcontext->pinformatctx->streams[pcontext->defstreamidx]->time_base);
	if(lduration<=mintm)
	{
		clips_log2(AV_LOG_DEBUG,"no clip mintime>=totaltime: %lld >= %lld\n",mintm,lduration);
		return 0;
	}
	
	//计算视频帧用时
	AVRational avTmp=pcontext->pinformatctx->streams[pcontext->defstreamidx]->avg_frame_rate;
	int frameDuration=0;
    if(avTmp.den>0)
    {
       AVRational avframetime={avTmp.den,avTmp.num};
       frameDuration=av_rescale_q(1,avframetime,pcontext->pinformatctx->streams[pcontext->defstreamidx]->time_base);
    }
    //AVRational avframetime={avTmp.den,avTmp.num};
	//int frameDuration=av_rescale_q(1,avframetime,pcontext->pinformatctx->streams[pcontext->defstreamidx]->time_base);

	AVPacket avpacket;
	av_init_packet(&avpacket);
	while(av_read_frame(pcontext->pinformatctx,&avpacket)>=0)
	{
		if(avpacket.stream_index==pcontext->defstreamidx )
		{
			pcontext->lasttm = pcontext->curtm;
			pcontext->curtm=avpacket.pts;
		}	

		if(avpacket.stream_index==pcontext->defstreamidx 
			&& (avpacket.flags & AV_PKT_FLAG_KEY) 
			&& avpacket.pos>=0)
		{
//			clips_log2(AV_LOG_INFO,"pts: %lld\n",avpacket.pts);

			if(pcontext->clipstarttm==MIN_DTS)
			{
				pcontext->clipstarttm=avpacket.pts;
				pcontext->defstreamst=avpacket.pts;
				av_free_packet(&avpacket);
				av_init_packet(&avpacket);
				continue;
			}
			if(pcontext->curtm-pcontext->clipstarttm>=pcontext->clipblock)
			{
				// change duration times 2015-2-27
				if(frameDuration<=0)
					pcontext->curtm-=avpacket.duration;
				else
					pcontext->curtm-=frameDuration;
				
				ouput_virtualclip(pcontext);

				pcontext->clipstarttm=avpacket.pts;
//					pcontext->clipstarttm=pcontext->curtm;
				if(avpacket.pts+pcontext->clipblock>=lduration)
				{
					av_free_packet(&avpacket);
					break;
				}
			}
		}
		if(avpacket.pos>0 && pcontext->filesize)
		{
			int tmp=avpacket.pos * 100 / pcontext->filesize;
			if(tmp>pcontext->pprogress)
			{
				pcontext->pprogress=tmp;
				//clips_log2(AV_LOG_INFO,"progress: %d\n",pcontext->pprogress);
			}
		}
		av_free_packet(&avpacket);
		av_init_packet(&avpacket);
	}

	pcontext->iseof=1;
	ouput_virtualclip(pcontext);
	pcontext->pprogress=100;
	//clips_log2(AV_LOG_INFO,"progress: %d\n",pcontext->pprogress);
	return 0;
}

int seekframe(AVFormatContext* context, int64_t seekFrameTime)//跳转到指定位置
{
	int defaultStreamIndex = av_find_default_stream_index(context);
	AVRational time_base = context->streams[defaultStreamIndex]->time_base;
	int64_t seekTime = context->streams[defaultStreamIndex]->start_time + av_rescale(seekFrameTime, time_base.den, time_base.num);
	int ret = av_seek_frame(context, defaultStreamIndex, seekTime, AVSEEK_FLAG_ANY);
//	int ret = av_seek_frame(context, defaultStreamIndex, seekTime, AVSEEK_FLAG_ANY | AVSEEK_FLAG_BACKWARD);

	return 0;
}