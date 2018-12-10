#ifdef  __cplusplus    
extern "C" {    
#endif    
    #include <libavcodec/avcodec.h>  
    #include <libavformat/avformat.h>  
    #include <libswscale/swscale.h>  
    #include <libavutil/imgutils.h>
#ifdef  __cplusplus    
}    
#endif    

#include "avreader.h"

int read_file(const char* filepath, rgb24_data_callback callback)
{  
//    char *filename = "/home/deepglint/projects/111.264";  

    av_register_all();  
    avcodec_register_all();  
    avformat_network_init();  
    AVFormatContext *pFormatCtx = NULL;  
    if( avformat_open_input(&pFormatCtx,filepath,NULL,NULL) !=  0)  
        return 1;  
    if( avformat_find_stream_info(pFormatCtx, NULL) <0)  
        return 2;  

    av_dump_format(pFormatCtx,0,filepath,0);  
      
    AVCodecContext *pCodecCtx;  
    int i=-1;  
    int videoStream =-1;  
    for(i=0;i<pFormatCtx->nb_streams;i++)  
    {  
        if(pFormatCtx->streams[i]->codec->codec_type== AVMEDIA_TYPE_VIDEO)  
        {  
            videoStream = i;  
            break;  
        }  
    }//for  
    if(videoStream ==-1)  
    {  
        return 3;  
    }  
    pCodecCtx= pFormatCtx->streams[videoStream]->codec;  
    AVCodec *pCodec ;  
    pCodec = avcodec_find_decoder(pCodecCtx->codec_id);  
    if(pCodec ==NULL)  
        return 4;  
    if(avcodec_open2(pCodecCtx,pCodec,NULL)<0)  
        return 5;  

    AVFrame *pFrame = av_frame_alloc();  
    AVFrame  *pFrameRGB = av_frame_alloc();  
    int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, pCodecCtx->width, pCodecCtx->height, 1);  
    uint8_t *buffer  = (uint8_t *)av_malloc(numBytes*sizeof(char) );  
    av_image_fill_arrays(((AVPicture*)pFrameRGB)->data, pFrameRGB->linesize, buffer, AV_PIX_FMT_RGB24, pCodecCtx->width, pCodecCtx->height,1);  
  
    int frameFinished;  
    AVPacket packet;  
    i=0;  
    SwsContext *pSwsCtx = NULL;

    while(av_read_frame(pFormatCtx,&packet)>=0 )  
    {  
        if(packet.stream_index == videoStream )  
        {  
            avcodec_decode_video2(pCodecCtx,pFrame,&frameFinished,&packet);  
            if(frameFinished)  
            {
                if( !pSwsCtx )
                    pSwsCtx = sws_getContext(pCodecCtx->width,  
                        pCodecCtx->height,pCodecCtx->pix_fmt,  
                        pCodecCtx->width,pCodecCtx->height,  
                        AV_PIX_FMT_BGR24,SWS_POINT,NULL,NULL,NULL  
                        );  

                sws_scale(pSwsCtx,pFrame->data,pFrame->linesize,0,  
                    pCodecCtx->height,pFrameRGB->data,pFrameRGB->linesize  
                    );//ffmpeg的从yuv420格式转换到bgr24格式。  

                printf("data coming \n");
                if( callback )
                    callback(pCodecCtx->width, pCodecCtx->height, *(const char **)pFrameRGB->data);

            }//得到了一帧frame  
        }//video  

        av_free_packet(&packet);
    }//while  
    printf("tttttt");  
}  
