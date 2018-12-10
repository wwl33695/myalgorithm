/*    gcc -o videoframe videoframe.c -lavformat -lavcodec -lavutil -lz -lm -lpthread -lswscale  */  
#include <libavformat/avformat.h>  
#include <libavcodec/avcodec.h>  
#include <libavutil/avutil.h>  
#include <libswscale/swscale.h>  
#include <stdio.h>  

static void SaveFrame(AVFrame *pFrame, int width, int height, int iFrame)  
{  
    FILE *pFile;  
    char szFilename[32];  
    int y;  
    sprintf(szFilename, "frame%d.ppm", iFrame);  
    pFile = fopen(szFilename, "wb");  
    if(!pFile)  
        return;  
    fprintf(pFile, "P6\n%d %d\n255\n", width, height);  //ppm 文件头  
    for(y=0; y<height; y++)  
        fwrite(pFrame->data[0] + y * pFrame->linesize[0], 1, width * 3, pFile);  
    fclose(pFile);  
}  

int main2(int argc, const char *video_path[])  
{  
    AVFormatContext *pFormatCtx = NULL;  
    int i, videoStream;  
    AVCodecContext *pCodecCtx;  
    AVCodec *pCodec;  
    AVFrame *pFrame;  
    AVFrame *pFrameRGB;  
    AVPacket packet;  
    int frameFinished;  
    int numBytes;  
    uint8_t *buffer;  

    av_register_all();  
    if(avformat_open_input(&pFormatCtx, video_path[1], NULL, NULL) != 0) {  
        return -1;  
    }  
    if(avformat_find_stream_info(pFormatCtx, NULL) < 0) {  
        return -1;  
    }  
    av_dump_format(pFormatCtx, -1, video_path[1], 0);  
    videoStream = -1;  
    for(i=0; i<pFormatCtx->nb_streams; i++)  
        if(pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {  
            videoStream = i;  
            break;  
        }  
    if(videoStream == -1) {  
        return -1;  
    }  
    pCodecCtx = pFormatCtx->streams[videoStream]->codec;  
    pCodec = avcodec_find_decoder(pCodecCtx->codec_id);  
    if(pCodec == NULL) {  
        return -1;  
    }  
    if(avcodec_open2(pCodecCtx, pCodec, NULL) < 0) {  
        return -1;  
    }  
    pFrame = av_frame_alloc();  
    if(pFrame == NULL) {  
        return -1;  
    }  
    pFrameRGB = av_frame_alloc();  
    if(pFrameRGB == NULL) {  
        return -1;  
    }  
    numBytes = avpicture_get_size(AV_PIX_FMT_RGB24, pCodecCtx->width, pCodecCtx->height);  
    buffer = av_malloc(numBytes);  
    avpicture_fill((AVPicture *)pFrameRGB, buffer, AV_PIX_FMT_RGB24, pCodecCtx->width, pCodecCtx->height);  
    i = 0;  
    while(av_read_frame(pFormatCtx, &packet) >=0) {  
        if(packet.stream_index == videoStream) {  
            avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, &packet);  
            if(frameFinished) {  
                struct SwsContext *img_convert_ctx = NULL;  
                img_convert_ctx = sws_getCachedContext(img_convert_ctx, pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt, pCodecCtx->width, pCodecCtx->height, AV_PIX_FMT_RGB24, SWS_BICUBIC, NULL, NULL, NULL);  
                if(!img_convert_ctx) {  
                    fprintf(stderr, "Cannot initialize sws conversion context\n");  
                    exit(1);  
                }  
                sws_scale(img_convert_ctx,(const uint8_t * const *)pFrame->data, pFrame->linesize, 0 , pCodecCtx->height, pFrameRGB->data, pFrameRGB->linesize);  
                //if(i++ < 50) {  
                    //SaveFrame(pFrameRGB, pCodecCtx->width, pCodecCtx->height, i);  
                    printf("%d ", i++);
                //}  
            }  
        }  
        av_free_packet(&packet);  
    }  
    av_free(buffer);  
    av_free(pFrameRGB);  
    av_free(pFrame);  
    avcodec_close(pCodecCtx);  
    avformat_close_input(&pFormatCtx);  
    return 0;  
}