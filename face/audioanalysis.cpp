#include <stdlib.h>
#include <stdio.h>
#include <math.h>
extern "C" {

#ifdef __cplusplus
#define __STDC_CONSTANT_MACROS
#ifdef _STDINT_H
#undef _STDINT_H
#endif
#include <stdint.h>
#endif

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
#include <libavutil/common.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/samplefmt.h>
//#include <libavutil/timestamp.h>

}

/*
 * 声音分贝的计算公式为 20*log10(x/1.0)，其中x为声音在0到1范围时间内采样数值
 * 音频采样大小一般分为3种：8位、16位、32位，对于8位来说范围是[0,255]，对于
 * 16位来说范围是[-32768,32767]，32位是[-1.0,1.0]，数值0代表静音，0向外声音越来越大（取绝对值），在计算时需要取得数值的绝对值
 * NOTE: 因为在此处计算的时nb_samples个采样数据的平均分贝，而声音数据呈现分布为正弦，所以一般分贝数值为平均值/2
 */
#define FILTER_SECTION(T, MIN, MAX)                     \
    T *ptr = (T*)data;                                  \
    if(!ptr) return -1;                                 \
    float v, gap = (MAX - MIN)/2;                       \
    int i;                                              \
    for(i=0; i<nb_samples; i++){                        \
        v = fabs(ptr[i])/gap + 0.0000001;               \
        if(v > 1.0) v = 1.0;                            \
        db[i] = 20*log10(v);                            \
    }                                                   \
    return nb_samples;

static int filterU8(unsigned int nb_samples, void *data, float *db){
    FILTER_SECTION(unsigned char, -255, 255)
}
static int filterS16(unsigned int nb_samples, void *data, float *db){
    FILTER_SECTION(short, -32768, 32767)
}
static int filterS32(unsigned int nb_samples, void *data, float *db){
    FILTER_SECTION(int, -2147483648, 2147483647)
}
static int filterFLT(unsigned int nb_samples, void *data, float *db){
    FILTER_SECTION(float, -1.0, 1.0)
}
static int bytes_to_db(unsigned int nb_samples, void *data, float *db, AVSampleFormat format)
{
    if(format == 5 || format == 0)
        return filterU8(nb_samples, data, db);
    else if(format == 6 || format == 1)
        return filterS16(nb_samples, data, db);
    else if(format == 7 || format == 2)
        return filterS32(nb_samples, data, db);
    else if(format == 8 || format == 3)
        return filterFLT(nb_samples, data, db);
}

extern "C" int audio_decode(const char *filename, float *dbv, unsigned int length, unsigned int fps)
{
    //const char *filename = "a.avi";
    AVFormatContext *avfctx;
    AVPacket pkt;
    AVFrame *decoded_frame = NULL;
    AVCodec *codec = NULL;
    unsigned int buffer_len = 2048;
    float *buffer = (float*)malloc(buffer_len*sizeof(float));
    unsigned int aac_len = 0, aac_limit = 0, count = 0;
    float aac;
    int i;

    av_register_all();

    avfctx = avformat_alloc_context();
    decoded_frame = av_frame_alloc();
    if (avformat_open_input(&avfctx, filename, NULL, NULL) != 0) {
        fprintf(stderr, "open file error: %s\n", filename);
        return -1;
    }

#ifdef _FFMPEG_0_6__
     if(av_find_stream_info(avfctx) < 0)
#else
     if (avformat_find_stream_info(avfctx, NULL) < 0)
#endif
    {
        fprintf(stderr, "find stream info error\n");
        return -1;
    }

    av_dump_format(avfctx, 0, filename, 0);
    int audio_index = -1;

    for (i=0; i<avfctx->nb_streams; i++) {
        if (avfctx->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO) {
            codec = avcodec_find_decoder(avfctx->streams[i]->codec->codec_id);

            if (codec == NULL)
                continue;
            if (avcodec_open2(avfctx->streams[i]->codec, codec, NULL) < 0)
                continue;

            audio_index = i;
        }
    }
    if (audio_index == -1) {
        return -1;
    }

    bool flag = true;
    AVCodecContext *c = NULL;
    while (av_read_frame(avfctx, &pkt) >= 0) {
        c = avfctx->streams[pkt.stream_index]->codec;
        if (c->codec_type == AVMEDIA_TYPE_AUDIO && audio_index == pkt.stream_index) {

            int size = pkt.size;
            int got_frame = 0;
            AVSampleFormat format = c->sample_fmt;
            int sample_rate;

            while (size > 0) {
                int len = avcodec_decode_audio4(c, decoded_frame, &got_frame, &pkt);
                //int second = av_rescale_q(decoded_frame->pts, AV_TIME_BASE, (AVRational){1, 1000});
                sample_rate = decoded_frame->sample_rate;

                // 分贝累加器，用于求一段时间内分贝平均值
                if (aac_limit == 0) {
                    aac_limit = sample_rate/fps;
                    if (aac_limit == 0)
                        aac_limit = 1;
                }

                if (got_frame) {
                    //int data_size = av_get_bytes_per_sample(format);
                    //for (i=0; i<decoded_frame->nb_samples; i++)
                        //for (ch=0; ch<avfctx->streams[pkt.stream_index]->codec->channels; ch++)
                        //    fwrite(decoded_frame->data[ch] + data_size*i, 1, data_size, outfile);

                    // 根据需要，重新分配缓冲大小
                    if (decoded_frame->nb_samples > buffer_len) {
                        free(buffer); buffer = NULL;
                        buffer_len = decoded_frame->nb_samples + 64;
                        buffer = (float*)malloc(buffer_len*sizeof(float));
                        if (buffer == NULL) {
                            flag = false;
                            goto end;
                        }
                    }

                    // 计算解码后数据的分贝数到缓冲区
                    if (bytes_to_db(decoded_frame->nb_samples, (void*)decoded_frame->data[0], buffer, format) == -1) {
                        flag = false;
                        goto end;
                    }

                    // 进入累加器，计算均分贝
                    for (int k=0; k<decoded_frame->nb_samples; k++) {
                        aac += buffer[k];
                        aac_len += 1;

                        if (aac_len == aac_limit) {
                            dbv[count] = aac/aac_len/1.33333;
                            count += 1;
                            aac_len = 0;
                            aac = 0;

                            if (count >= length)
                                goto end;
                        }
                    }
                }

                //bytes_to_db(decoded_frame->nb_samples, decoded_frame->extended_data[0], audiobuffer, format);

                size = pkt.size-len;
            }
        }
    }

end:
    if (buffer != NULL) {
        free(buffer);
        buffer = NULL;
        buffer_len = 0;
    }
    av_free(decoded_frame);
    avformat_free_context(avfctx);

    if (flag)
        return count;
    else
        return -1;
}


/*
av_rescale_q(in->pts, inlink->time_base, (AVRational){1, 1000}),

aud_t aud_in = {
        .nb_samples      = in->nb_samples,
        .sample_rate     = in->sample_rate,
        .channels        = in->channels,
        .channel_layout  = in->channel_layout,
        .pts             = in->pkt_pts,
        .dts             = in->pkt_dts,
        .msc             = av_rescale_q(in->pts, inlink->time_base, (AVRational){1, 1000}),
        .frame_number    = ldaf->num_frames,
        .audio_fmt       = in->format,
    };
    for(i=0; i<in->channels; i++){
        aud_in.extended_data[i] = in->extended_data[i];
    }
    */
