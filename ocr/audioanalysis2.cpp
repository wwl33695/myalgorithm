#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
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

using namespace std;

static inline int sign(float v)
{
    if (v > 0) return 1;
    if (v < 0) return -1;
    return 0;
}
static double mean(vector<float> &v)
{
    size_t n = v.size();
    if (n == 0)
        return 0;

    double r = 0;
    for (int i=0; i<n; i++) {
        r += v[i];
    }
    return r/n;
}
static double __max(float *v, unsigned int len)
{
    double ma = -100;
    for (int i=0; i<len; i++) {
        if (ma < v[i])
            ma = v[i];
    }
    return ma;
}
static double __min(float *v, unsigned int len)
{
    double mi = 100;
    for (int i=0; i<len; i++) {
        if (mi > v[i])
            mi = v[i];
    }
    return mi;
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
    double v, gap = 2.0/double(MAX-MIN), off=double(MIN)*gap + 1.0;                         \
    int i;                                              \
    for(i=0; i<nb_samples; i++){                        \
        v = ptr[i]*gap-off;                             \
        if(v > 1.0) v = 1.0;                            \
        if(v < -1.0) v = -1.0;                          \
        db[i] = v;                                      \
    }                                                   \
    return nb_samples;

// 不同音频数据类型，做不同处理
static int filterU8(unsigned int nb_samples, void *data, float *db){
    FILTER_SECTION(unsigned char, 0, 255)
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
static int norm_bytes(unsigned int nb_samples, void *data, float *db, AVSampleFormat format)
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



#define _TIME_BLOCK_ 20
extern "C" int audio_decode2(const char *filename, float *dbv, unsigned int length, unsigned int fps)
{
    //const char *filename = "a.avi";
    AVFormatContext *avfctx;
    AVPacket pkt;
    AVFrame *decoded_frame = NULL;
    AVCodec *codec = NULL;
    unsigned int buffer_len = 2048;
    float *buffer = (float*)malloc(buffer_len*sizeof(float));
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
    AVCodecContext *c = avfctx->streams[audio_index]->codec;
    int sample_rate = c->sample_rate;
    AVSampleFormat format = c->sample_fmt;
    // 开辟内存
    unsigned int window_size = sample_rate/_TIME_BLOCK_;      // 1秒钟分4块
    vector<float> ste, zcr;     // short time energy / zero crossing rate
    double ste_0 = 0, zcr_0 = 0; unsigned int len_0 = 0;    // 临时存储区

    while (av_read_frame(avfctx, &pkt) >= 0) {
        if (audio_index != pkt.stream_index)
            continue;

        int size = pkt.size;
        int got_frame = 0;
        while (size > 0) {
            int len = avcodec_decode_audio4(c, decoded_frame, &got_frame, &pkt);

            if (got_frame == 0) {
                size = pkt.size-len;
                continue;
            }

            // 根据需要，重新分配缓冲大小
            if (decoded_frame->nb_samples > buffer_len) {
                free(buffer);
                buffer = NULL;
                buffer_len = decoded_frame->nb_samples + 64;
                buffer = (float*)malloc(buffer_len*sizeof(float));
                if (buffer == NULL) {
                    flag = false;
                    goto end;
                }
            }

            // 计算解码后数据的分贝数到缓冲区
            if (norm_bytes(decoded_frame->nb_samples, (void*)decoded_frame->data[0], buffer, format) == -1) {
                flag = false;
                goto end;
            }
            //double m1 = __max(buffer, decoded_frame->nb_samples);
            //double m2 = __min(buffer, decoded_frame->nb_samples);
            //printf("%f %f\n", m1, m2);

            // 计算short time energy 和 zero crossing rate
            for (int k=0; k<decoded_frame->nb_samples; k++) {
                ste_0 += buffer[k]*buffer[k];
                if (k == 0) zcr_0 += 0;
                else zcr_0 += abs(sign(buffer[k]) - sign(buffer[k-1]));
                len_0 += 1;

                if (len_0 == window_size) {
                    ste.push_back(ste_0/window_size);
                    zcr.push_back(zcr_0/window_size);
                    len_0 = 0;
                    ste_0 = 0;
                    zcr_0 = 0;
                }
            }

            size = pkt.size-len;
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

    if (!flag)
        return -1;

    // 根据短时能量和过零率阈值判定是否为语音分割处
    unsigned int nn = ste.size();
    ste_0 = mean(ste);
    zcr_0 = mean(zcr);
    //vector<unsigned int> mark_idx;
    double t1 = ste_0*0.08;
    double interval = double(fps)/double(_TIME_BLOCK_);

    for (int i=0; i<length; i++)
        dbv[i] = 0;
    int last_idx = -2;
    for (int i=0; i<nn; i++) {
        if (ste[i] >= t1 || zcr[i] <= zcr_0)
            continue;

        //mark_idx.push_back(i);
        int pos = i*interval;    // equal to float(i)/_TIME_BLOCK_*fps;
        if (pos >= length)
            continue;

        for (int k=0; k<interval; k++) {
            if ((pos+k) >= length || (pos+k) < 0)
                continue;
            dbv[pos+k] = 1;
        }

        if (i-2 == last_idx) {      // 如果2个分割直接有1个非分割，则同化其为分割
            pos = (i-2)*interval;
            for (int k=0; k<interval*3; k++) {
                if ((pos+k) >= length || (pos+k) < 0)
                    continue;
                dbv[pos+k] = 1;
            }
            last_idx = i;
        }
    }

    return 0;
}