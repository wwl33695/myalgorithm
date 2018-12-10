#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <getopt.h>

#include "./include/qisr.h" 
#include "./include/msp_cmn.h"
#include "./include/msp_errors.h"
              
#include <iterator>
#include <cstring>
#include "iostream" 
#include "convert.h"

using namespace std;


#define FRAME_LEN	640 
#define HINTS_SIZE  100

bool bIsSucceed = 1;

void run_ist(const char* audio_file, const char* session_begin_params)
{
	const char*		session_id					=	NULL;
	char			hints[HINTS_SIZE]			=	{0}; //hints为结束本次会话的原因描述，由用户自定义
	unsigned int	total_len					=	0; 
	int				aud_stat					=	MSP_AUDIO_SAMPLE_CONTINUE ;		//音频状态
	int				rec_stat					=	MSP_REC_STATUS_SUCCESS ;			//识别状态
	int				errcode						=	MSP_SUCCESS ;

	FILE*			f_pcm						=	NULL;
	char*			p_pcm						=	NULL;
	long			pcm_count					=	0;
	long			pcm_size					=	0;
	long			read_size					=	0;
	char            send_len[32]                =   {0};
	
	if (NULL == audio_file)
	{
		bIsSucceed = 0;
		goto ist_exit;
	}
	f_pcm = fopen(audio_file, "rb");
	if (NULL == f_pcm) 
	{
		bIsSucceed = 0;
		printf("\nopen [%s] failed! \n", audio_file);
		goto ist_exit;
	}
	
	fseek(f_pcm, 0, SEEK_END);
	pcm_size = ftell(f_pcm); //获取音频文件大小 
	fseek(f_pcm, 0, SEEK_SET);		

	p_pcm = (char *)malloc(pcm_size);
	if (NULL == p_pcm)
	{
		bIsSucceed = 0;
		printf("\nout of memory! \n");
		goto ist_exit;
	}

	read_size = fread((void *)p_pcm, 1, pcm_size, f_pcm); //读取音频文件内容
	if (read_size != pcm_size)
	{
		bIsSucceed = 0;
		printf("\nread [%s] error!\n", audio_file);
		goto ist_exit;
	}
	
	session_id = QISRSessionBegin(NULL, session_begin_params, &errcode); 
	if (MSP_SUCCESS != errcode)
	{
		bIsSucceed = 0;
		printf("\nQISRSessionBegin failed! error code:%d\n", errcode);
		goto ist_exit;
	}
	
	printf("\n开始语音转写 ...\n");
	while (1) 
	{
		unsigned int len = 100 * FRAME_LEN; // 可以根据内存大小，调整每次写入音频大小
		int ret = 0;
		unsigned int    send_buflen                 =   sizeof(send_len);
		if (pcm_size < 2 * len) 
			len = pcm_size;
		if (len <= 0)
			break;

		aud_stat = MSP_AUDIO_SAMPLE_CONTINUE;
		if (0 == pcm_count)
			aud_stat = MSP_AUDIO_SAMPLE_FIRST;

		printf(">");
		ret = QISRAudioWrite(session_id, (const void *)&p_pcm[pcm_count], len, aud_stat, NULL, &rec_stat);
		if (MSP_SUCCESS != ret)
		{
			bIsSucceed = 0;
			printf("\nQISRAudioWrite failed! error code:%d\n", ret);
			goto ist_exit;
		}
			
		pcm_count += (long)len;
		pcm_size  -= (long)len;
		
		if (MSP_REC_STATUS_SUCCESS == rec_stat) //已经有部分转写结果
		{
			const char *rslt = QISRGetResult(session_id, &rec_stat, 0, &errcode);
			if (MSP_SUCCESS != errcode)
			{
				bIsSucceed = 0;
				printf("\nQISRGetResult failed! error code: %d\n", errcode);
				goto ist_exit;
			}
			if (NULL != rslt)
			{
//				char buf[100000] = {0};			
//				g2u((char*)rslt, strlen(rslt), buf, 100000);

				printf("\nresult_start\n%s\nresult_end\n", rslt);
			}
//			QISRGetParam(session_id, "sendaudlen", send_len, &send_buflen);// 获取服务端已经接收的音频长度（应用可以根据已经写入的音频长度和服务端已经接收的音频长度得到msc在本地缓存的音频长度）
//			printf("send aud len=%s\n",send_len);
		}
		usleep(50000); //防止频繁占用CPU（可以调整）
	}

	printf("\nQISRAudioWrite last \n");
	errcode = QISRAudioWrite(session_id, NULL, 0, MSP_AUDIO_SAMPLE_LAST, NULL, &rec_stat); //写入最后一个音频，通知音频已写入完毕
	if (MSP_SUCCESS != errcode)
	{
		bIsSucceed = 0;
		printf("\nQISRAudioWrite last failed! error code:%d \n", errcode);
		goto ist_exit;	
	}

	while (MSP_REC_STATUS_COMPLETE != rec_stat) 
	{
		unsigned int    send_buflen                 =   sizeof(send_len);
		const char *rslt = QISRGetResult(session_id, &rec_stat, 0, &errcode);
		if (MSP_SUCCESS != errcode)
		{
			bIsSucceed = 0;
			printf("\nQISRGetResult last failed, error code: %d\n", errcode);
			goto ist_exit;
		}
		if (NULL != rslt)
		{
//			char buf[100000] = {0};
//			g2u((char*)rslt, strlen(rslt), buf, 100000);

            printf("\nresult_start\n%s\nresult_end\n", rslt);
		}
//		QISRGetParam(session_id, "sendaudlen", send_len, &send_buflen); // 获取服务端已经接收的音频长度（应用可以根据已经写入的音频长度和服务端已经接收的音频长度得到msc在本地缓存的音频长度）
//		printf("send aud len=%s\n",send_len);
		usleep(100000); //防止频繁占用CPU（可以调整）
	}
	printf("\n语音转写结束\n");

ist_exit:
	if (NULL != f_pcm)
	{
		fclose(f_pcm);
		f_pcm = NULL;
	}
	if (NULL != p_pcm)
	{	free(p_pcm);
		p_pcm = NULL;
	}

	QISRSessionEnd(session_id, hints);
}

static struct option long_options[] = {
	   {"i", required_argument, NULL, 'i'},
	   {"id",  no_argument,       NULL, 'n'},
	   {0, 0, 0, 0}
};
char *optstring = ":";
static const char* usage = "Usage: options_description [options]\r\n\
Allowed options:\r\n\
	--i arg      input audio file\r\n";

int main(int argc, char* argv[])
{
	const char *inputAudio = NULL;	
	int option_index = 0;
	int opt = 0;
	while ( (opt = getopt_long(argc, argv, optstring, long_options, &option_index)) != -1)
	{
		if( opt == 'i' )
			inputAudio = optarg;
	}

	if( !inputAudio )
	{
		printf("%s", usage);
//  		printf("version 1.0\nusage:\n	--i [wavfilepath]\n");
		return -1;
	}
	printf("inputfile=%s \n", inputAudio);

	const char* login_params			=	"appid = 5638807a"; // 登录参数，appid与msc库绑定,请勿随意改动

	const char* session_begin_params	=	"sub=iat,aue=speex-wb;7,ent=sms16k,rst=json,rse=gb2312,auf=audio/L16;rate=16000";
//	const char* session_begin_params	=	"sub=iat,auf=audio/L16;rate=16000,aue=speex-wb,ent=sms16k,rst=json,rse=gb2312";
//	const char* session_begin_params	=	"sub=iat,auf=audio/L16;rate=16000,aue=speex-wb,ent=sms16k,rst=json,rse=utf8";

	/* 用户登录 */
	int ret = MSPLogin(NULL, NULL, login_params); //第一个参数是用户名，第二个参数是密码，均传NULL即可，第三个参数是登录参数	
	if (MSP_SUCCESS != ret)
	{
		printf("MSPLogin failed , Error code %d.\n",ret);
		bIsSucceed = 0;
		goto exit; //登录失败，退出登录
	}

	printf("\n########################################################################\n");
	printf("## 语音转写技术能够实时地将语音转换成对应的文字。##\n");
	printf("########################################################################\n\n");
	run_ist(inputAudio, session_begin_params);

exit:
	printf("over!!!\n");
	MSPLogout(); //退出登录
	
	if (bIsSucceed == 1)
	{
		return 0;
	}
	else {
	   return -1;
	}
}
