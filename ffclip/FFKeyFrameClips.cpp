// FFKeyFrameClips.cpp : Defines the entry point for the console application.
//

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "FFClips.h"

clipscontext ccontext;

void parseoptions(int argc, char* argv[]){
	for(int n=1;n<argc;n++){
		if(!strcmp(argv[n],"-t")){
			ccontext.clipblock=atol(argv[++n]);
		}else if(!strcmp(argv[n],"-c")){
			ccontext.clipcount=atol(argv[++n]);
		}else if(!strcmp(argv[n],"-i")){
			strcpy(ccontext.inputfile,argv[++n]);
		}else if(!strcmp(argv[n],"-o")){
			strcpy(ccontext.outputfile,argv[++n]);
		}else if(!strcmp(argv[n],"-virtual")){
			memset(ccontext.cuttype,0,sizeof(ccontext.cuttype));
			strcpy(ccontext.cuttype,argv[n]);
		}else if(!strcmp(argv[n],"-real")){
			memset(ccontext.cuttype,0,sizeof(ccontext.cuttype));
			strcpy(ccontext.cuttype,argv[n]);
		}else if(!strcmp(argv[n],"-mintm")){
			ccontext.mintime=atol(argv[++n]);
		}else{
			clips_log2(AV_LOG_WARNING,"invalid  parameter: %s\n",argv[n]);
		}
	}
	if(ccontext.inputfile[0]=='\'' || ccontext.inputfile[0]=='"'){
		ccontext.inputfile[strlen(ccontext.inputfile)-1]='\0';
		memmove(ccontext.inputfile,ccontext.inputfile+1,strlen(ccontext.inputfile));
	}
	if(ccontext.outputfile[0]=='\'' || ccontext.outputfile[0]=='"'){
		ccontext.outputfile[strlen(ccontext.outputfile)-1]='\0';
		memmove(ccontext.outputfile,ccontext.outputfile+1,strlen(ccontext.outputfile));
	}
}

int main(int argc, char* argv[])
{
	memset(&ccontext, 0, sizeof(clipscontext));

	parseoptions(argc,argv);

	//DWORD starttm=GetTickCount();
	clock_t starttm, endtm;
    starttm = clock();
	if(ccontext.clipcount>1 || ccontext.clipcount==0)
	{
		if(clips_init(&ccontext)<0)
			return -1;
	
		if(!strcmp(ccontext.cuttype,"-virtual")){
			clips_virtualcut(&ccontext);
		}else{
			clips_cuttime(&ccontext);
		}
	
		clips_uninit(&ccontext);
	}
	else
	{
		//clips_log2(AV_LOG_INFO,"progress: %d\n",100);
		clips_log2(AV_LOG_INFO,"clip: %s\n",ccontext.inputfile);
	}

	//DWORD endtm=GetTickCount();
    endtm = clock();
	printf("use time: %.1f\n", double(endtm-starttm)/double(CLOCKS_PER_SEC));
//#ifdef _DEBUG
//	getchar();
//#endif
	return 0;
}

