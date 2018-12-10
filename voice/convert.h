#include "cstdio"
#include "iconv.h"
#include "string.h"
#include <assert.h>
#define OUTLEN 255

int u2g(char *inbuf,int inlen,char *outbuf,int outlen);
int g2u(char *inbuf,size_t inlen,char *outbuf,size_t outlen);
