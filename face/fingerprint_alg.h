#ifndef FINGERPRINT_ALG_H
#define FINGERPRINT_ALG_H

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include "types.h"

typedef struct{
    char fp[256];               // 图像指纹
    unsigned int len;           // 指纹长度
    char type;                  // 指纹类型
} fp_t;

#ifdef __cplusplus
extern "C"{
#endif

/* 
 * 计算图像指纹，phash算法
 */
void calc_fingerprint_phash(fp_t *fp, im_t *img);

/* 
 * 计算图像指纹，均值hash算法
 */
void calc_fingerprint_hash(fp_t *fp, im_t *img);

/* 
 * 汉明距离
 */
int hanming_distance(fp_t *fp1, fp_t *fp2);

#ifdef __cplusplus
};
#endif

#endif