//
// mycuda.h ... CUDA用基礎ルーチン
// Copyright (c) 2015 T.Kouya
//
#ifndef __MYCUDA_H_

#include "cuda.h"
#include "driver_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// GPU上に行列格納領域を確保
void *mycuda_calloc(int, size_t);

// GPU上のメモリ領域を解放
void mycuda_free(void *);

#ifdef __cplusplus
}
#endif

#endif // __MYCUDA_H
