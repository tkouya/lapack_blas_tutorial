/**********************************************/
/* get_sec.c:                                 */
/* Copyright (C) 2003-2016 Tomonori Kouya     */
/*                                            */
/* This library is free software; you can re- */
/* distribute it and/or modify it under the   */
/* terms of the GNU Lesser General Public     */
/* License as published by the Free Software  */
/* Foundation; either version 2.1 of the      */
/* License, or (at your option) any later     */
/* version.                                   */
/*                                            */
/* This library is distributed in the hope    */
/* that it will be useful, but WITHOUT ANY    */
/* WARRANTY; without even the implied         */
/* warranty of MERCHANTABILITY or FITNESS FOR */
/* A PARTICULAR PURPOSE.  See the GNU Lesser  */
/* General Public License for more details.   */
/**********************************************/
#include <stdio.h>
#include <time.h>

#ifdef WIN32
#include <windows.h>
#else
#include <sys/times.h>
#include <sys/time.h>
#ifndef CLK_TCK
	#include <unistd.h>
	#define CLK_TCK (sysconf(_SC_CLK_TCK))
#endif
#endif

#include "get_secv.h"

#define DIVTIMES 2

//#define USE_CLOCK

/* flag == 0: No print */
double get_sec(int flag)
{
#ifdef WIN32
	static int first = 1;
	static LARGE_INTEGER _tstart;
	static LARGE_INTEGER freq;

	if(first) {
		QueryPerformanceFrequency(&freq);
		first = 0;
	}
	QueryPerformanceCounter(&_tstart);
	return ((double)_tstart.QuadPart)/((double)freq.QuadPart);
#else
	double ret;

	struct tms tmp;

#ifdef USE_CLOCK
	if(flag != 0)
		printf("Time  : %d / %d\n", (int)clock(), CLOCKS_PER_SEC);
	ret = (double)(clock()) / CLOCKS_PER_SEC;
#else
#ifdef USE_MPFRTIME
//int
//cputime ()
//{
#include <sys/types.h>
#include <sys/resource.h>
//
  struct rusage rus;

  getrusage (0, &rus);
  //return rus.ru_utime.tv_sec * 1000 + rus.ru_utime.tv_usec / 1000;
  ret =  rus.ru_utime.tv_sec + rus.ru_utime.tv_usec / 1000 / 1000;
//}
#else
	times(&tmp);
	if(flag != 0)
	{
		printf("User Time  : %ld / %ld\n", tmp.tms_utime, CLK_TCK);
		printf("System Time: %ld / %ld\n", tmp.tms_stime, CLK_TCK);
		printf("CUser Time  : %ld / %ld\n", tmp.tms_cutime, CLK_TCK);
		printf("CSystem Time: %ld / %ld\n", tmp.tms_cstime, CLK_TCK);
		printf("Ret(utime+stime)   : %g\n", (double)(tmp.tms_utime + tmp.tms_stime) / CLK_TCK);
		printf("Ret(cutime+cstime) : %g\n", (double)(tmp.tms_cutime + tmp.tms_cstime) / CLK_TCK);
	}
	ret = (double)(tmp.tms_utime + tmp.tms_stime) / CLK_TCK;
#endif
#endif
	return ret;
#endif
}

/* double get_secv(void) */
double get_secv(void)
{
	return get_sec(0);
}

/* float fget_sec(int flag) */
float fget_sec(int flag)
{
	return (float)get_sec(flag);
}

/* float fget_sec(void) */
float fget_secv(void)
{
	return (float)get_sec(0);
}

/* get REAL time */
/* for Multi-threads programming */
double get_real_sec(int flag)
{
#ifdef WIN32
	static int first = 1;
	static LARGE_INTEGER _tstart;
	static LARGE_INTEGER freq;

	if(first) {
		QueryPerformanceFrequency(&freq);
		first = 0;
	}
	QueryPerformanceCounter(&_tstart);
	return ((double)_tstart.QuadPart)/((double)freq.QuadPart);
#else
	double ret;
	struct timeval tmp;

	gettimeofday(&tmp, NULL);
	if(flag != 0)
	{
		printf("tv_sec : %ld\n", tmp.tv_sec);
		printf("tv_usec: %ld\n", tmp.tv_usec);
		printf("Ret    : %g\n", (double)tmp.tv_sec + (double)tmp.tv_usec / 1000.0 / 1000.0);
	}
	ret = (double)tmp.tv_sec + (double)tmp.tv_usec / 1000.0 / 1000.0;

	return ret;
#endif
}
 
double get_real_secv(void)
{
	return get_real_sec(0);
}

float fget_real_sec(int flag)
{
	return (float)get_real_sec(flag);
}

float fget_real_secv(void)
{
	return (float)get_real_sec(0);
}
