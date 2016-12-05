/**********************************************/
/* get_secv.h:                                */
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
#ifndef __TK_GET_SECV_H
#define __TK_GET_SECV_H

double get_sec(int flag);

/* double get_secv(void) */
double get_secv(void);

/* float fget_sec(int flag) */
float fget_sec(int flag);

/* float fget_sec(void) */
float fget_secv(void);

/* get REAL time */
/* for Multi-threads programming */
double get_real_sec(int flag);
 
double get_real_secv(void);

float fget_real_sec(int flag);

float fget_real_secv(void);

#endif // __TK_GET_SECV_H
