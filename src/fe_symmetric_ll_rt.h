/****************************************************************************
 *
 *  fe_symmetric_ll_rt.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Shan Chen (shan.chen@epfl.ch)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#ifndef LUDWIG_FE_SYMMETRIC_LL_RT_H
#define LUDWIG_FE_SYMMETRIC_LL_RT_H

#include "pe.h"
#include "runtime.h"
#include "fe_symmetric_ll.h"

__host__ int fe_symmetric_ll_param_rt(pe_t * pe, rt_t * rt, fe_symmetric_ll_param_t * p);
__host__ int fe_symmetric_ll_init_rt(pe_t * pe, rt_t * rt, fe_symmetric_ll_t * fe,
				field_t * phi);
#endif
