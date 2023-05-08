/****************************************************************************
 *
 *  fe_double_symmetric.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#ifndef LUDWIG_FE_DOUBLE_SYMMETRIC_H
#define LUDWIG_FE_DOUBLE_SYMMETRIC_H

#include "memory.h"
#include "free_energy.h"
#include "field.h"
#include "field_grad.h"

typedef struct fe_double_symmetric_s fe_double_symmetric_t;
typedef struct fe_double_symmetric_param_s fe_double_symmetric_param_t;

struct fe_double_symmetric_param_s {
  double a0;              /* Symmetric a */
  double b0;              /* Symmetric b */
  double kappa0;          /* Symmetric kappa */

  double a1;              /* Symmetric a */
  double b1;              /* Symmetric b */
  double kappa1;          /* Symmetric kappa */

  double c;               /* Not meant to be used at first. If used, phi and psi will have the same
                             c and h */
  double h;
};

struct fe_double_symmetric_s {
  fe_t super;                      /* "Superclass" block */
  pe_t * pe;                       /* Parallel environment */
  cs_t * cs;                       /* Coordinate system */
  fe_double_symmetric_param_t * param;         /* Parameters */
  field_t * phi;                   /* Single field with {phi,psi} */
  field_grad_t * dphi;             /* gradients thereof */
  fe_double_symmetric_t * target;              /* Device copy */
};

__host__ int fe_double_symmetric_create(pe_t * pe, cs_t * cs, field_t * phi,
			    field_grad_t * dphi, fe_double_symmetric_param_t param,
			    fe_double_symmetric_t ** fe);
__host__ int fe_double_symmetric_free(fe_double_symmetric_t * fe);
__host__ int fe_double_symmetric_info(fe_double_symmetric_t * fe);
__host__ int fe_double_symmetric_param_set(fe_double_symmetric_t * fe, fe_double_symmetric_param_t vals);
__host__ int fe_double_symmetric_sigma(fe_double_symmetric_t * fe, double * sigma0, double * sigma1);
__host__ int fe_double_symmetric_xi0(fe_double_symmetric_t * fe,  double * xi0, double * xi1);
__host__ int fe_double_symmetric_langmuir_isotherm(fe_double_symmetric_t * fe, double * psi_c);
__host__ int fe_double_symmetric_target(fe_double_symmetric_t * fe, fe_t ** target);

__host__ int fe_double_symmetric_param(fe_double_symmetric_t * fe, fe_double_symmetric_param_t * param);
__host__ int fe_double_symmetric_fed(fe_double_symmetric_t * fe, int index, double * fed);
__host__ int fe_double_symmetric_mu(fe_double_symmetric_t * fe, int index, double * mu);
__host__ int fe_double_symmetric_str(fe_double_symmetric_t * fe, int index, double s[3][3]);
__host__ int fe_double_symmetric_str_v(fe_double_symmetric_t * fe, int index, double s[3][3][NSIMDVL]);

#endif
