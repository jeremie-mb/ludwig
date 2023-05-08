/****************************************************************************
 *
 *  double_symmetric_rt.c
 *
 *  Run time initialisation for the double_symmetric free energy.
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

#include <assert.h>

#include "pe.h"
#include "runtime.h"
#include "double_symmetric.h"
#include "double_symmetric_rt.h"

#include "field_phi_init_rt.h"
#include "field_psi_init_rt.h"

int field_init_combine_insert_double_symmetric(field_t * array, field_t * scalar, int nfin);

/****************************************************************************
 *
 *  fe_double_symmetric_param_rt
 *
 ****************************************************************************/

__host__ int fe_double_symmetric_param_rt(pe_t * pe, rt_t * rt, fe_double_symmetric_param_t * p) {

  assert(pe);
  assert(rt);
  assert(p);

  /* Parameters */

  rt_double_parameter(rt, "phi_a",       &p->a0);
  rt_double_parameter(rt, "phi_b",       &p->b0);
  rt_double_parameter(rt, "phi_kappa",   &p->kappa0);

  rt_double_parameter(rt, "psi_a",       &p->a1);
  rt_double_parameter(rt, "psi_b",       &p->b1);
  rt_double_parameter(rt, "psi_kappa",   &p->kappa1);

  rt_double_parameter(rt, "c",   &p->c);
  rt_double_parameter(rt, "h",   &p->h);
  return 0;
}

/*****************************************************************************
 *
 *  fe_double_symmetric_phi_init_rt
 *
 *  Initialise the composition part of the order parameter.
 *
 *****************************************************************************/

__host__ int fe_double_symmetric_phi_init_rt(pe_t * pe, rt_t * rt, fe_double_symmetric_t * fe,
				  field_t * phi) {

  field_phi_info_t param = {0};
  field_options_t opts = field_options_default();
  field_t * tmp = NULL;

  assert(pe);
  assert(rt);
  assert(fe);
  assert(phi);

  /* Parameters xi0, phi0, phistar */
  fe_double_symmetric_xi0(fe, &param.xi0, &param.xi1);

  /* Initialise phi via a temporary scalar field */

  field_create(pe, phi->cs, NULL, "tmp", &opts, &tmp);

  field_phi_init_rt(pe, rt, param, tmp);
  field_init_combine_insert_double_symmetric(phi, tmp, 0);

  field_free(tmp);

  return 0;
}

/*****************************************************************************
 *
 *  fe_double_symmetric_psi_init_rt
 *
 *****************************************************************************/

__host__ int fe_double_symmetric_psi_init_rt(pe_t * pe, rt_t * rt, fe_double_symmetric_t * fe,
				  field_t * phi) {

  field_options_t opts = field_options_default();
  field_psi_info_t param = {0};
  field_t * tmp = NULL;

  assert(pe);
  assert(rt);
  assert(fe);
  assert(phi);

  /* Initialise double_symmetric via a temporary field */

  field_create(pe, phi->cs, NULL, "tmp", &opts, &tmp);

  field_psi_init_rt(pe, rt, param, tmp);
  field_init_combine_insert_double_symmetric(phi, tmp, 1);

  field_free(tmp);

  return 0;
}

/*****************************************************************************
 *
 *  field_init_combine_insert_double_symmetric (same as field_init_combine_ *  insert)
 *
 *  Insert scalar field into array field at position nfin
 *
 ****************************************************************************/

int field_init_combine_insert_double_symmetric(field_t * array, field_t * scalar, int nfin) {

  int nlocal[3];
  int ic, jc, kc, index;
  double val[2];

  assert(array);
  assert(scalar);
  assert(array->nf == 2);
  assert(scalar->nf == 1);
  assert(nfin <= array->nf);

  cs_nlocal(array->cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

  index = cs_index(array->cs, ic, jc, kc);
  field_scalar_array(array, index, val);
  field_scalar(scalar, index, val + nfin);

  field_scalar_array_set(array, index, val);
      }
    }
  }

  return 0;
}
