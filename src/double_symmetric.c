/****************************************************************************
 *
 *  fe_double_symmetric.c
 *
 *  Vector of two binary binary mixtures 
 *
 *  Two order parameters are required:
 *
 *  [0] \phi is compositional order parameter (cf symmetric free energy)
 *  [1] \psi is compositional order parameter (cf symmetric free energy)
 *
 *  The free energy density is:
 *
 *    F = F_\phi + F_\psi 
 *
 *  with
 *
 *    F_phi  = symmetric phi^4 free energy
 *    F_psi  = symmetric psi^4 free energy
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "double_symmetric.h"

/* Virtual function table (host) */

static fe_vt_t fe_double_symmetric_hvt = {
  (fe_free_ft)      fe_double_symmetric_free,     /* Virtual destructor */
  (fe_target_ft)    fe_double_symmetric_target,   /* Return target pointer */
  (fe_fed_ft)       fe_double_symmetric_fed,      /* Free energy density */
  (fe_mu_ft)        fe_double_symmetric_mu,       /* Chemical potential */
  (fe_mu_solv_ft)   NULL,
  (fe_str_ft)       fe_double_symmetric_str,      /* Total stress */
  (fe_str_ft)       fe_double_symmetric_str,      /* Symmetric stress */
  (fe_str_ft)       NULL,             /* Antisymmetric stress (not relevant) */
  (fe_hvector_ft)   NULL,             /* Not relevant */
  (fe_htensor_ft)   NULL,             /* Not relevant */
  (fe_htensor_v_ft) NULL,             /* Not reelvant */
  (fe_stress_v_ft)  fe_double_symmetric_str_v,    /* Total stress (vectorised version) */
  (fe_stress_v_ft)  fe_double_symmetric_str_v,    /* Symmetric part (vectorised) */
  (fe_stress_v_ft)  NULL              /* Antisymmetric part */
};


static __constant__ fe_double_symmetric_param_t const_param;

/****************************************************************************
 *
 *  fe_double_symmetric_create
 *
 ****************************************************************************/

int fe_double_symmetric_create(pe_t * pe, cs_t * cs, field_t * phi,
		    field_grad_t * dphi, fe_double_symmetric_param_t param,
		    fe_double_symmetric_t ** fe) {
  int ndevice;
  fe_double_symmetric_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(fe);
  assert(phi);
  assert(dphi);

  obj = (fe_double_symmetric_t *) calloc(1, sizeof(fe_double_symmetric_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(fe_double_symmetric_t) failed\n");

  obj->param = (fe_double_symmetric_param_t *) calloc(1, sizeof(fe_double_symmetric_param_t));
  assert(obj->param);
  if (obj->param == NULL) pe_fatal(pe, "calloc(fe_double_symmetric_param_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;
  obj->phi = phi;
  obj->dphi = dphi;
  obj->super.func = &fe_double_symmetric_hvt;
  obj->super.id = FE_DOUBLE_SYMMETRIC;

  /* Allocate target memory, or alias */

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    fe_double_symmetric_param_set(obj, param);
    obj->target = obj;
  }
  else {
    fe_double_symmetric_param_t * tmp;
    tdpMalloc((void **) &obj->target, sizeof(fe_double_symmetric_t));
    tdpGetSymbolAddress((void **) &tmp, tdpSymbol(const_param));
    tdpMemcpy(&obj->target->param, tmp, sizeof(fe_double_symmetric_param_t *),
	      tdpMemcpyHostToDevice);
    /* Now copy. */
    assert(0); /* No implementation */
  }

  *fe = obj;

  return 0;
}

/****************************************************************************
 *
 *  fe_double_symmetric_free
 *
 ****************************************************************************/

__host__ int fe_double_symmetric_free(fe_double_symmetric_t * fe) {

  int ndevice;

  assert(fe);

  tdpGetDeviceCount(&ndevice);
  if (ndevice > 0) tdpFree(fe->target);

  free(fe->param);
  free(fe);

  return 0;
}

/****************************************************************************
 *
 *  fe_double_symmetric_info
 *
 *  Some information on parameters etc.
 *
 ****************************************************************************/

__host__ int fe_double_symmetric_info(fe_double_symmetric_t * fe) {

  double sigma0, sigma1, xi0, xi1;
  double psi_c;
  pe_t * pe = NULL;

  assert(fe);

  pe = fe->pe;

  fe_double_symmetric_sigma(fe, &sigma0, &sigma1);
  fe_double_symmetric_xi0(fe, &xi0, &xi1);

  pe_info(pe, "Surfactant free energy parameters:\n");
  pe_info(pe, "Bulk parameter phi_A      = %12.5e\n", fe->param->a0);
  pe_info(pe, "Bulk parameter phi_B      = %12.5e\n", fe->param->b0);
  pe_info(pe, "Surface penalty phi_kappa = %12.5e\n", fe->param->kappa0);
  pe_info(pe, "Bulk parameter psi_A      = %12.5e\n", fe->param->a1);
  pe_info(pe, "Bulk parameter psi_B      = %12.5e\n", fe->param->b1);
  pe_info(pe, "Surface penalty psi_kappa = %12.5e\n", fe->param->kappa1);


  pe_info(pe, "\n");
  pe_info(pe, "Derived quantities\n");
  pe_info(pe, "Interfacial tension sigma0   = %12.5e\n", sigma0);
  pe_info(pe, "Interfacial width xi0    = %12.5e\n", xi0);
  pe_info(pe, "Interfacial tension sigma1  = %12.5e\n", sigma1);
  pe_info(pe, "Interfacial width xi1    = %12.5e\n", xi1);

  return 0;
}

/****************************************************************************
 *
 *  fe_double_symmetric_target
 *
 ****************************************************************************/

__host__ int fe_double_symmetric_target(fe_double_symmetric_t * fe, fe_t ** target) {

  assert(fe);
  assert(target);

  *target = (fe_t *) fe->target;

  return 0;
}

/****************************************************************************
 *
 *  fe_double_symmetric_param_set
 *
 ****************************************************************************/

__host__ int fe_double_symmetric_param_set(fe_double_symmetric_t * fe, fe_double_symmetric_param_t vals) {

  assert(fe);

  *fe->param = vals;

  return 0;
}

/*****************************************************************************
 *
 *  fe_double_symmetric_param
 *
 *****************************************************************************/

__host__ int fe_double_symmetric_param(fe_double_symmetric_t * fe, fe_double_symmetric_param_t * values) {
  assert(fe);

  *values = *fe->param;

  return 0;
}

/****************************************************************************
 *
 *  fe_double_symmetric_sigma
 *
 *  Assumes phi^* = (-a/b)^1/2
 *
 ****************************************************************************/

__host__ int fe_double_symmetric_sigma(fe_double_symmetric_t * fe,  double * sigma0, double * sigma1) {

  double a0, b0, kappa0;
  double a1, b1, kappa1;

  assert(fe);
  assert(sigma0);
  assert(sigma1);

  a0 = fe->param->a0;
  b0 = fe->param->b0;
  kappa0 = fe->param->kappa0;

  a1 = fe->param->a1;
  b1 = fe->param->b1;
  kappa1 = fe->param->kappa1;

  *sigma0 = sqrt(-8.0*kappa0*a0*a0*a0/(9.0*b0*b0));
  *sigma1 = sqrt(-8.0*kappa1*a1*a1*a1/(9.0*b1*b1));

  return 0;
}

/****************************************************************************
 *
 *  fe_double_symmetric_xi0
 *
 *  Interfacial width.
 *
 ****************************************************************************/

__host__ int fe_double_symmetric_xi0(fe_double_symmetric_t * fe, double * xi0, double * xi1) {

  assert(fe);
  assert(xi0);
  assert(xi1);

  *xi0 = sqrt(-2.0*fe->param->kappa0/fe->param->a0);
  *xi1 = sqrt(-2.0*fe->param->kappa1/fe->param->a1);

  return 0;
}


/****************************************************************************
 *
 *  fe_double_symmetric_fed
 *
 *  This is:
 *     (1/2) a0 \phi^2 + (1/4) b0 \phi^4 + (1/2) kappa0 (\nabla\phi)^2
 *   + (1/2) a1 \psi^2 + (1/4) b1 \psi^4 + (1/2) kappa1 (\nabla\psi)^2
 *
 ****************************************************************************/

__host__ int fe_double_symmetric_fed(fe_double_symmetric_t * fe, int index, double * fed) {

  double field[2];
  double phi;
  double psi;
  double dphi[2][3];
  double dphisq, dpsisq;

  assert(fe);

  field_scalar_array(fe->phi, index, field);
  field_grad_pair_grad(fe->dphi, index, dphi);

  phi = field[0];
  psi = field[1];

  dphisq = dphi[0][X]*dphi[0][X] + dphi[0][Y]*dphi[0][Y]
         + dphi[0][Z]*dphi[0][Z];

  dpsisq = dphi[1][X]*dphi[1][X] + dphi[1][Y]*dphi[1][Y]
         + dphi[1][Z]*dphi[1][Z];

  /* We have the symmetric piece followed by terms in psi */

  *fed = 0.5*fe->param->a0*phi*phi + 0.25*fe->param->b0*phi*phi*phi*phi
    + 0.5*fe->param->kappa0*dphisq;

  *fed += 0.5*fe->param->a1*psi*psi + 0.25*fe->param->b1*psi*psi*psi*psi
    + 0.5*fe->param->kappa1*dpsisq;

  return 0;
}

/****************************************************************************
 *
 *  fe_double_symmetric_mu
 * 
 *  Two chemical potentials are present:
 *
 *  \mu_\phi = a0 \phi + b0 \phi^3 - kappa0 \nabla^2 \phi
 *
 *  \mu_\psi = a1 \psi + b1 \psi^3 - kappa1 \nabla^2 \psi
 * 
 *
 ****************************************************************************/

__host__ int fe_double_symmetric_mu(fe_double_symmetric_t * fe, int index, double * mu) {

  double phi;
  double psi;
  double field[2];
  double grad[2][3];
  double delsq[2];

  assert(fe);
  assert(mu); assert(mu + 1);

  field_scalar_array(fe->phi, index, field);
  field_grad_pair_grad(fe->dphi, index, grad);
  field_grad_pair_delsq(fe->dphi, index, delsq);

  phi = field[0];
  psi = field[1];

  /* mu_phi */

  mu[0] = fe->param->a0*phi + fe->param->b0*phi*phi*phi
    - fe->param->kappa0*delsq[0];

  /* mu_psi */

  mu[1] = fe->param->a1*psi + fe->param->b1*psi*psi*psi
    - fe->param->kappa1*delsq[1];

  return 0;
}

/****************************************************************************
 *
 *  fe_double_symmetric_str
 *
 *  Thermodynamic stress S_ab = p0 delta_ab + P_ab
 *
 *  p0 = (1/2) A \phi^2 + (3/4) B \phi^4 - (1/2) \kappa \nabla^2 \phi
 *     - (1/2) kappa (\nabla phi)^2
 *
 *  P_ab = \kappa \nabla_a \phi \nabla_b \phi + same with psip
 *
 ****************************************************************************/

__host__ int fe_double_symmetric_str(fe_double_symmetric_t * fe, int index, double s[3][3]) {

  int ia, ib;
  double field[2];
  double phi;
  double psi;
  double delsq[2];
  double grad[2][3];
  double p0;
  KRONECKER_DELTA_CHAR(d);

  assert(fe);

  field_scalar_array(fe->phi, index, field);
  field_grad_pair_grad(fe->dphi, index, grad);
  field_grad_pair_delsq(fe->dphi, index, delsq);

  phi = field[0];
  psi = field[1];

  p0 = 0.5*fe->param->a0*phi*phi + 0.75*fe->param->b0*phi*phi*phi*phi
    - fe->param->kappa0*(phi*delsq[0] - 0.5*dot_product(grad[0], grad[0]))

     + 0.5*fe->param->a1*psi*psi + 0.75*fe->param->b1*psi*psi*psi*psi
    - fe->param->kappa1*(psi*delsq[1] - 0.5*dot_product(grad[1], grad[1]));

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = p0*d[ia][ib]
	    + fe->param->kappa0*grad[0][ia]*grad[0][ib];
	    + fe->param->kappa1*grad[1][ia]*grad[1][ib];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_double_symmetric_str_v
 *
 *  Stress (vectorised version) Currently a patch-up.
 *
 *****************************************************************************/

int fe_double_symmetric_str_v(fe_double_symmetric_t * fe, int index, double s[3][3][NSIMDVL]) {

  int ia, ib;
  int iv;
  double s1[3][3];

  assert(fe);

  for (iv = 0; iv < NSIMDVL; iv++) {
    fe_double_symmetric_str(fe, index + iv, s1);
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
	s[ia][ib][iv] = s1[ia][ib];
      }
    }
  }

  return 0;
}
