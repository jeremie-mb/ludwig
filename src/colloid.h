/*****************************************************************************
 *
 *  colloid.h
 *
 *  The implementation is exposed for the time being.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_COLLOID_H
#define LUDWIG_COLLOID_H

#include <stdio.h>

/* Tag to describe I/O format version appearing in files */

enum colloid_io_version {COLLOID_IO_VERSION = 0200};
typedef enum colloid_io_version colloid_io_version_t;

/* These describe the padding etc, and are really for internal
 * unit test consumption. The total number of variables is
 * useful to know to check the ASCII read/write. */

#define NTOT_VAR (32+48 + 9) /* force[3], torque[3], t0[3] */
#define NPAD_INT  9 /* 13 - shape - isfixedwxyz[3] */
#define NPAD_DBL  0 /* 15 - tumbletheta - tumblephi - mu - alpha - n[3] - alpha_pacman_mn - alpha_pacman_mp - deltapsi  - lm_rectangle - ln_rectangle - lp_rectangle - alpha_vesicle - width_vesicle */
#define NBOND_MAX  2

enum colloid_type_enum {COLLOID_TYPE_DEFAULT = 0,
			COLLOID_TYPE_ACTIVE,
			COLLOID_TYPE_SUBGRID,
			COLLOID_TYPE_JANUS};

enum colloid_shape_enum {COLLOID_SHAPE_DEFAULT = 0,
			COLLOID_SHAPE_PACMAN,
      COLLOID_SHAPE_RECTANGLE,
      COLLOID_SHAPE_VESICLE};

typedef enum colloid_type_enum colloid_type_enum_t;
typedef struct colloid_state_type colloid_state_t;

struct colloid_state_type {

  int index;            /* Unique global index for colloid */
  int rebuild;          /* Rebuild flag */
  int nbonds;           /* Number of bonds e.g. fene (to NBOND_MAX) */
  int nangles;          /* Number of angles, e.g., fene (1 at the moment) */

  int isfixedr;         /* Set to 1 for no position update */
  int isfixedv;         /* Set to 1 for no velocity update */
  int isfixedw;         /* Set to 1 for no angular velocity update */
  int isfixeds;         /* Set to zero for no s, m update */

  int type;             /* Particle type */
  int shape;             /* Particle shape */
  int bond[NBOND_MAX];  /* Bonded neighbours ids (index) */

  int rng;              /* Random number state */

  int isfixedrxyz[3];   /* Position update in specific coordinate directions */
  int isfixedvxyz[3];   /* Velocity update in specific coordinate directions */
  int isfixedwxyz[3];   /* Velocity update in specific coordinate directions */

  int inter_type;         /* Interaction type of a particle */

  /* New integer additions can be immediately before the padding */
  /* This should allow existing binary files to be read correctly */

  int intpad[NPAD_INT]; /* I'm going to pad to 32 ints to allow for future
			 * expansion. Additions should be appended here,
			 * and the padding reduced appropriately. */

  double a0;            /* Input radius (lattice units) */
  double ah;            /* Hydrodynamic radius (from calibration) */
  double r[3];          /* Position */
  double v[3];          /* Velocity */
  double w[3];          /* Angular velocity omega */
  double s[3];          /* Magnetic dipole, or spin */
  double m[3];          /* Primary orientation of colloid (used for building)*/
  double n[3];          /* Secondary orientation of colloid */
  double b1;	        /* squirmer active parameter b1 */
  double b2;            /* squirmer active parameter b2 */
  double c;             /* Wetting free energy parameter C */
  double h;             /* Wetting free energy parameter H */
  double dr[3];         /* r update (pending refactor of move/build process) */
  double deltaphi;      /* used to conserve phi[0] when the colloid moves */
  double deltapsi;      /* used to conserve phi[1] when the colloid moves */

  /* Charges. We allow two charge valencies (cf a general number
   * number in the electrokinetics section). q0 will be associated
   * with psi->rho[0] and q1 to psi->rho[1] in the electrokinetics.
   * The charge will
   * be converted to a density by dividing by the current discrete
   * volume to ensure conservation. */

  double q0;            /* magnitude charge 0 */
  double q1;            /* magnitude charge 1 */
  double epsilon;       /* permittivity */

  double deltaq0;       /* surplus/deficit of charge 0 at change of shape */
  double deltaq1;       /* surplus/deficit of charge 1 at change of shape */
  double sa;            /* surface area (finite difference) */
  double saf;           /* surface area to fluid (finite difference grid) */

  double al;            /* Offset parameter used for subgrid particles */
  double tumbletheta;
  double tumblephi;
  double mu_phoretic;
  double alpha_prod;
  double alpha_pacman_mn;
  double alpha_pacman_mp;
  double alpha_vesicle;
  double width_vesicle;
  double lm_rectangle;
  double ln_rectangle;
  double lp_rectangle;
  double force[3];
  double t0[3];
  double torque[3];
    double dpad[NPAD_DBL];/* Again, this pads to 512 bytes to allow
			 * for future expansion. */
};

int colloid_state_read_ascii(colloid_state_t * ps, FILE * fp);
int colloid_state_read_binary(colloid_state_t * ps, FILE * fp);
int colloid_state_write_ascii(const colloid_state_t * ps, FILE * fp);
int colloid_state_write_binary(const colloid_state_t * ps, FILE * fp);

#endif
