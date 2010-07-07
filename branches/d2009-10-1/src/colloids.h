/*****************************************************************************
 *
 *  colloids.h
 *
 *  Data structures holding linked list of colloids.
 *
 *  $Id: colloids.h,v 1.9.2.9 2010-07-07 11:03:25 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef COLLOIDS_H
#define COLLOIDS_H

#include "colloid.h"
#include "colloid_link.h"

typedef struct colloid Colloid;
typedef struct colloid colloid_t;

struct colloid {

  colloid_state_t s;

  /* AUXILARY */

  double random[6];     /* Random numbers for MC/Brownian dynamics */
  double force[3];      /* Total force on colloid */
  double torque[3];     /* Total torque on colloid */
  double f0[3];         /* Velocity independent force */
  double t0[3];         /* Velocity independent torque */
  double cbar[3];       /* Mean boundary link vector */
  double rxcbar[3];     /* Mean r_b x c_b */
  double deltam;        /* Mass difference owing to change in shape */
  double sumw;          /* Sum of weights over links */
  double zeta[21];      /* Upper triangle of 6x6 drag matrix zeta */
  double stats[3];      /* Particle statisitics */
  double fc0[3];        /* total force on squirmer for mass conservation */
  double tc0[3];        /* total torque on squirmer for mass conservation */
  double sump;          /* flux through squirmer surface */ 

  /* Pointers */

  colloid_link_t * lnk; /* Pointer to the list of links defining surface */
  Colloid   * next;     /* colloid is a linked list */

};

void      colloids_init(void);
void      colloids_finish(void);
void      colloids_ntotal_set(void);
void      colloids_cell_ncell_set(const int ncell[3]);
void      colloids_cell_ncell(int ncell[3]);
void      colloids_cell_coords(const double r[3], int icell[3]);
void      colloids_cell_insert_colloid(Colloid *);
void      colloids_cell_update(void);
int       colloids_nalloc(void);
int       Ncell(const int dim);
double    colloids_lcell(const int dim);
Colloid * colloids_cell_list(const int, const int, const int);

Colloid * colloid_allocate(void);
Colloid * colloid_add_local(const int index, const double r[3]);
Colloid * colloid_add(const int index, const double r[3]);
void      colloid_free(Colloid *);
double    colloid_rho0(void);
int       colloid_ntotal(void);
int       colloid_nlocal(void);

#endif
