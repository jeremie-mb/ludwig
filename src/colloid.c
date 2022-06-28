/*****************************************************************************
 *
 *  colloid.c
 *
 *  State type for particles including bounce-back on links, wetting,
 *  magnetic dipoles, and squirmers.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2021 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>

#include "colloid.h"

/*****************************************************************************
 *
 *  colloid_state_read_ascii
 *
 *  Returns zero on success.
 *
 *****************************************************************************/

int colloid_state_read_ascii(colloid_state_t * ps, FILE * fp) {

  int n;
  int nread = 0;
  int ifail = 0;

  const char * isformat = "%24d\n";
  const char * sformat  = "%24le\n";
  const char * vformat  = "%24le %24le %24le\n";

  assert(ps);
  assert(fp);

  nread += fscanf(fp, isformat, &ps->index);
  nread += fscanf(fp, isformat, &ps->rebuild);
  nread += fscanf(fp, isformat, &ps->nbonds);
  nread += fscanf(fp, isformat, &ps->nbonds2);
  nread += fscanf(fp, isformat, &ps->nbonds3);
  nread += fscanf(fp, isformat, &ps->nbonds_mesh);
  nread += fscanf(fp, isformat, &ps->nangles);
  nread += fscanf(fp, isformat, &ps->isfixedr);
  nread += fscanf(fp, isformat, &ps->isfixedv);
  nread += fscanf(fp, isformat, &ps->isfixedw);
  nread += fscanf(fp, isformat, &ps->isfixeds);
  nread += fscanf(fp, isformat, &ps->type);

  for (n = 0; n < NBOND_MAX; n++) {
    nread += fscanf(fp, isformat, &ps->bond[n]);
  }

  for (n = 0; n < NBOND_MAX2; n++) {
    nread += fscanf(fp, isformat, &ps->bond2[n]);
  }

  for (n = 0; n < NBOND_MAX3; n++) {
    nread += fscanf(fp, isformat, &ps->bond3[n]);
  }

  for (n = 0; n < NBOND_MAX_MESH; n++) {
    nread += fscanf(fp, isformat, &ps->bond_mesh[n]);
  }

  nread += fscanf(fp, isformat, &ps->rng);

  for (n = 0; n < 3; n++) {
    nread += fscanf(fp, isformat, ps->isfixedrxyz + n);
  }
  for (n = 0; n < 3; n++) {
    nread += fscanf(fp, isformat, ps->isfixedvxyz + n);
  }

  nread += fscanf(fp, isformat, &ps->inter_type);
/* -----> CHEMOVESICLE V2 */
  nread += fscanf(fp, isformat, &ps->iscentre);
  nread += fscanf(fp, isformat, &ps->indexcentre);
  nread += fscanf(fp, isformat, &ps->ishole);
/* <----- */

  for (n = 0; n < NPAD_INT; n++) {
    nread += fscanf(fp, isformat, &ps->intpad[n]);
  }

  nread += fscanf(fp, sformat, &ps->a0);
  nread += fscanf(fp, sformat, &ps->ah);
  nread += fscanf(fp, vformat, &ps->r[0], &ps->r[1], &ps->r[2]);
  nread += fscanf(fp, vformat, &ps->v[0], &ps->v[1], &ps->v[2]);
  nread += fscanf(fp, vformat, &ps->w[0], &ps->w[1], &ps->w[2]);
  nread += fscanf(fp, vformat, &ps->s[0], &ps->s[1], &ps->s[2]);
  nread += fscanf(fp, vformat, &ps->m[0], &ps->m[1], &ps->m[2]);
  nread += fscanf(fp, vformat, &ps->n[0], &ps->n[1], &ps->n[2]);
  nread += fscanf(fp, sformat, &ps->b1);
  nread += fscanf(fp, sformat, &ps->b2);
  nread += fscanf(fp, sformat, &ps->c);
  nread += fscanf(fp, sformat, &ps->h);
  nread += fscanf(fp, vformat, &ps->dr[0], &ps->dr[1], &ps->dr[2]);
  nread += fscanf(fp, sformat, &ps->deltaphi);

  nread += fscanf(fp, sformat, &ps->q0);
  nread += fscanf(fp, sformat, &ps->q1);
  nread += fscanf(fp, sformat, &ps->epsilon);

  nread += fscanf(fp, sformat, &ps->deltaq0);
  nread += fscanf(fp, sformat, &ps->deltaq1);
  nread += fscanf(fp, sformat, &ps->sa);
  nread += fscanf(fp, sformat, &ps->saf);
  nread += fscanf(fp, sformat, &ps->al);

/* -----> CHEMOVESICLE V2 */
/* Parameters of the interaction and possibility to create PHI */
  nread += fscanf(fp, sformat, &ps->u0);
  nread += fscanf(fp, sformat, &ps->delta);
  nread += fscanf(fp, sformat, &ps->cutoff);
  nread += fscanf(fp, vformat, &ps->fsub[0], &ps->fsub[1], &ps->fsub[2]);
  nread += fscanf(fp, vformat, &ps->fphi[0], &ps->fphi[1], &ps->fphi[2]);
  nread += fscanf(fp, vformat, &ps->fsprings[0], &ps->fsprings[1], &ps->fsprings[2]);
/* <----- */

  for (n = 0; n < NPAD_DBL; n++) {
    nread += fscanf(fp, sformat, &ps->dpad[n]);
  }

  for (n = 0; n < 7; n++) {
    nread += fscanf(fp, isformat, &ps->tuple.indices[n]);
    nread += fscanf(fp, sformat, &ps->tuple.r0s[n]);
  }  //LIGHTHOUSE 7 is the maximum number of springs for the edge particles

  if (nread != NTOT_VAR) ifail = 1;

  /* If assertions are off, we may want to catch this failure elsewhere */
  assert(ifail == 0);

  /* Always set the rebuild flag (even if file has zero) */

  ps->rebuild = 1;

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_state_read_binary
 *
 *  Returns zero on success.
 *
 *****************************************************************************/

int colloid_state_read_binary(colloid_state_t * ps, FILE * fp) {

  int nread;

  assert(ps);
  assert(fp);

  nread = fread(ps, sizeof(colloid_state_t), 1, fp);

  /* Always set the rebuild flag (even if file has zero) */

  ps->rebuild = 1;

  return (1 - nread);
}

/*****************************************************************************
 *
 *  colloid_state_write_ascii
 *
 *  Returns zero on success.
 *
 *****************************************************************************/

int colloid_state_write_ascii(const colloid_state_t * s, FILE * fp) {

  int n;
  int nwrite = 0;
  int ifail = 0;

  const char * isformat = "%24d\n";
  const char * sformat  = "%24.15e\n";
  const char * vformat  = "%24.15e %24.15e %24.15e\n";

  assert(s);
  assert(fp);
  nwrite += fprintf(fp, isformat, s->index);
  nwrite += fprintf(fp, isformat, s->rebuild);
  nwrite += fprintf(fp, isformat, s->nbonds);
  nwrite += fprintf(fp, isformat, s->nbonds2);
  nwrite += fprintf(fp, isformat, s->nbonds3);
  nwrite += fprintf(fp, isformat, s->nbonds_mesh);
  nwrite += fprintf(fp, isformat, s->nangles);
  nwrite += fprintf(fp, isformat, s->isfixedr);
  nwrite += fprintf(fp, isformat, s->isfixedv);
  nwrite += fprintf(fp, isformat, s->isfixedw);
  nwrite += fprintf(fp, isformat, s->isfixeds);
  nwrite += fprintf(fp, isformat, s->type);

  for (n = 0; n < NBOND_MAX; n++) {
    nwrite += fprintf(fp, isformat, s->bond[n]);
  }

  for (n = 0; n < NBOND_MAX2; n++) {
    nwrite += fprintf(fp, isformat, s->bond2[n]);
  }

  for (n = 0; n < NBOND_MAX3; n++) {
    nwrite += fprintf(fp, isformat, s->bond3[n]);
  }

  for (n = 0; n < NBOND_MAX_MESH; n++) {
    nwrite += fprintf(fp, isformat, s->bond_mesh[n]);
  }

  nwrite += fprintf(fp, isformat, s->rng);

  /* isfixedrxyz and isfixedvxyz are writen as 3 x scalars as they
   * have replaced padding */

  for (n = 0; n < 3; n++) {
    nwrite += fprintf(fp, isformat, s->isfixedrxyz[n]);
  }
  for (n = 0; n < 3; n++) {
    nwrite += fprintf(fp, isformat, s->isfixedvxyz[n]);
  }

  nwrite += fprintf(fp, isformat, s->inter_type);

/* -----> CHEMOVESICLE V2 */
/* Attributes of edge/central vesicles */
  nwrite += fprintf(fp, isformat, s->iscentre);
  nwrite += fprintf(fp, isformat, s->indexcentre);
  nwrite += fprintf(fp, isformat, s->ishole);
/* <----- */

  for (n = 0; n < NPAD_INT; n++) {
    nwrite += fprintf(fp, isformat, s->intpad[n]);
  }

  nwrite += fprintf(fp, sformat, s->a0);
  nwrite += fprintf(fp, sformat, s->ah);
  nwrite += fprintf(fp, vformat, s->r[0], s->r[1], s->r[2]);
  nwrite += fprintf(fp, vformat, s->v[0], s->v[1], s->v[2]);
  nwrite += fprintf(fp, vformat, s->w[0], s->w[1], s->w[2]);
  nwrite += fprintf(fp, vformat, s->s[0], s->s[1], s->s[2]);
  nwrite += fprintf(fp, vformat, s->m[0], s->m[1], s->m[2]);
  nwrite += fprintf(fp, vformat, s->n[0], s->n[1], s->n[2]);
  nwrite += fprintf(fp, sformat, s->b1);
  nwrite += fprintf(fp, sformat, s->b2);
  nwrite += fprintf(fp, sformat, s->c);
  nwrite += fprintf(fp, sformat, s->h);
  nwrite += fprintf(fp, vformat, s->dr[0], s->dr[1], s->dr[2]);
  nwrite += fprintf(fp, sformat, s->deltaphi);

  nwrite += fprintf(fp, sformat, s->q0);
  nwrite += fprintf(fp, sformat, s->q1);
  nwrite += fprintf(fp, sformat, s->epsilon);

  nwrite += fprintf(fp, sformat, s->deltaq0);
  nwrite += fprintf(fp, sformat, s->deltaq1);
  nwrite += fprintf(fp, sformat, s->sa);
  nwrite += fprintf(fp, sformat, s->saf);
  nwrite += fprintf(fp, sformat, s->al);

/* -----> CHEMOVESICLE V2 */
  nwrite += fprintf(fp, sformat, s->u0);
  nwrite += fprintf(fp, sformat, s->delta);
  nwrite += fprintf(fp, sformat, s->cutoff);
  nwrite += fprintf(fp, vformat, s->fsub[0], s->fsub[1], s->fsub[2]);
  nwrite += fprintf(fp, vformat, s->fphi[0], s->fphi[1], s->fphi[2]);
  nwrite += fprintf(fp, vformat, s->fsprings[0], s->fsprings[1], s->fsprings[2]);
/* <----- */

  for (n = 0; n < NPAD_DBL; n++) {
    nwrite += fprintf(fp, sformat, s->dpad[n]);
  }

  for (n = 0; n < 7; n++) {
    nwrite += fprintf(fp, isformat, s->tuple.indices[n]);
    nwrite += fprintf(fp, sformat, s->tuple.r0s[n]);
  } 

  /* ... should be NTOT_VAR items of format + 1 characters */

  if (nwrite != NTOT_VAR*25) ifail = 1;
  /* If assertions are off, responsibility passes to caller */
  
  assert(ifail == 0);

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_state_write_binary
 *
 *  Returns zero on success.
 *
 *****************************************************************************/

int colloid_state_write_binary(const colloid_state_t * s, FILE * fp) {

  int nwrite;

  assert(s);
  assert(fp);

  nwrite = fwrite(s, sizeof(colloid_state_t), 1, fp);

  return (1 - nwrite);
}
