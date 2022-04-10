/*****************************************************************************
 *
 *  subgrid.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2021 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_SUBGRID_H
#define LUDWIG_SUBGRID_H

#include "colloids.h"
#include "hydro.h"
#include "wall.h"
#include "field.h"

int subgrid_update(colloids_info_t * cinfo, hydro_t * hydro, int noise_flag);

/* -----> CHEMOVESICLE V2 */
int subgrid_centre_update(colloids_info_t * cinfo, hydro_t * hydro, int noise_flag);
int subgrid_centre_update2(colloids_info_t * cinfo, hydro_t * hydro, int noise_flag);
int subgrid_phi_production(colloids_info_t * cinfo, field_t * phi);
int subgrid_mobility_map(colloids_info_t * cinfo, field_t * mobility_map, rt_t * rt);
int subgrid_mobility_map_vesicle2(colloids_info_t * cinfo, field_t * mobility_map, rt_t * rt);
/* <----- */


int subgrid_force_from_particles(colloids_info_t * cinfo, hydro_t * hydro,
				 wall_t * wall);
int subgrid_wall_lubrication(colloids_info_t * cinfo, wall_t * wall);

#endif
