/* ================================
   SU(8 CFL Modification
   File: src/phys/timestep.c
   ================================ */

#include "su8_params.h"

double compute_dt(const struct of_state *q, double dx)
{
    double c_mhd = q->cmax;
    double eps_ac = q->prim[EPS_AC];

    double c_ac = sqrt(fabs(gamma_ac * eps_ac));
    double c_eff = (c_mhd > c_ac) ? c_mhd : c_ac;

    return cfl * dx / c_eff;
}
