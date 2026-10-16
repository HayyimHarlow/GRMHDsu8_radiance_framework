/* su8_emissivity.c
 * SU(8 radiance-channel modification to synchrotron emissivity
 * and polarization for ipole/RAPTOR-style GRRT codes.
 */

#include "grrt_defs.h"
#include "grrt_globals.h"

/* SU(8 parameters */
double kappa_ac;
double gamma_ac;
double alpha_su8;

/* Base synchrotron emissivity (existing implementation) */
double j_nu_synch(const double B, const double ne, const double nu);

/* SU(8-modified emissivity */
double j_nu_su8(const double B, const double ne, const double nu,
                const double eps_su8, const double eps_ac)
{
    double j_base = j_nu_synch(B, ne, nu);

    /* Radiance-channel enhancement/suppression factor */
    double R_su8 = 1.0 + kappa_ac * (eps_su8 - eps_ac);

    /* Acoustic damping factor */
    double D_ac = 1.0 / (1.0 + gamma_ac * eps_ac);

    return j_base * R_su8 * D_ac;
}

/* EVPA modification via SU(8 radiance channel */
double evpa_su8(const double Q, const double U,
                const double eps_su8, const double eps_ac)
{
    double evpa_base = 0.5 * atan2(U, Q);

    /* Small SU(8-induced rotation */
    double delta_evpa = alpha_su8 * (eps_su8 - eps_ac);

    return evpa_base + delta_evpa;
}
