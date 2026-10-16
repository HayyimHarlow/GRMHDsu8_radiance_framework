/* harm_su8_patch.c
 * SU(8 radiance-channel source-term patch for HARM/BHAC.
 * This assumes eps_su8 and eps_ac are tracked as additional
 * primitive variables or derived from existing fields.
 */

#include "harm_defs.h"
#include "harm_globals.h"

/* SU(8 parameters (to be set in parameter file or runtime config) */
double kappa_ac;   /* acoustic coupling */
double gamma_ac;   /* acoustic damping */
double alpha_su8;  /* radiance-channel momentum coupling */
double lambda_ac;  /* magnetic damping */

/* Compute SU(8 source terms for energy and momentum */
void su8_source_terms(const double *prim, const double *grad_eps_su8,
                      double *dU, double *dMom, double *dB2,
                      double dt)
{
    /* Extract SU(8 energy densities from primitive array or state */
    double eps_su8 = prim[EPS_SU8];  /* index to be defined */
    double eps_ac  = prim[EPS_AC];   /* index to be defined */

    /* Radiance-channel energy injection */
    double J_sup = kappa_ac * (eps_su8 - eps_ac);

    /* Acoustic bleed-off */
    double Gamma_ac = gamma_ac * eps_ac;

    /* Net SU(8 contribution to internal energy */
    double dU_su8 = (J_sup - Gamma_ac) * dt;
    *dU += dU_su8;

    /* Radiance-channel momentum correction */
    int i;
    for (i = 0; i < NDIM; i++) {
        dMom[i] += alpha_su8 * grad_eps_su8[i] * dt;
    }

    /* Magnetic energy damping */
    *dB2 -= lambda_ac * eps_ac * dt;

    /* Stability checks (optional: abort or flag cell) */
    if (eps_su8 < 0.0 || eps_ac < 0.0) {
        fprintf(stderr, "SU(8 stability violation: eps_su8=%g eps_ac=%g\n",
                eps_su8, eps_ac);
        /* handle according to code policy: floor, flag, or abort */
    }
}

/* Modified CFL condition including acoustic speed */
double su8_cfl_dt(double dx, double c_mhd, double eps_ac)
{
    double c_ac = sqrt(fabs(gamma_ac * eps_ac));
    double c_max = (c_mhd > c_ac) ? c_mhd : c_ac;
    return cfl_factor * dx / c_max;
}
