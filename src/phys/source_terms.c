/* ================================
   SU(8 Radiance-Channel Source Terms
   File: src/phys/source_terms.c
   ================================ */

#include "decs.h"
#include "su8_params.h"   /* new header */

void add_source_terms(double *dU, double dUgeom[NDIM],
                      const struct of_state *q,
                      const struct of_geom *geom,
                      double dt)
{
    /* Existing GRMHD source terms here */

    /* --- SU(8) energy densities --- */
    double eps_su8 = q->prim[EPS_SU8];
    double eps_ac  = q->prim[EPS_AC];

    /* Radiance-channel energy injection */
    double J_sup = kappa_ac * (eps_su8 - eps_ac);

    /* Acoustic bleed-off */
    double Gamma_ac = gamma_ac * eps_ac;

    /* Net SU(8 energy update */
    dU[UU] += (J_sup - Gamma_ac) * dt;

    /* --- SU(8 momentum correction --- */
    for (int mu = 1; mu < NDIM; mu++) {
        dU[U1 + mu - 1] += alpha_su8 * q->grad_eps_su8[mu] * dt;
    }

    /* --- SU(8 magnetic damping --- */
    dU[UB2] -= lambda_ac * eps_ac * dt;

    /* Stability check */
    if (eps_su8 < 0 || eps_ac < 0) {
        fprintf(stderr, "SU(8 stability violation at zone\n");
    }
}
/* ================================
   SU(8 Radiance-Channel Source Terms
   File: src/phys/source_terms.c
   ================================ */

#include "decs.h"
#include "su8_params.h"   /* new header */

void add_source_terms(double *dU, double dUgeom[NDIM],
                      const struct of_state *q,
                      const struct of_geom *geom,
                      double dt)
{
    /* Existing GRMHD source terms here */

    /* --- SU(8) energy densities --- */
    double eps_su8 = q->prim[EPS_SU8];
    double eps_ac  = q->prim[EPS_AC];

    /* Radiance-channel energy injection */
    double J_sup = kappa_ac * (eps_su8 - eps_ac);

    /* Acoustic bleed-off */
    double Gamma_ac = gamma_ac * eps_ac;

    /* Net SU(8 energy update */
    dU[UU] += (J_sup - Gamma_ac) * dt;

    /* --- SU(8 momentum correction --- */
    for (int mu = 1; mu < NDIM; mu++) {
        dU[U1 + mu - 1] += alpha_su8 * q->grad_eps_su8[mu] * dt;
    }

    /* --- SU(8 magnetic damping --- */
    dU[UB2] -= lambda_ac * eps_ac * dt;

    /* Stability check */
    if (eps_su8 < 0 || eps_ac < 0) {
        fprintf(stderr, "SU(8 stability violation at zone\n");
    }
}
