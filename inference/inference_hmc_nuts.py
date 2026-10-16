"""
inference_hmc_nuts.py
SU(8) HMC/NUTS inference using BlackJAX + JAX (no NumPyro).

This module provides a deterministic, single-entry HMC/NUTS driver
for the SU(8) surrogate forward model, returning NumPy arrays in the
format expected by executor.py.
"""

from typing import Dict, Any

import jax
import jax.numpy as jnp
from jax import random
import blackjax
import numpy as np


# ------------------------------------------------------------
# Log-posterior for SU(8) parameters
# ------------------------------------------------------------
def su8_logprob(params: jnp.ndarray, eht_data: Dict[str, Any]) -> jnp.ndarray:
    """
    Log-posterior for the SU(8) surrogate model.

    Parameters
    ----------
    params : jnp.ndarray
        1D parameter vector (length = design_matrix.shape[1]).
    eht_data : dict
        Structured EHT data with at least:
        - 'visibilities' : jnp.ndarray
        - 'errors'       : jnp.ndarray
        - 'design_matrix': jnp.ndarray

    Returns
    -------
    jnp.ndarray
        Scalar log-posterior value.
    """
    vis = eht_data["visibilities"]
    err = eht_data["errors"]
    design_matrix = eht_data["design_matrix"]

    model_vis = jnp.dot(design_matrix, params)

    loglik = -0.5 * jnp.sum(((vis - model_vis) / err) ** 2)
    logprior = -0.5 * jnp.sum(params ** 2)

    return loglik + logprior


# ------------------------------------------------------------
# Single-chain NUTS runner (JIT-compatible)
# ------------------------------------------------------------
def _run_chain(
    key: random.PRNGKey,
    init_params: jnp.ndarray,
    eht_data: Dict[str, Any],
    num_warmup: int,
    num_samples: int,
    step_size: float,
    inverse_mass_matrix: jnp.ndarray,
) -> jnp.ndarray:
    """
    Run a single NUTS chain with BlackJAX.

    Returns
    -------
    jnp.ndarray
        Array of shape (num_samples, dim) with sampled positions.
    """
    logprob_fn = lambda p: su8_logprob(p, eht_data)

    nuts = blackjax.nuts(
        logprob_fn,
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix,
    )

    state = nuts.init(init_params)

    def one_step(state, key):
        key, subkey = random.split(key)
        state, info = nuts.step(subkey, state)
        return state, info

    def one_sample(state, key):
        key, subkey = random.split(key)
        state, info = nuts.step(subkey, state)
        return state, state.position

    # Warmup
    key, warmup_key = random.split(key)
    state, _ = jax.lax.scan(
        one_step,
        state,
        random.split(warmup_key, num_warmup),
    )

    # Sampling
    key, sample_key = random.split(key)
    state, positions = jax.lax.scan(
        one_sample,
        state,
        random.split(sample_key, num_samples),
    )

    return positions


_run_chain_jit = jax.jit(_run_chain, static_argnums=(3, 4))


# ------------------------------------------------------------
# Public API: run_su8_hmc
# ------------------------------------------------------------
def run_su8_hmc(
    eht_data: Dict[str, Any],
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    step_size: float = 1e-3,
    rng_seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Deterministic HMC/NUTS driver for the SU(8) surrogate model.

    Parameters
    ----------
    eht_data : dict
        Structured EHT data with keys:
        - 'visibilities'
        - 'errors'
        - 'design_matrix'
    num_warmup : int, optional
        Number of warmup steps per chain.
    num_samples : int, optional
        Number of posterior samples per chain.
    num_chains : int, optional
        Number of independent chains.
    step_size : float, optional
        NUTS step size.
    rng_seed : int, optional
        Master RNG seed for reproducibility.

    Returns
    -------
    dict
        {
            "params": np.ndarray of shape (num_chains, num_samples, dim)
        }
    """
    design_matrix = eht_data["design_matrix"]
    dim = design_matrix.shape[1]

    init_params = jnp.zeros((dim,))
    inverse_mass_matrix = jnp.ones((dim,))  # diagonal metric

    master_key = random.PRNGKey(rng_seed)
    keys = random.split(master_key, num_chains)

    def run_chain_vmap(key):
        return _run_chain_jit(
            key,
            init_params,
            eht_data,
            num_warmup,
            num_samples,
            step_size,
            inverse_mass_matrix,
        )

    positions = jax.vmap(run_chain_vmap)(keys)  # (num_chains, num_samples, dim)
    samples_np = np.asarray(positions)

    return {"params": samples_np}


