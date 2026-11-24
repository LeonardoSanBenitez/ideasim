Configuration
=============

ideasim is configured via JSON files. Export the default configuration to see
all available parameters::

    ideasim default-config -o config.json

Top-Level Structure
-------------------

Below, N is the number of ideologies and M the number of environmental
variables. The configuration file has the following top-level keys (all flat):

- ``ideologies``: List of ideological types.
- ``interaction_matrix``: Competitive interaction matrix (B), N x N.
- ``suppression_matrix``: Propensity of j to suppress i when in power (s_ij),
  N x N.
- ``environment``: Environment configuration.
- ``stochastic``: Stochastic event parameters.
- ``initial_shares``: x(0) on the simplex, shape (N,).
- ``initial_latent``: y(0), shape (N,).
- ``rho``: Replicator-mutator immigration rate (latent -> manifest).
- ``t_start``: Start year.
- ``t_end``: End year.
- ``dt``: Integration time step (years).
- ``record_interval``: How often to record state (in time steps).
- ``seed``: Random seed for reproducibility.

Ideology Parameters
-------------------

Each ideology entry defines:

- ``name``: Human-readable name.
- ``env_response``: How this ideology responds to environmental conditions
  (w_i), length M.
- ``latent_conversion``: Rate at which latent sympathy converts to manifest
  adoption (gamma_i).
- ``latent_generation``: Rate at which active adherents generate latent
  sympathizers (alpha_i).
- ``latent_decay``: Natural decay rate of latent sympathy (beta_i).
- ``martyrdom_coeff``: How efficiently suppression creates sympathy
  (delta_i).
- ``institutional_amp``: Amplification rate above capture threshold (mu_i).
- ``capture_threshold``: Share above which institutional capture kicks in
  (theta_i).
- ``governance_threshold``: Share above which ideology can suppress others
  (theta_i^gov).
- ``mutation_rate``: Rate of parameter drift per year (nu_i).

Environment
-----------

The ``environment`` block has:

- ``variable_names``: Names of the M environmental variables.
- ``initial``: e(0), shape (M,).
- ``exogenous_shocks``: List of (time, variable_index, delta), discrete shocks.
- ``autonomous_decay``: Decay rate toward baseline for each variable (g_k).
- ``policy_influence``: How ideologies change the environment (P), N x M.

Stochastic Events
-----------------

The ``stochastic`` block has:

- ``hybridization_rates``: Rate of hybridization between i and j (p_ij),
  shape (N, N).
- ``hybridization_thresholds``: Latent sympathy overlap threshold (H_ij),
  shape (N, N).
- ``charisma_rate``: Global rate of charismatic events per year
  (lambda_char).
- ``charisma_decay``: Decay time of charismatic effect in years (tau_char).
- ``charisma_amplitude_mean``: Mean of log-normal charisma amplitude.
- ``charisma_amplitude_std``: Std of log-normal charisma amplitude.
- ``mutation_step_size``: Std of Gaussian mutation perturbation
  (sigma_mut).
- ``hybridization_seed_share``: Initial share of new hybrid (epsilon_h).

See the API reference for the full dataclass specifications.
