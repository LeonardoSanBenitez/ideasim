Configuration
=============

ideasim is configured via JSON files. Export the default configuration to see
all available parameters::

    ideasim default-config -o config.json

Top-Level Structure
-------------------

The configuration file has the following top-level fields:

- **ideologies**: List of ideology definitions, each with fitness parameters,
  latent sympathy coefficients, and mutation matrices.
- **environment**: Environmental conditions over time (GDP growth, inequality,
  media freedom, social media penetration, conflict intensity).
- **simulation**: Simulation parameters (time span, step size, random seed,
  event rates).

Ideology Parameters
-------------------

Each ideology entry defines:

- ``name``: Human-readable name.
- ``env_response``: Weight vector controlling how the ideology responds to
  environmental conditions.
- ``latent_conversion``: Rate of latent-to-manifest conversion.
- ``latent_generation``: Rate at which adherents generate latent sympathizers.
- ``latent_decay``: Natural decay of latent sympathy.
- ``martyrdom_coeff``: How efficiently suppression creates sympathy.

See the API reference for the full ``Ideology`` dataclass specification.
