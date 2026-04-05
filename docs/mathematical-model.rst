Mathematical Model
==================

ideasim implements a Piecewise Deterministic Markov Process (PDMP) for
ideological competition. The continuous dynamics follow a replicator-mutator
equation with latent-driven immigration.

Continuous Dynamics
-------------------

The manifest adoption share :math:`x_i` evolves according to:

.. math::

    \frac{dx_i}{dt} = x_i (f_i - \bar{f}) + \rho (y_i - x_i Y)

where:

- :math:`f_i` is the fitness of ideology :math:`i` (environmental response +
  competitive interaction + institutional capture + latent reservoir).
- :math:`\bar{f} = \sum_j x_j f_j` is the mean population fitness.
- :math:`\rho` controls the latent-to-manifest coupling strength.
- :math:`y_i` is the latent sympathy for ideology :math:`i`.
- :math:`Y = \sum_j y_j` is the total latent sympathy.

Latent Sympathy
---------------

Latent sympathy evolves via:

.. math::

    \frac{dy_i}{dt} = \alpha_i x_i - \beta_i y_i + \sigma_i (1 - x_i) + \delta_i S_i

where:

- :math:`\alpha_i` is the generation rate from active adherents.
- :math:`\beta_i` is the natural decay rate.
- :math:`\sigma_i` is background sympathy generation.
- :math:`\delta_i S_i` is the martyrdom effect (suppression creates sympathy).

Stochastic Events
-----------------

Overlaying the continuous dynamics, stochastic events occur as a Poisson
process:

- **Hybridization**: Two parent ideologies blend to create a new hybrid.
- **Charismatic bursts**: Sudden surges in adoption of a particular ideology.
- **Parameter mutation**: Gradual drift in ideology parameters over time.

For the complete formal treatment, see the ``paper/`` directory in the
repository.
