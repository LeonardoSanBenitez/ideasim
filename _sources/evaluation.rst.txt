Evaluation
==========

ideasim includes a validation framework for comparing simulation outputs
against observed data from the `ParlGov <https://www.parlgov.org/>`_ database.

ParlGov Integration
-------------------

The evaluation module fetches party-level electoral data from ParlGov and
maps parties to ideological families for comparison with simulated trajectories.

To download the data and build the observed dataset from the command line::

    ideasim ingest-parlgov -o parlgov_dataset.json

The first run needs network access. The two CSV files are cached in
``data/parlgov_cache/`` (``--cache-dir`` changes it) and reused afterwards.

The tests in ``tests/test_parlgov.py`` run offline on small hand-written files.
The few that read the real data use ``data/parlgov_cache/`` in the repository
when it exists and are skipped otherwise::

    python -m pytest tests/test_parlgov.py -v

Data Fetching
-------------

ParlGov data is not bundled with the package. To fetch it manually:

.. docs-test: slow

.. code-block:: python

    from pathlib import Path

    from ideasim.evaluation.parlgov import download_and_build

    dataset = download_and_build(Path("data/parlgov_cache"))

    print(f"Number of observations: {len(dataset.observations)}")

This downloads the election results and the party table, which also carries
the parties' positions. The dataset builder uses only each party's family,
mapped to an ideology.

Fitted Initial-State Epochs
---------------------------

When ``fit_parameters(..., align_t_start=True)`` advances the simulation start
to ten years before the first training observation, the returned configuration
preserves that starting year. Its initial adoption and latent-sympathy values
are states at this aligned epoch. The effective year is also recorded as
``effective_t_start`` in the fitting metadata.

For example, a configuration starting in 1800 with its first observation in
1900 is fitted and returned with a start of 1890. Earlier environmental shocks
are skipped. This defines a model starting in 1890; it does not reconstruct
the original trajectory from 1800.

``fit_dt`` may use a different integration step during optimization. The
returned configuration and reported training metrics retain the original
evaluation step, so their loss can differ from the optimization loss when
``fit_dt`` is supplied.

Fitting a Subset of Parameters
------------------------------

``fit_parameters(..., free=[...])`` optimizes only the named parameters and
keeps every other one at its value in the base configuration. The names are
those of ``build_parameter_spec`` (for example ``rho`` or
``liberalism.latent_decay``); an empty, repeated or unknown name raises
``ValueError``. Without ``free``, every parameter in the spec is fitted, as
before.

Parameter Recovery
------------------

``ideasim.evaluation.recovery`` checks whether the fitter can find parameters
that are known: draw a configuration, simulate data from it, fit, and compare.

.. code-block:: python

    import numpy as np

    from ideasim.defaults import make_default_config
    from ideasim.evaluation.fitting import (
        build_parameter_spec,
        fit_parameters,
        select_parameters,
    )
    from ideasim.evaluation.recovery import (
        compare_to_truth,
        draw_truth,
        synthetic_dataset,
    )

    base = make_default_config()
    base.t_start, base.dt = 1890.0, 0.5
    spec = select_parameters(build_parameter_spec(base), ["rho"])
    truth = draw_truth(base, spec, np.random.default_rng(0), margin=0.1)
    data = synthetic_dataset(truth, [1900.0, 1910.0, 1920.0], noise_sd=0.0)
    fit = fit_parameters(
        base, data, free=["rho"], max_iterations=10, population_size=5, seed=3
    )
    for row in compare_to_truth(fit, truth):
        print(row.name, row.true, row.fitted, row.scaled_error, row.at_bound)

``synthetic_dataset`` switches stochastic events and mutation off, as the
fitter does, so noise-free data is data the fitter can reproduce exactly;
``noise_sd`` adds Gaussian noise clipped to [0, 1]. ``scaled_error`` is the
error divided by the width of the parameter's bounds, so errors of different
parameters are comparable. ``draw_truth(..., margin=m)`` keeps drawn values a
fraction ``m`` of the width away from each bound, so a fit that ends at a bound
is always wrong.
