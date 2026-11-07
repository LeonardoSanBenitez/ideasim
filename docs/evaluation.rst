Evaluation
==========

ideasim includes a validation framework for comparing simulation outputs
against observed data from the `ParlGov <https://www.parlgov.org/>`_ database.

ParlGov Integration
-------------------

The evaluation module fetches party-level electoral data from ParlGov and
maps parties to ideological families for comparison with simulated trajectories.

To run the evaluation (requires network access for initial data fetch)::

    python -m pytest tests/test_parlgov.py -v --run-slow

The data is cached locally in the ``data/`` directory after the first fetch.

Data Fetching
-------------

ParlGov data is not bundled with the package. To fetch it manually:

.. code-block:: python

    from ideasim.evaluation.parlgov import fetch_parlgov_data

    data = fetch_parlgov_data(cache_dir="data/parlgov")

The fetched data includes party positions, election results, and cabinet
compositions across European democracies.

Fitted Initial-State Epochs
--------------------------

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
