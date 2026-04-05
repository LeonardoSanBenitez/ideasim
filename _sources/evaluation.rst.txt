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
