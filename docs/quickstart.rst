Quick Start
===========

Installation
------------

From PyPI::

    pip install ideasim

From source::

    git clone https://github.com/LeonardoSanBenitez/ideasim.git
    cd ideasim
    pip install -e ".[dev]"

Running Your First Simulation
-----------------------------

Using the CLI::

    ideasim simulate -o results.json

Using the Python API:

.. code-block:: python

    from ideasim.defaults import make_default_config
    from ideasim.engine import simulate

    config = make_default_config()
    config.seed = 42
    result = simulate(config)

    for name, share in zip(result.ideology_names, result.history[-1].shares):
        print(f"{name}: {share:.2f}")
    print(f"Events triggered: {len(result.events)}")

Custom Configurations
---------------------

Export the default configuration to customize it::

    ideasim default-config -o my_config.json

Edit the JSON file, then run::

    ideasim simulate my_config.json -o results.json
