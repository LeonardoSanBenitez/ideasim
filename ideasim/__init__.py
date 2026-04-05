"""Ideological Dynamics Simulator -- models competition between philosophical currents.

SPDX-License-Identifier: Apache-2.0
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("ideasim")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
