# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages

extras_require = {
    "visualization": ["graphviz"],
    "tests": ["pytest", "pytest-cov"],
}
extras_require["complete"] = sorted(set(sum(extras_require.values(), [])))

setup(
    name="dagger",
    version="0.1.0",
    install_requires=["dill", "dask"],
    packages=find_packages(),
    extras_require=extras_require,
)
