"""
@brief Setup the package. We'll automatically fetch the parquet files in the 
    setup process.
"""

# ------------- Bare Environment Setup ------------- #
# built-in modules
from setuptools import setup, find_packages


# ------------- Dependencies Setup ------------- #
with open("README.md", "r") as f:
    long_description = f.read()
with open("requirements.txt", "r") as f:
    reqs = f.read().splitlines()

setup(
    name="dfc",
    version="1.1",
    description="Module for on the fly socio-technical network data generation, forecasting, and trajectory building.",
    license="Apache",
    long_description=long_description,
    author="na",
    author_email="na",
    url="na",
    # package_dir={"": "dfc"},
    packages=find_packages(),
    install_requires=reqs,
    scripts=[
        "forecast.sh",
        "net-gen.sh"
    ]
)

