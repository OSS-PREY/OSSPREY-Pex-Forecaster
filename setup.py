"""
@brief Setup the package. We'll automatically fetch the parquet files in the 
    setup process.
@author Arjun Ashok (arjun3.ashok@gmail.com)
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
    name="decalfc",
    version="1.1",
    description="Module for on the fly socio-technical network data generation, forecasting, and trajectory building.",
    license="Apache",
    long_description=long_description,
    author="Arjun Ashok",
    author_email="arjun3.ashok@gmail.com",
    url="https://github.com/arjashok/pex-forecaster",
    # package_dir={"": "decalfc"},
    packages=find_packages(),
    install_requires=reqs,
    scripts=[
        "forecast.sh",
        "net-gen.sh"
    ]
)

