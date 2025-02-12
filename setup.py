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


# # ------------- Comprehensive Environment Setup ------------- #
# # external packages
# import gdown

# # built-in modules
# from pathlib import Path

# # DECAL modules
# from decalfc.utils import _load_params
# params_dict = _load_params()


# # ------------- Data Setup ------------- #
# _DATA_URLS = {
#     "apache": {
#         "tech": "https://drive.google.com/file/d/1TWfOUGlqfHXGY0H2fpxnw8U3NmN_RV7C/view?usp=sharing",
#         "social": "https://drive.google.com/file/d/11H0lRToXQq83Vmfhoj3d7yLIyTKyieGE/view?usp=drive_link"
#     },
#     "github": {
#         "tech": "https://drive.google.com/file/d/1Y-afVT8XVlvQELGbi-1JZS-a6gk8pQHH/view?usp=drive_link",
#         "social": "https://drive.google.com/file/d/1RlRAp0jRscgVoNRAVkzK3DBytkS10j0K/view?usp=drive_link"
#     },
#     "eclipse": {
#         "tech": "https://drive.google.com/file/d/1zF0MAXte7O1Pkiq-e561isCjVdUTk_YD/view?usp=drive_link",
#         "social": "https://drive.google.com/file/d/1LczzKH4jG8jKrRb7zmQ0DItCW77SYz5B/view?usp=drive_link"
#     }
# }
# _DATA_PATH = Path(params_dict["dataset-dir"])
# _DATA_PATH.mkdir(exist_ok=True)

# for ds, url_pkg in _DATA_URLS.items():
#     # download each of the sub-items
#     for dtype, url in url_pkg.items():
#         gdown.download(
#             url,
#             _DATA_PATH / f"{ds}_data" / f"{params_dict[f'{dtype}-type'][ds]}.parquet",
#             quiet=True
#         )

