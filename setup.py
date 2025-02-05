"""
@brief Setup the package. We'll automatically fetch the parquet files in the 
    setup process.
@author Arjun Ashok (arjun3.ashok@gmail.com)
"""

# ------------- Environment Setup ------------- #
# external packages
import gdown

# built-in modules
from pathlib import Path
from setuptools import setup, find_packages


# ------------- Data Setup ------------- #

url = 'https://drive.google.com/uc?id=0B9P1L--7Wd2vNm9zMTJWOGxobkU'
output = '20150428_collected_images.tgz'
gdown.download(url, output, quiet=False)
