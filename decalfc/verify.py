"""
    @brief Defines a setup script for ensuring the rawdata, file structure, and 
        anything else is there for all trials to work.
    @author Arjun Ashok (arjun3.ashok@gmail.com)
    @creation-date February 2025
"""

# --- Environment Setup --- #
## packages
import gdown

## DECAL modules
from decalfc.utils import *

## constants
__IMPL_INCUBATORS = [
    "apache",
    "eclipse",
    "github",
    # "osgeo"
]
_DATA_URLS = {
    "apache": {
        "tech": {
            "commits": "https://drive.google.com/file/d/1TWfOUGlqfHXGY0H2fpxnw8U3NmN_RV7C/view?usp=sharing"
        },
        "social": {
            "emails": "https://drive.google.com/file/d/11H0lRToXQq83Vmfhoj3d7yLIyTKyieGE/view?usp=drive_link"
        }
    },
    "github": {
        "tech": {
            "commits": "https://drive.google.com/file/d/1Y-afVT8XVlvQELGbi-1JZS-a6gk8pQHH/view?usp=drive_link"
        },
        "social": {
            "issues": "https://drive.google.com/file/d/1RlRAp0jRscgVoNRAVkzK3DBytkS10j0K/view?usp=drive_link"
        }
    },
    "eclipse": {
        "tech": {
            "commits": "https://drive.google.com/file/d/1zF0MAXte7O1Pkiq-e561isCjVdUTk_YD/view?usp=drive_link"
        },
        "social": {
            "issues": "https://drive.google.com/file/d/1LczzKH4jG8jKrRb7zmQ0DItCW77SYz5B/view?usp=drive_link"
        }
    },
    "osgeo": {
        "tech": {
            "commits": "https://drive.google.com/file/d/1LTex3X2_sgAy2Dy79wYvVURbi3sJzYyA/view?usp=sharing"
        },
        "social": {
            "emails": "",
            "issues": ""
        }
    }
}


# --- Auxiliary Utility --- #
def check_raw_data(verbosity: int=2) -> int:
    """Checks the raw data from the certified incubators are installed, 
    otherwise performs the download as required.

    Args:
        verbosity (int, optional): how verbose the prints should be:
            - 0 -- no logging should be done
            - 1 -- summary w/ no progress bar
            - 2 -- log each download/check, progress bar, summary
            Defaults to 2.

    Returns:
        int: number of downloads required; treating each file as separate.
    """
    
    # crawler setup, check directory exist
    _DATA_PATH = Path(params_dict["dataset-dir"])
    _DATA_PATH.mkdir(exist_ok=True)
    ndownloads = 0

    # crawl the data paths
    for ds, dtype_urls in _DATA_URLS.items():
        # ensure the directory exists
        incubator_data_path = _DATA_PATH / f"{ds}_data"
        incubator_data_path.mkdir(exist_ok=True)
        
        # download each of the sub-items if they don't exist
        for dtype, urls_pkg in dtype_urls.items():
            # check each sub-list of the given dtype and combine if needed
            for subtype, url in urls_pkg.items():
                
                # unpack file path
                dl_path = incubator_data_path / f"{params_dict[f'{dtype}-type'][ds]}.parquet"
                
                # check if we need to download
                if dl_path.exists():
                    if verbosity:
                        log(
                            f"Skipping the <{dtype}> data for <{ds}>, exists @ {dl_path}",
                            "log", check_verbosity=(verbosity > 1)
                        )
                    continue
                
                # download required
                log(
                    f"Downloading the <{dtype}> data for <{ds}>, saving to {dl_path}",
                    "log", check_verbosity=(verbosity > 0)
                )
                ndownloads += 1
                gdown.download(
                    url=str(url),
                    output=str(dl_path),
                    quiet=(verbosity < 2),
                    fuzzy=True
                )
            
    # summary
    log(log_type="summary")
    log(f"Downloads Required: {ndownloads}\n")

def check_directory_structure(verbosity: int=2) -> int:
    """Checks all expected folders are present.

    Args:
        verbosity (int, optional): how verbose the prints should be:
            - 0 -- no logging should be done
            - 1 -- summary w/ no progress bar
            - 2 -- log each check, progress bar, summary
            Defaults to 2.

    Returns:
        int: number of creations required; treating each folder as separate.
    """
    
    # required folders
    __FOLDERS = {
        "dataset-dir": "./data/",
        "network-dir": "./network-data/",
        "network-visualization-dir": "./net-vis/",
        "ref-dir": "./ref/",
        "weights-dir": "./model-weights/",
        "reports-dir": "./model-reports/",
        "forecast-dir": "./forecasts/",
        "trajectory-dir": "./trajectories/",
        "visuals-dir": "visuals"
    }
    
    # crawler setup
    BASE_PATH = Path().cwd()
    ncreations = 0
    
    # create all necessary folders
    for folder_name, folder in __FOLDERS.items():
        # join paths
        dir_path = BASE_PATH / folder
        
        # ensure exists
        if not dir_path.exists():
            ncreations += 1
            dir_path.mkdir(parents=False)
            log(
                f"Creating the <{folder_name.replace('-', ' ')}> @ {dir_path}",
                "log", check_verbosity=(verbosity > 1)
            )
    
    # summary
    log(log_type="summary")
    log(f"Number of Folders Created: {ncreations}\n")


# --- Setup Script --- #
def verify_pkg() -> None:
    """Checks the package is successfully installed with all necessary data and 
    folder structure present.
    """
    
    # checks
    check_directory_structure()
    check_raw_data()

if __name__ == "__main__":
    verify_pkg()

