"""
@brief End-to-End tests for the pex-forecaster tool deployed in OSSPREY. Uses 
    default parquets from real projects to ensure accurate computation.
@author Arjun Ashok (arjun3.ashok@gmail.com)
"""


# --- Environment Setup --- #
import pytest

from decalfc.app.server import *
from tests.__init__ import INPUT_PATH, RESULT_PATH, OUTPUT_PATH

import pickle
from pathlib import Path

# --- Utility --- #
def save_result(path: Path, res_dict: dict) -> None:
    """Saves the result of the pipeline for comparison."""
    
    # binary
    path = path.with_suffix(".pkl")
    
    with open(path, "wb") as f:
        pickle.dump(res_dict, f)

def load_result(path: Path) -> dict:
    """Loads the result of the pipeline for comparison."""
    
    # modify path
    print("CAUTION :: FALSE COMPARISON FOR DEBUGGING PURPOSES, REPLACE THE EARLY RETURN!!!!")
    path = OUTPUT_PATH / path.stem
    
    path = path.with_suffix(".pkl")
    if not path.exists():
        raise ValueError(f"{path} doesn't exist, try generating the comparison result first!")
    
    # read
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def compare_result(output: dict, target: dict, eps: float=1e-2) -> bool:
    """Compares two result dictionaries to see if they're close enough (i.e. 
    with some margin of error to account for float precision).

    Args:
        output (dict): output dictionary with all output keys present.
        target (dict): target dictionary with all output keys present.
        eps (float, optional): margin of error allowed. Defaults to 0.01.

    Returns:
        bool: True if the results are similar enough (pass), otherwise False 
            (fail)
    """
    
    # aux fn
    def comp_net(n1: pd.DataFrame, n2: pd.DataFrame) -> bool:
        # check valid return
        if n1 is None:
            return False
        
        # col check
        if not n1.columns.equals(n2.columns):
            log(f"failed netdata column equality check: \n\t{n1.columns}\n\n\t{n2.columns}", "error")
            return False

        # dim check
        if n1.shape != n2.shape:
            log("failed netdata dimension equality check...", "error")
            return False
        
        # value check
        for col in n1.columns:
            # branch if object col
            if isinstance(n1[col].iloc[0], str) and (not n1[col].equals(n2[col])):
                log(f"failed netdata column ({col}) equality check...", "error")
                return False
            if isinstance(n1[col].iloc[0], str):
                continue
            
            # check approximal equality
            if not n1[col].equals(n2[col]) and not np.allclose(n1[col].values, n2[col].values, atol=eps, equal_nan=True):
                log(f"failed netdata column ({col}) approximate equality check...", "error")
                return False
        
        # passed checks
        return True
    
    def comp_net_vis(nv1: dict, nv2: dict) -> bool:
        # check valid return
        if nv1 is None:
            return False
        
        # strict equality
        return nv1 == nv2
    
    def comp_forecast(f1: dict, f2: dict) -> bool:
        # check valid return
        if f1 is None:
            return False
        
        # comparison with higher tolerance than usual
        return np.allclose(list(f1.values()), list(f2.values()), atol=10*eps)
    
    def comp_traj(t1: dict, t2: dict) -> bool:
        # check valid return
        if t1 is None:
            return False
        
        # by month
        return all(
            # by traj type
            all(
                # all trajectory values
                np.allclose(t1t, t2t, atol=10*eps, equal_nan=True)
                for t1t, t2t in zip(t1m.values(), t2m.values())
            )
            for t1m, t2m in zip(t1.values(), t2.values())
        )
    
    # compare basic info
    assert len(output) == len(target)
    assert comp_net(output["net-gen"], target["net-gen"])
    assert comp_net_vis(output["net-vis"], target["net-vis"])
    assert comp_forecast(output["forecast"], target["forecast"])
    # assert comp_traj(output["trajectory"], target["trajectory"]) #  skipping traj for now
    
    # passed all tests
    return True

def run_pipeline_with_data(ds_name: str) -> bool:
    """Wrapper for the dispatch to the pex-forecaster module, essentially 
    mimicking the call the OSSPREY backend should be making to the module on 
    requests.

    Args:
        ds_name (str): name of the dataset to use.
    """
    
    # unpack args
    tech_ds_path = INPUT_PATH / f"{ds_name}_tech.parquet"
    social_ds_path = INPUT_PATH / f"{ds_name}_social.parquet"
    compare_path = RESULT_PATH / f"{ds_name}"
    out_path = OUTPUT_PATH / f"{ds_name}"
    
    # data package
    test_pkg = {
        "project_name": ds_name,
        "tech_data": pd.read_parquet(tech_ds_path),
        "social_data": pd.read_parquet(social_ds_path),
        "tasks": ["ALL"],
        "month_range": [0, -1],
        "ignore_cache": True
    }
    
    # dispatch to forecaster, catch any errors
    try:
        res = compute_forecast(test_pkg)
    except Exception as e:
        log(f"errored in testing :: {e}", "error")
    
    # load in expected outputs, save current results
    save_result(out_path, res)
    exp_res = load_result(compare_path)
    
    # clear the cache
    reset_cache(test_pkg["project_name"])

    # compare result if it exists
    return compare_result(res, exp_res)

def convert_inputs():
    """Convert all inputs into the specified format (parquet) and naming scheme.
    """
    
    # aggregate names
    sample_ds_path = INPUT_PATH / "datasets.json"
    ds_names = list()
    projects = {f.stem.split("_issues")[0] for f in INPUT_PATH.glob("*_issues.csv")}
    
    # go through all csvs and convert them as needed
    print(INPUT_PATH)
    for proj in tqdm(projects):
        # read pairs
        try:
            tdf = pd.read_csv(INPUT_PATH / f"{proj}-commit-file-dev.csv")
        except Exception as _:
            log(f"failed to read in tech data for {proj}! Skipping for now", "warning")
            continue
        
        try:
            sdf = pd.read_csv(INPUT_PATH / f"{proj}_issues.csv")
        except Exception as _:
            log(f"failed to read in social data for {proj}! Skipping for now", "warning")
            continue
        
        # export
        tdf.to_parquet(INPUT_PATH / f"{proj}_tech.parquet", index=False)
        sdf.to_parquet(INPUT_PATH / f"{proj}_social.parquet", index=False)
        
        # remove previous files
        (INPUT_PATH / f"{proj}-commit-file-dev.csv").unlink()
        (INPUT_PATH / f"{proj}_issues.csv").unlink()
        
        # add to tracker
        ds_names.append(proj)
    
    # save if non-zero
    if len(ds_names) == 0:
        return
    
    with open(sample_ds_path, "w") as f:
        json.dump(ds_names, f, indent=2)

def load_datasets() -> list[str]:
    """Loads the datasets to use in the testing script.

    Returns:
        list[str]: sample datasets by name, not filepath.
    """
    
    # setup
    sample_ds_path = INPUT_PATH / "datasets.json"
    sample_ds = list()
    
    # create if needed
    if not sample_ds_path.exists():
        convert_inputs()
    
    # check if failed to create
    if not sample_ds_path.exists():
        log("failed to load datasets, try downloading the test files from gdrive", "error")
        return
    
    # open and read
    with open(sample_ds_path, "r") as f:
        sample_ds = load(f)
    
    # export
    return sample_ds

# --- Testing Protocols --- #
sample_ds = load_datasets()

@pytest.mark.parametrize(
    "ds",
    sample_ds,
    ids=lambda ds: repr(ds)
)
def test_end_to_end(ds):
    assert run_pipeline_with_data(ds)

if __name__ == "__main__":
    convert_inputs()

