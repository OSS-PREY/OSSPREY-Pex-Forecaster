"""
@brief API endpoints are routed through here, handling interface between 
    frontend and backend of the PEX tool.
@author Arjun Ashok (arjun3.ashok@gmail.com)
"""

# ------------- Environment Setup ------------- #
# external packages -- none for now
import pandas as pd
from flask import Flask, request, jsonify

# built-in modules
import sys
from pathlib import Path

# DECAL modules
import decalforecaster.utils as util
from decalforecaster.pipeline import *

# setup the App for communication
app = Flask(__name__)


# ------------- API Endpoints ------------- #
# local vars for storage
proj_name = None
tdata = None
sdata = None

# receiver route
@app.route("/grab_data/", methods=["POST"])
def pull_raw_data():
    """Route for grabbing the raw data as a JSON dictionary.
    
    The information expected is as follows:
    {
        (project_name): str
        (commits_data): dict, should be easily convertable into a pandas 
            dataframe, i.e. {column: data}.
        (issues_data): dict, should be easily convertable into a pandas 
            dataframe, i.e. {column: data}.
        (tasks): list[str], match the tasks implemented
    }
    
    Implemented Tasks:
        - *(pp-paths): Pre-processing via File Path Cleansing
        - *(pp-names): Pre-processing via Sender Name Cleansing
        - *(pp-months): Pre-processing via Month Imputation
        - *(pp-msg-id): Pre-processing via Message ID Imputation
        - *(pp-is-coding): Pre-processing via Source File Identification
        - *(pp-replies): Pre-processing via Reply Inferencing
        - *(pp-bots): Pre-processing via Bot Inference
        - *(pp-de-alias): Pre-processing via De-Aliasing
        - (net-gen) Network Generation
        - (traj) Trajectories
        - (forecast) Forecast predictions
        
        * All pre-processing tasks are done by default on any received data

    Request Args:
        data (dict[str, Iterable[int | float | str]]): pandas dataframe in 
            dictionary representation
    
    Verification Messages:
        (200) Data is successfully received
        (400) Failed to provide data with the request; invalid format for data
        (500) Unidentified error
    """
    
    # setup the input validation
    needed_keys = ["project_name", "commits_data", "issues_data", "tasks"]
    val_types = [str, dict, dict, list]
    
    try:
        # grab the JSON request
        data = request.get_json()
        
        # error handling if missing
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # error handling if incorrectly structured
        if not isinstance(data, dict):
            return jsonify({"error": "Data not formatted as a dictionary"}), 400
        
        if not all(key in data for key in needed_keys):
            return jsonify({
                "error": f"Missing some expected information; expected {needed_keys}, got {list(data.keys())}"
            }), 400
        
        if not all(isinstance(val, val_type) for val, val_type in zip(data.values(), val_types)):
            actual_types = [str(type(v)) for v in data.values()]
            return jsonify({
                "error": f"Values of are an unexpected type; expected {val_types}, got {actual_types}"
            }), 400
            
        # error handling if incorrectly formatted
        try:
            # parse dataframes
            tdata = pd.DataFrame(data["commits_data"])
            sdata = pd.DataFrame(data["issues_data"])
        except ValueError as ve:
            return jsonify({"error": f"Data not parse-able by pandas: {str(ve)}"}), 400
        
        proj_name = data["project_name"]
        
        # store the data received into local var
        return jsonify({"message": "Data received successfully", "data": data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# exporter route: forecasts
@app.route("/forecaster/predictions", methods=["GET"])
def export_forecasts():
    """Route for exporting the predictions for a given project as JSON object.
    
    Verification Messages:
    """
    pass

# exporter route: trajectories
@app.route("/forecaster/trajectories", methods=["GET"])
def export_trajectories():
    """Route for exporting the predictions for a given project as JSON object.
    
    Verification Messages:
    """
    pass

# exporter route: networks
@app.route("/forecaster/networks", methods=["GET"])
def export_trajectories():
    """Route for grabbing the raw data as a JSON dictionary. Specified format 
    should be easily convertable into a pandas dataframe, i.e. {column: data}.
    
    Verification Messages:
    """
    pass

if __name__ == '__main__':
    app.run(debug=True)
