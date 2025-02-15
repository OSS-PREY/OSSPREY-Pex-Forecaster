"""
    @brief Generates network edgelists and mapping
    @author Dr. Likang Yin (lykin@ucdavis.edu), Arjun Ashok 
			(arjun3.ashok@gmail.com)
    @creation-date (unknown, modified by Arjun @ later date)
"""

    #coding:utf-8
    # import re
    # import base64
    # import quopri

# ------------- Environment Setup ------------- #
# external packages
import networkx as nx
import pandas as pd
from tqdm import tqdm

# built-in modules
import os
import json
import sys
from collections import Counter
from typing import Any
from pathlib import Path

# DECAL modules
import decalfc.utils as util
from decalfc.utils import PARQUET_ENGINE, CSV_ENGINE


"""
# convert to UTF-8 coding
def text_encoding(encoded_words):
    # return ascii code
    if "=" not in encoded_words:
        return encoded_words
    try:
        encoded_word_regex = r"=\?{1}(.+)\?{1}([B|Q])\?{1}(.+)\?{1}="
        charset, encoding, encoded_text = re.match(encoded_word_regex, encoded_words).groups()
        if encoding == "B":
            byte_string = base64.b64decode(encoded_text)
        elif encoding == "Q":
            byte_string = quopri.decodestring(encoded_text)
        return byte_string.decode(charset)
    except:
        return encoded_words
"""


# ---------------- processing utility ---------------- #
def process_tech_nets(author_field: str, t_source: Path, t_output: Path) -> None:
    projects = os.listdir(t_source)
    util.check_dir(t_output)

    for project in tqdm(projects):
        technical_net = {}
        project_name, period = project.replace(".parquet", "").split("__")
        df = pd.read_parquet(t_source / project, engine=PARQUET_ENGINE)
        df.query("is_bot == False and is_coding == True", inplace=True)
        df = df[df[author_field].notna()]

        for index, row in df.iterrows():
            fils_source = row["file_name"]
            # file extension = "." + fils_source.split("/")[-1].split(".")[-1].split(" ")[0]
            file_name = fils_source.split("/")[-1]
            author_name = row[author_field]
            
            if file_name not in technical_net:
                technical_net[file_name] = {}
            if author_name not in technical_net[file_name]:
                technical_net[file_name][author_name] = {}
                technical_net[file_name][author_name]["weight"] = 0
            technical_net[file_name][author_name]["weight"] += 1

        #save as directed graph
        g = nx.DiGraph(technical_net)
        # add disconnected nodes
        g.add_nodes_from(technical_net.keys())
        nx.write_edgelist(g, t_output / "{}__{}.edgelist".format(project_name, str(period)), delimiter="##", data=["weight"])


# ---------------- processing social nets ---------------------- #
def process_social_nets(author_field: str, s_source: Path, s_output: Path, mapping_path: Path) -> None:
    # directory handling
    projects = os.listdir(s_source)
    util.check_dir(s_output)

    # get sender-receiver timestamp
    sender_dic = {"project": [], "message_id":[], "sender":[], "receiver":[], \
                  "timestamp":[], "broadcast":[]}

    # process each project
    for project in tqdm(projects):
        # setup project social network
        social_net = {}
        emailID_to_author = {}
        project_name, period = project.replace(".parquet", "").split("__")

        # load project data
        df = pd.read_parquet(s_source / project, engine=PARQUET_ENGINE)
        df.query("is_bot == False", inplace=True)

        df = df[df[author_field].notna()]
        
        # generate dict lookup
        for index, row in df.iterrows():
            message_id = str(row["message_id"]).strip()
            sender_name = row[author_field]
            timestamp = row["date"]
            emailID_to_author[message_id] = (sender_name, timestamp)

        # raise KeyError
        for index, row in df.iterrows():
            message_id = str(row["message_id"]).strip()
            references = str(row["in_reply_to"]).strip()
            sender_name = row[author_field]
            timestamp = row["date"]
            

            # ignores if this email does not reply to previous emails
            if pd.isna(references) or references == "None":
                continue

            # deal with the issue that a line breaker exists in message_id:
            # e.g., <4\n829AB62.6000302@apache.org>
            references = [r.strip() for r in references.replace("\n", " ").replace("\t", " ").split(" ") if r.strip()]

            new_refs = set()
            for i in range(len(references)-1):
                if "<" in references[i] and ">" not in references[i] and "<" not in references[i+1] and ">" in references[i+1]:
                    new_refs.add(references[i] + references[i+1])
            for r in references:
                if "<" in r and ">" in r:
                    new_refs.add(r)

            references = new_refs

            for reference_id in references:
                if reference_id not in emailID_to_author:
                    continue
                prev_author, prev_timestamp = emailID_to_author[reference_id]
                # if it's the same person, continue
                if prev_author == sender_name:
                    continue

                # add to the sender-receiver mapping
                # sender drops an email to previous author
                sender_dic["project"].append(project_name)
                sender_dic["message_id"].append(message_id)
                sender_dic["sender"].append(sender_name)
                sender_dic["receiver"].append(prev_author)
                sender_dic["timestamp"].append(timestamp)
                sender_dic["broadcast"].append(0)    

                # since the previous author sent information to 
                # this receiver since the receivier replied.
                sender_dic["project"].append(project_name)
                sender_dic["message_id"].append(reference_id)
                sender_dic["sender"].append(prev_author)
                sender_dic["receiver"].append(sender_name)
                sender_dic["timestamp"].append(prev_timestamp)
                sender_dic["broadcast"].append(1)

                if sender_name not in social_net:
                    social_net[sender_name] = {}
                    
                if prev_author not in social_net:
                    social_net[prev_author] = {}

                # if node B replies node A, it means B sends signal to A
                if prev_author not in social_net[sender_name]:
                    social_net[sender_name][prev_author] = {}
                    social_net[sender_name][prev_author]["weight"] = 0
                social_net[sender_name][prev_author]["weight"] += 1

                # if node B replies node A, it means A also sent signal to B
                if sender_name not in social_net[prev_author]:
                    social_net[prev_author][sender_name] = {}
                    social_net[prev_author][sender_name]["weight"] = 0
                social_net[prev_author][sender_name]["weight"] += 1

        # save as directed graph
        g = nx.DiGraph(social_net)
        
        ## add disconnected nodes
        g.add_nodes_from(social_net.keys())
        nx.write_edgelist(g, s_output / "{}__{}.edgelist".format(
            project_name, 
            str(period)
        ), delimiter="##", data=["weight"])

    # export
    df = pd.DataFrame.from_dict(sender_dic)
    df.to_csv(mapping_path, index=False)


# Script
def create_networks(args_dict: dict[str, Any]):
    """
        Wrapper for creating networks from the segmented data.
    """

    # setup
    print("\n<Creating Socio-Technical Networks>")
    params_dict = util.load_params()
    author_field = "dealised_author_full_name"

    # execute input
    incubator = args_dict["incubator"]
    social_type = params_dict["social-type"][incubator]
    tech_type = params_dict["tech-type"][incubator]
    dataset_dir = Path(params_dict["dataset-dir"])
    network_dir = Path(params_dict["network-dir"])

    t_dir = dataset_dir / f"{incubator}_data" / "monthly_data" / f"{tech_type}/"
    s_dir = dataset_dir / f"{incubator}_data" / "monthly_data" / f"{social_type}/"
    t_output_dir = network_dir / f"{incubator}_{tech_type}/"
    s_output_dir = network_dir / f"{incubator}_{social_type}/"
    mapping_out_dir = network_dir / "mappings/"
    mapping_out_source = str(mapping_out_dir) + f"/{incubator}-mapping.csv"

    # ensure clean save
    util.check_dir(t_output_dir)
    util.check_dir(s_output_dir)
    util.clear_dir(t_output_dir, skip_input=True)
    util.clear_dir(s_output_dir, skip_input=True)
    util.check_dir(mapping_out_dir)

    # process
    process_tech_nets(author_field, t_dir, t_output_dir)
    process_social_nets(author_field, s_dir, s_output_dir, mapping_path=mapping_out_source)


if __name__ == "__main__":
    # load input
    args_dict = util.parse_input(sys.argv)
    create_networks(args_dict)
    
