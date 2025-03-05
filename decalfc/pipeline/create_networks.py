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
from decalfc.utils import *


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
    check_dir(t_output)

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

        # save as directed graph
        g = nx.DiGraph(technical_net)
        
        # add disconnected nodes
        g.add_nodes_from(technical_net.keys())
        nx.write_edgelist(g, t_output / "{}__{}.edgelist".format(project_name, str(period)), delimiter="##", data=["weight"])


# ---------------- processing social nets ---------------------- #
def process_social_nets(author_field: str, s_source: Path, s_output: Path, mapping_path: Path) -> None:
    # auxiliary utility
    def gen_email_id_lookup(df: pd.DataFrame, author_field: str) -> dict[str, tuple[str, str]]:
        """Generates a lookup of the email ID to author.

        Args:
            df (pd.DataFrame): dataframe containing the email data
            author_field (str): name of the author field in the dataframe

        Returns:
            dict[str, tuple[str, str]]: email ID to author lookup
        """
        
        # store lookup
        email_to_author = {}

        # generate the lookup
        for index, row in df.iterrows():
            message_id = str(row["message_id"]).strip()
            sender_name = row[author_field]
            timestamp = row["date"]
            email_to_author[message_id] = (sender_name, timestamp)
        
        # export
        return email_to_author
    
    def update_sr_mapping(
        sender_dic: dict[str, list], project_name: str, message_id: str,
        sender_name: str, prev_author: list[str], timestamp: str,
        prev_timestamp: str
    ) -> None:
        """Updates the sender-receiver mapping with a single 
        interaction.

        Args:
            sender_dic (dict[str, list]): sender-receiver mapping.
            project_name (str): project name.
            message_id (str): unique communication id for the 
                current communication.
            sender_name (str): sender of the email.
            prev_author (list[str]): single previous email author, 
                i.e. whoever sent a previous email.
            timestamp (str): timestamp of the current email.
            prev_timestamp (str): timestamp of the email being 
                replied to.
        
        Returns:
            None: all actions happen inplace
        """
        
        # add to the sender-receiver mapping sender drops an email to 
        # previous author
        sender_dic["project"].append(project_name)
        sender_dic["message_id"].append(message_id)
        sender_dic["sender"].append(sender_name)
        sender_dic["receiver"].append(prev_author)
        sender_dic["timestamp"].append(timestamp)
        sender_dic["broadcast"].append(0)    

        # since the previous author sent information to this receiver 
        # since the receivier replied
        sender_dic["project"].append(project_name)
        sender_dic["message_id"].append(reference_id)
        sender_dic["sender"].append(prev_author)
        sender_dic["receiver"].append(sender_name)
        sender_dic["timestamp"].append(prev_timestamp)
        sender_dic["broadcast"].append(1)
    
    def track_social_signal(social_net: dict[str, dict[str, dict[str, int]]], sender_name: str, prev_author: str) -> None:
        """Tracks the bi-directional interaction between two social collaborators.

        Args:
            social_net (dict[str, dict[str, dict[str, int]]]): social network as a dictionary lookup.
            sender_name (str): unique sender name
            prev_author (str): unique previous author name, i.e. person replying to the sender
        """
        
        # ensure the nodes are in the network
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
    
    # directory handling
    projects = os.listdir(s_source)
    check_dir(s_output)

    # get sender-receiver timestamp
    sender_dic = {
        "project": [], "message_id": [], "sender": [], "receiver": [], 
        "timestamp": [], "broadcast": []
    }

    # process each project
    for project in tqdm(projects):
        # setup project social network
        social_net = {}
        emailID_to_author = {}
        project_name, period = project.replace(".parquet", "").split("__")

        # load project data & ensure only valid communications are considered
        df = pd.read_parquet(s_source / project, engine=PARQUET_ENGINE)
        df.query("is_bot == False", inplace=True)
        df = df[df[author_field].notna()]
        
        # generate a lookup of the email ID to author
        emailID_to_author = gen_email_id_lookup(df, author_field)

        # for each communication in the social data, let's track the 
        # sender-receiver relationship in a graph
        for index, row in df.iterrows():
            # unpack useful information
            message_id = str(row["message_id"]).strip()
            references = str(row["in_reply_to"]).strip()
            sender_name = row[author_field]
            timestamp = row["date"]
            
            # regardless of reply information, let's track that there exists a 
            # node in this month
            social_net[sender_name]
            
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

            print(references, new_refs)
            exit()
            references = new_refs

            # for each previous replier that this current communication refers 
            # to, we'll track the social activity
            for reference_id in references:
                # if we can't identify this communicator, we skip them
                if reference_id not in emailID_to_author:
                    continue
                
                # unpack the previous author's information
                prev_author, prev_timestamp = emailID_to_author[reference_id]
                
                # if replying to themselves, continue
                if prev_author == sender_name:
                    continue
                
                # update the social network
                update_sr_mapping(
                    sender_dic, project_name, message_id, sender_name, 
                    prev_author, timestamp, prev_timestamp
                )
                track_social_signal(social_net, sender_name, prev_author)

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
    check_dir(t_output_dir)
    check_dir(s_output_dir)
    clear_dir(t_output_dir, skip_input=True)
    clear_dir(s_output_dir, skip_input=True)
    check_dir(mapping_out_dir)

    # process
    process_tech_nets(author_field, t_dir, t_output_dir)
    process_social_nets(author_field, s_dir, s_output_dir, mapping_path=mapping_out_source)


if __name__ == "__main__":
    # load input
    args_dict = parse_input(sys.argv)
    create_networks(args_dict)
    
