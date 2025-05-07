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
# # convert to UTF-8 coding
# def text_encoding(encoded_words):
#     # return ascii code
#     if "=" not in encoded_words:
#         return encoded_words
#     try:
#         encoded_word_regex = r"=\?{1}(.+)\?{1}([B|Q])\?{1}(.+)\?{1}="
#         charset, encoding, encoded_text = re.match(encoded_word_regex, encoded_words).groups()
#         if encoding == "B":
#             byte_string = base64.b64decode(encoded_text)
#         elif encoding == "Q":
#             byte_string = quopri.decodestring(encoded_text)
#         return byte_string.decode(charset)
#     except:
#         return encoded_words
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
    def gen_msg_id_lookup(df: pd.DataFrame, author_field: str) -> dict[str, tuple[str, str]]:
        """Generates a lookup of the message id to the author and timestamp.

        Args:
            df (pd.DataFrame): dataframe containing the email data
            author_field (str): name of the author field in the dataframe

        Returns:
            dict[str, tuple[str, str]]: message id: sender, timestamp
        """
        
        # store lookup
        email_to_author = {}

        # generate the lookup
        for _, row in df.iterrows():
            message_id = str(row["message_id"]).strip()
            sender_name = row[author_field]
            timestamp = row["date"]
            email_to_author[message_id] = (sender_name, timestamp)
        
        # export
        return email_to_author
    
    def update_sr_mapping(
        sender_dic: dict[str, list], project_name: str, message_id: str,
        sender_name: str, reference_id: str, prev_author: list[str],
        timestamp: str, prev_timestamp: str
    ) -> None:
        """Updates the sender-receiver mapping with a single 
        interaction.

        Args:
            sender_dic (dict[str, list]): sender-receiver mapping.
            project_name (str): project name.
            message_id (str): unique communication id for the 
                current communication.
            sender_name (str): sender of the email.
            reference_id (str): id for the message
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
    
    def clean_references(references: str) -> list[str]:
        """Cleans up the references into a list of references to be used in 
        joining nodes in the network.

        Args:
            references (str): references string, i.e. in_reply_to field.

        Returns:
            list[str]: list of cleaned references, one per entry
        """
        
        # deal with the issue that a line breaker exists in message_id:
        # e.g., <4\n829AB62.6000302@apache.org>
        references = references.replace("\n", " ").replace("\t", " ").split(" ")
        
        # remove empty references
        references = [r.strip() for r in references if r.strip()]

        # store references as a list of individual references
        new_refs = list()
        
        # if the references cross the whitespace boundary, try to combine them
        # into one reference
        for i in range(len(references) - 1):
            if "<" in references[i] and ">" not in references[i] and "<" not in references[i+1] and ">" in references[i+1]:
                new_refs.append(references[i] + references[i + 1])
        
        # add all valid references
        for r in references:
            if "<" in r and ">" in r:
                new_refs.append(r)
        
        # export references
        return new_refs
    
    def track_only_node(social_net: dict[str, dict[str, dict[str, int]]], sender_name: str, prev_author: str) -> None:
        """Tracks only the social node exists with no interaction; for now, 
        creates a self-referential edge with weight zero. DOES NOT track the 
        sender-receiver mapping for this pseudo-edge.

        Args:
            social_net (dict[str, dict[str, dict[str, int]]]): social network as
                as dictionary lookup.
            sender_name (str): sender name
            prev_author (str): previous author
        """
        
        # track the signal and ensure 0 weight
        track_social_signal(social_net, sender_name, prev_author)
        social_net[sender_name][prev_author]["weight"] = 0
    
    def process_communication(
        row: pd.DataFrame, social_net: dict[str, dict[str, dict[str, int]]],
        sender_dic: dict[str, list], msgid_to_author: dict[str, tuple[str, str]]
    ) -> None:
        """Processes a row of communication into the network and mapping. All 
        work happens in-place due to the mutable nature of dictionaries.

        Args:
            row (pd.DataFrame): entry in the social data.
            social_net (dict[str, dict[str, dict[str, int]]]): social network
                representation as a dictionary.
            sender_dic (dict[str, list]): sender-receiver mapping.
            msgid_to_author (dict[str, tuple[str, str]]): constant lookup of 
                message ID to author and timestamp.
        """
        
        # unpack useful information
        message_id = str(row["message_id"]).strip()
        references = str(row["in_reply_to"]).strip()
        sender_name = row[author_field]
        timestamp = row["date"]
        
        # REGARDLESS OF ANYTHING, this sender exists so the number of social 
        # nodes should reflect this person's existence; any future checks for an
        # early return or skipping will assume this node is already tracked
        if sender_name not in social_net:
            track_only_node(social_net, sender_name, sender_name)
        
        # ignores if this email does not reply to previous emails;
        # regardless of reply information, let's track that there exists a 
        # node in this month if we haven't already
        if pd.isna(references) or references == "None":
            return
        
        # clean and transform references into a list of references
        references = clean_references(references)

        # for each previous replier that this current communication refers 
        # to, we'll track the social activity. Note that we'll make sure to not
        # double count any of the activity to the same person, i.e. we can only
        # increase the weight of the edge by one per reply rather than allowing 
        # the weights to stack. e.g.
        #
        #   1. A sends email to B
        #   2. B replies to A
        #   3. A replies to B
        #   4. B replies to A
        #   5. A replies to B
        #
        # step 5 will only increase the weight of the edge between A & B by 1 
        # rather than by 4 (number of times A replied to B or vice versa)
        unique_prev_authors = set()
        
        for reference_id in references:
            # if we can't identify this communicator, we skip them
            if reference_id not in msgid_to_author:
                continue
            
            # unpack the previous author's information
            prev_author, prev_timestamp = msgid_to_author[reference_id]
            
            # if replying to themselves, continue but track that there's one 
            # social node with a self-referencing edge of weight zero
            if prev_author == sender_name:
                continue
            
            # if replying to someone already tracked in this email, skip
            if prev_author in unique_prev_authors:
                continue
            
            # update the social network
            update_sr_mapping(
                sender_dic, project_name, message_id, sender_name, reference_id,
                prev_author, timestamp, prev_timestamp
            )
            track_social_signal(social_net, sender_name, prev_author)
            
            # update the unique authors we've encountered
            unique_prev_authors.add(prev_author)
    
    def export_graph(
        social_net: dict[str, dict[str, dict[str, int]]], project_name: str,
        period: str | int, s_output: Path
    ) -> None:
        """Exports the graph for this project's month of data into an edgelist 
        format.

        Args:
            social_net (dict[str, dict[str, dict[str, int]]]): social network 
                representation of the graph for this month.
            project_name (str): name of the project.
            period (str | int): month number.
            s_output (Path): output directory.
        """
        
        # create the graph
        g = nx.DiGraph(social_net)
        
        # add any disconnected nodes
        g.add_nodes_from(social_net.keys())
        nx.write_edgelist(
            g, s_output / f"{project_name}__{str(period)}.edgelist",
            delimiter="##", data=["weight"]
        )
    
    # directory handling
    projects = os.listdir(s_source)
    check_dir(s_output)

    # get sender-receiver timestamp
    sender_dic = {
        "project": [], "message_id": [], "sender": [], "receiver": [], 
        "timestamp": [], "broadcast": []
    }

    # process each project's month of data
    for project in tqdm(projects):
        # setup project social network
        social_net = {}
        msgid_to_author = {}
        project_name, period = project.replace(".parquet", "").split("__")

        # load project data & ensure only valid communications are considered
        df = pd.read_parquet(s_source / project, engine=PARQUET_ENGINE)
        df.query("is_bot == False", inplace=True)
        df = df[df[author_field].notna()]
        
        # generate a lookup of the email ID to author
        msgid_to_author = gen_msg_id_lookup(df, author_field)

        # for each communication in the social data, let's track the 
        # sender-receiver relationship in a graph
        for _, row in df.iterrows():
            process_communication(row, social_net, sender_dic, msgid_to_author)
        if project_name == "ActionBarSherlock" and period == "78":
            print(json.dumps(social_net))

        # save as directed graph
        export_graph(social_net, project_name, period, s_output)

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

