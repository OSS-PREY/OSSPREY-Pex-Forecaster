"""
    @brief One-time script for combining individual csvs into one dataset.
    @author Arjun Ashok (arjun3.ashok@gmail.com)
"""

# ------------- Environment Setup ------------- #
# external packages
import pandas as pd
from tqdm import tqdm

# built-in modules
import json
import mailbox
import os
import sys
import re
from pathlib import Path

# DECAL modules
import decalforecaster.utils as util
import decalforecaster.abstractions.rawdata as rd


# ------------- utility ------------- #
def clean_strings(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, int]:
    """
        Clean strings in the df to not cause issues in reading.
    """

    # setup
    df_bef = df.copy()

    # clean
    df[cols] = df[cols].replace(
        r"[\r\n\cr\r_nl_]+"
        ". ",
        regex=True
    )

    # remove any chars that can interfere with network creation
    for col in cols:
        # replace any network delims
        df[col] = df[col].str.replace(
            "#",
            ""
        )

        # replace any newlines
        df[col] = df[col].str.replace(
            r"\r\n|\r|\n",
            ". ",
            regex=True
        )
    
    # report
    num_unique_replaced_rows = len(df[df[cols].ne(df_bef[cols]).any(axis=1)].drop_duplicates())
    return df, num_unique_replaced_rows


def clean_file(filepath: str) -> None:
    file_str = ""
    with open(filepath, "r") as input_file:
        for line in input_file:
            # replace all line terminators except newline characters ('\n')
            modified_line = line.replace("\r", ". ").replace("_nl_", ". ")
            file_str += modified_line

    with open(filepath, "w") as output_file:
        output_file.write(file_str)


def enforce_cols(incubator: str, df: pd.DataFrame, dtype: str) -> None:
    """
        Ensures columns exist.
    """

    # remove cols
    remove_cols = [col for col in df.columns if "[IGNORE]" in col]
    remove_cols += ["body", "commit_msg", "reactions", "reactions\r"]

    remove_cols = set(remove_cols) & set(df.columns)
    for col in remove_cols:
        df.drop(labels=[col], axis=1, inplace=True)

    # check
    if "dealised_author_full_name" not in df.columns:
        df["dealised_author_full_name"] = df[util._load_params()["author-source-field"][incubator]]
    if "is_bot" not in df.columns:
        df["is_bot"] = 0
    if dtype == "tech" and "is_coding" not in df.columns:
        df["is_coding"] = 1
    if dtype == "social" and "in_reply_to" not in df.columns:
        df["in_reply_to"] = ""
    if dtype == "social" and "message_id" not in df.columns:
        df["message_id"] = list(range(df.shape[0]))


def clean_proj_name(proj_name: str) -> str:
    """
        Cleans the project name.
    """
    
    # lowercase
    proj_name = proj_name.lower()

    # remove descriptive info (anything in parentheses or after hyphen/colon)
    proj_name = re.sub(r"\s*\([^()]*\)\s*", "", proj_name)
    proj_name = re.sub(r"[-:].*$", "", proj_name)
    
    # remove any slashes (messes up with directories)
    proj_name = proj_name.replace("/", "-")

    # replace spaces
    proj_name = proj_name.strip()
    proj_name = proj_name.replace(" ", "-")

    # return cleaned str
    return proj_name


def rearrange_project_groups(incubator: str) -> None:
    """
        One-time script for rearranging the data in the JSON for aggregating 
        projects.
    """

    # load
    with open(f"./utility/{incubator}_project_groups.json", "r") as f:
        proj_groups = json.load(f)

    # remove technology groups
    pure_proj_groups = dict()

    for technology, projects_data in proj_groups.items():
        pure_proj_groups.update(projects_data)
    
    # remove empty projects
    pure_proj_groups = {k: v for k, v in pure_proj_groups.items() if len(v) > 0}

    # clean project names (hyphens + lowercase)
    cleaned_proj_groups = {clean_proj_name(k): v for k, v in pure_proj_groups.items()}

    # flip for quick lookup
    flipped_proj_groups = dict()
    for parent, children in cleaned_proj_groups.items():
        # add parent
        flipped_proj_groups[parent] = parent

        # add all children
        for proj in children:
            flipped_proj_groups[proj] = parent

    # export
    util._log(set(flipped_proj_groups.values()))
    if input("y/n: ") == "y":
        with open(f"./utility/{incubator}_project_groups.json", "w") as f:
            json.dump(flipped_proj_groups, f, indent=4)
        
        if input("continue? [y/n]: ") != "y":
            exit(1)
    else:
        exit(1)


def aggregate_projects(incubator: str, df: pd.DataFrame) -> pd.DataFrame:
    """
        Given a JSON with projects and their grouped subprojects, constructs a 
        dataframe of their aggregated projects' data. Essentially replaces every 
        project with its parent.
    """

    # load lookup
    params_dict = util._load_params()
    util_dir = params_dict["ref-dir"]
    
    with open(Path(util_dir) / f"{incubator}_project_groups.json", "r") as f:
        proj_groups = json.load(f)
    
    # transform each project name
    def identify_proj(proj_name: str) -> str:
        """
            Utility for quick transformation.
        """
        
        # lookup, otherwise return the project itself if it doesn't exist
        return proj_groups.get(proj_name, proj_name)
    
    df["project_name"] = df["project_name"].apply(identify_proj)
    return df


def reinfer_project_status(incubator: str) -> None:
    """
        Given an updated mapping of project groups, we can regenerate the 
        project status dictionary with the new names for each project.
    """

    # load project status & groups
    params_dict = util._load_params()
    with open(params_dict["project-status"][incubator], "r") as f:
        proj_status = json.load(f)
    with open(f"./utility/{incubator}_project_groups.json", "r") as f:
        proj_groups = json.load(f)

    # replace names
    proj_status = {k: set(v) for k, v in proj_status.items()}       # set for quick lookup

    for proj, parent in proj_groups.items():
        if parent not in proj_status["graduated"] and proj in proj_status["graduated"]:
            proj_status["graduated"].remove(proj)
            proj_status["graduated"].add(parent)
        elif parent not in proj_status["retired"] and proj in proj_status["retired"]:
            proj_status["retired"].remove(proj)
            proj_status["retired"].add(parent)
        elif parent not in proj_status["incubating"] and proj in proj_status["incubating"]:
            proj_status["incubating"].remove(proj)
            proj_status["incubating"].add(parent)
        elif (parent not in proj_status["graduated"]) and (parent not in proj_status["retired"])\
            and (parent not in proj_status["incubating"]) and (parent != proj):
            util._log(f"skipping project {proj} w/ parent {parent}", "warning")
    
    # export
    proj_status = {k: list(v) for k, v in proj_status.items()}
    with open(params_dict["project-status"][incubator], "w") as f:
        json.dump(proj_status, f, indent=4)


def clean_project_status(incubator: str) -> None:
    """
        Cleans the project names in the project status JSON to match the 
        cleaned names in the dataset.
    """

    # load project status
    params_dict = util._load_params()
    with open(params_dict["project-status"][incubator], "r") as f:
        ps = json.load(f)
    
    # clean
    ps = {k: [clean_proj_name(proj_name) for proj_name in v] for k, v in ps.items()}

    # export
    with open(params_dict["project-status"][incubator], "w") as f:
        json.dump(ps, f, indent=4)


def combine_tech(incubator: str, params_dict: dict, tech_out: str) -> pd.DataFrame:
    """
        Iterates a directory of individual project commit histories and combines 
        into one dataset.
    """

    # explicitly define columns
    col_mapper = params_dict["field-mappings"][incubator]["tech"]
    dataset_dir = util._load_params()["dataset-dir"]

    # iterate dir
    input_dir = f"{dataset_dir}/{incubator}_data/{params_dict['tech-type'][incubator]}-raw/"
    col_names = list(col_mapper.values())
    dfs: list[pd.DataFrame] = []
    projects: list[str] = []
    total_changed = 0

    for filename in tqdm(os.listdir(input_dir)):
        # unpack
        filepath = os.path.join(input_dir, filename)
        proj_name = Path(filepath).stem
        projects.append(proj_name)

        clean_file(filepath)
        proj_df = pd.read_csv(filepath, engine="c", names=col_names)

        # clean df
        clean_string_cols = ["commit_id", "file_name", "commit_msg"]
        proj_df, num_changed = clean_strings(proj_df, clean_string_cols)

        # ensure project name
        proj_df["project_name"] = proj_name

        # add
        total_changed += num_changed
        dfs.append(proj_df)

    # merge
    util._log("merging...")
    tech_df = pd.concat(dfs, ignore_index=True)

    # check all projects included
    if set(projects) != set(tech_df["project_name"]):
        util._log(f"missing projects from compiled dataframe, likely due to blank files: {set(projects) - set(tech_df['project_name'])}", "warning")
    
    util._log("dropping cols...")
    enforce_cols(incubator, tech_df, "tech")

    # # months
    # util._log("imputing months...")
    # tech_dict = rd.impute_months({"tech": tech_df}, copy=False)

    # # aggregate projects together
    # tech_dict["tech"] = aggregate_projects(incubator, tech_dict["tech"])

    # export
    # util._log("exporting...")
    # tech_dict["tech"].to_csv(tech_out, index=False)
    util._log(f"\t<Number of Rows w/ Replacements> :: {total_changed}\n")
    return tech_df


def combine_social(incubator: str, params_dict: dict, social_out: str) -> pd.DataFrame:
    """
        Iterates a directory of individual project emails records and combines 
        into one dataset.
    """

    # explicitly define columns
    col_mapper = params_dict["field-mappings"][incubator]["social"]
    col_names = list(col_mapper.values())

    # conversion utility
    def conv_mbox(path: str) -> pd.DataFrame:
        """
            Reads mbox file into a pandas dataframe.
        """

        # generate dict
        msg_data = {col: [] for col in col_names}

        # read mbox
        with open(path, "rb") as f:
            # read file
            mbox = mailbox.mbox(f)

            # iterate messages
            for msg in mbox:
                # unpack & update
                msg_data = {msg_data[col].append(msg[col]) for col in col_names}
        
        # export
        return pd.DataFrame(msg_data, columns=list(col_names.keys()))
    

    # iterate dir
    dataset_dir = util._load_params()["dataset_dir"]
    input_dir = f"{dataset_dir}/{incubator}_data/{params_dict['social-type'][incubator]}-raw/"
    social_df = pd.DataFrame(columns=col_names)
    dfs: list[pd.DataFrame] = []
    projects: list[str] = []
    total_changed = 0

    for filename in tqdm(os.listdir(input_dir)):
        # unpack
        filepath = os.path.join(input_dir, filename)
        proj_df = pd.read_csv(filepath, engine="c", low_memory=False, lineterminator="\n")
        proj_df.rename(columns=col_mapper, inplace=True)
        proj_name = Path(filepath).stem
        projects.append(proj_name)

        # ensure project name
        proj_df["project_name"] = proj_name

        # clean
        clean_file(filepath)
        clean_string_cols = ["subject", "title"]
        proj_df, num_changed = clean_strings(proj_df, clean_string_cols)

        # add
        total_changed += num_changed
        dfs.append(proj_df)

    # merge
    util._log("merging...")
    social_df = pd.concat(dfs, ignore_index=True)
    enforce_cols(incubator, social_df, "social")

    # check all projects included
    if set(projects) != set(social_df["project_name"]):
        util._log(f"missing projects from compiled dataframe, likely due to blank files: {set(projects) - set(social_df['project_name'])}", "warning")

    # # months + export
    # util._log("imputing months...")
    # social_dict = rd.impute_months({"social": social_df}, copy=False)

    # # aggregate projects together
    # social_dict["social"] = aggregate_projects(incubator, social_dict["social"])
    
    # util._log("exporting...")
    # social_dict["social"].to_csv(social_out, index=False)
    util._log(f"\t<Number of Rows w/ Replacements> :: {total_changed}")
    return social_df


def incubator_augs(incubator: str, params_dict: dict, tech_out: str, social_out: str) -> None:
    """
        Performs the incubator-wise augmentations on the data prior to 
        pre-processing. Reads in the final CSVs produced for the base-form and 
        adjusts them, exporting to the same b

        @param incubator: incubator to use the strategy for
    """

    # define utility and call it
    util._log("augmenting specific to incubator...")
    def eclipse_augs(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """
            Ensures emails as usernames, infers subject-line from email title.
        """

        # setup
        s = dfs["social"]
        t = dfs["tech"]

        # ensure sender name is only the username
        s["sender_name"] = s["sender_name"].str.split("@").str[0]
        s["sender_name"] = s["sender_name"].str.replace("#", "")
        s["dealised_author_full_name"] = s["sender_name"]

        # ensure subject line is correct for emails (i.e. remove Re: and then 
        # pick correct title instead of URL)
        email_rows = s["type"] == "email"
        s.loc[email_rows, "title"] = s.loc[email_rows, "title"].str.replace(
            "Re:", ""
        )
        s.loc[email_rows, "subject"] = s.loc[email_rows, "title"].apply(
            lambda x: re.sub(r"\s+", "", x)
        )

        # return
        return dict(zip(["tech", "social"], [t, s]))

    # load & call
    new_versions = {"tech": "0", "social": "0"}
    dfs = rd._load_data(rd._load_paths(incubator, new_versions))
    router = {
        "eclipse": eclipse_augs
    }

    dfs = router[incubator](dfs)

    # export
    rd._save_data(dfs, incubator, new_versions)


# ------------- program ------------- #
def main(incubator: str, params_dict: dict, tech_out: str, social_out: str):
    # combine + generate lookup
    util._log(f"Combining Data for {incubator}", "new")
    df_lookup = dict()
    df_lookup["tech"] = combine_tech(incubator, params_dict, tech_out)
    df_lookup["social"] = combine_social(incubator, params_dict, social_out)

    # save
    versions = {"tech": "0", "social": "0"}
    rd._save_data(df_lookup, "eclipse", versions)

    # months augment
    df_lookup = rd._load_data(rd._load_paths(incubator, versions))
    df_lookup = rd.impute_months(df_lookup, copy=False)

    rd._save_data(df_lookup, "eclipse", versions)
    incubator_augs(incubator, params_dict, tech_out, social_out)


# script
if __name__ == "__main__":
    # unpack args
    incubator = str(sys.argv[1])
    params_dict = util._load_params()

    # # clean project groups
    # rearrange_project_groups(incubator)
    
    # combine
    dataset_dir = Path(params_dict["dataset-dir"])
    main(
        incubator=incubator,
        params_dict=params_dict,
        tech_out= dataset_dir / f"{incubator}_data" / f"{params_dict['augmentations'][incubator]['tech']['0']}.{params_dict['ext']}",
        social_out= dataset_dir / f"{incubator}_data" / f"{params_dict['augmentations'][incubator]['social']['0']}.{params_dict['ext']}"
    )

