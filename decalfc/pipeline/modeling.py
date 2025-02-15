"""
    @brief Modeling framework w/ testing built-in for switching out model types, 
           testing accuracies with different methods, and augmenting data prior
           to testing. 
    @author Arjun Ashok (arjun3.ashok@gmail.com)
    @acknowledgements Nafiz I. Khan, Dr. Likang Yin
    @creation-date October 2023
"""


# ---------------- Environment Setup ---------------- #
# external modules
import pandas as pd
from tqdm import tqdm

# built-in modules
import sys
from typing import Iterable, Any
from itertools import product, permutations, chain, combinations

# DECAL modules
import decalfc.utils as util
from decalfc.abstractions.modeldata import *
from decalfc.abstractions.perfdata import *
from decalfc.abstractions.tsmodel import *


# ---------------- modeling script ---------------- #
def modeling(params_dict: dict, args_dict: dict, *args, **kwargs):
    """
        Wraps modeling functionality.

        @param params_dict: params dictionary, centralized args
        @param args_dict: defines 'strategy' and any hyperparameters to override
    """

    # load all data
    md = ModelData(
        transfer_strategy=args_dict["strategy"],
        transform_kwargs=args_dict.get("transform-kwargs", dict()),
        soft_prob=args_dict.get("soft-prob", 1)
    )

    # train model & test
    ## grab sample tensor, i.e. first tensor we have
    sample_tensor = md.tensors["train"]["x"][0]
        
    ## ensure some hyperparams
    hyperparams = {"input_size": sample_tensor.shape[1]}
    hyperparams.update(args_dict.get("hyperparams", dict()))

    ## build model
    model = TimeSeriesModel(
        model_arch=args_dict.get("model-arch", "BLSTM"), hyperparams=hyperparams
    )
    
    # testing
    # model.visualize_model(
    #     data_shape=(md.tensors["train"]["x"][0].shape[0], hyperparams["input_size"])
    # )
    model.train_model(md)
    model.test_model(md)
    # model.interpret_model(md, strategy=hyperparams.get("interpretation-strategy", "SHAP"))

    # reporting
    model.report(display=True, save=False)

    if args_dict.get("perf-path", None) is None:
        perf_db = PerfData()
    else:
        perf_db = PerfData(args_dict["perf-path"])
    perf_db.add_entry(
        md.transfer_strategy,
        model_arch=model.model_arch,
        preds=model.preds,
        targets=model.targets,
        intervaled=md.is_interval["test"]
    )

    if md.is_interval["test"]:
        perf_db.perf_vs_time(md.transfer_strategy, model.model_arch)

    # monthly predictions
    if args_dict.get("monthly-preds", False):
        model.monthly_predictions(
            **args_dict["monthly-preds"]
        )


def monthly_predictions(params_dict: dict[str, Any], args_dict: dict[str, Any]) -> None:
    """
        Generates the predictions for every month of a given incubator's 
        projects. This utilizes the tsmodel's soft probability ability to 
        associate a project with its individual outcomes.

        @param params_dict: params.json
        @param args_dict: args to pass in that contain all the usual parameters
            plus any of the following specifics:
            
            @param transfer_strategy: str; notice that dynamic strategies for 
                model training are only enabled if the flags for train/test, 
                i.e. ^/^^, are used. If no flags are used, static is assumed. 
                This is used to specify how to train the model used for all soft 
                probabilities.
            
            @param projects: lookup of iterable of projects to use for that 
                incubator. If the set is equal to None, all projects from that 
                incubator are used. Note: if not specified, the default versions 
                will be used.
            
            @param gen_perf: perf vs time, track in perfdata. Defaults to True.
            
            @param greedy_load: use pre-trained result for the prediction; note, 
                this only works if we're not doing the train on all but one, 
                test on one strategy. Defaults to True.
    """
    
    # auxiliary functions
    def route_model(model: TimeSeriesModel, train_strat: str, route_strat: str,
                    soft_prob_model_strat: str=None, train_inc: bool=False,
                    attempt_load: bool=True) -> TimeSeriesModel:
        """
            Routes the correct model based on how the weights should be updated.

            @param model (TimeSeriesModel): current model, to aid in optimizing
                loading for static weights
            @param train_strat (str): transfer strategy to use when training or 
                loading the model
            @param route_strat (str): "static" or "dynamic". Static will attempt 
                to find the best generalizing model based on how the 
                transfer/training is supposed to happen. Dynamic uses the all 
                but one, predict on one strategy. Defaults to "static".
            @param soft_prob_model_strat: strategy to use for loading the model
                in and training.
            @param train_inc: if the target incubator is used in training

        Returns:
            TimeSeriesModel: instance of the model ready for soft probabilities
        """
        
        # return if model is initialized and the strategy is static
        if (model) and (route_strat == "static"):
            return model

        # re-train; branch based on what is the optimal path, but first 
        # determine the optimal validation incubators
        
        ## optimal validation
        if soft_prob_model_strat is None:
            ### get all incubators not used for training to gauge 
            ### generalizability
            incubator_abbrv = set(util.load_params()["abbreviation"].keys())
            train_incubators = {char for char in train_strat if char.isupper()}
            validation_incubators = incubator_abbrv - train_incubators
            
            ### use all incubators if they're all used in the train
            if len(validation_incubators) == 0:
                validation_incubators = incubator_abbrv
            
            ### in progress
            soft_prob_model_strat = f"{train_strat} --> {optimal_validation}"
            raise NotImplementedError("in progress :(")
        
        
        ## optimal is to re-train each time
        if train_inc and route_strat == "dynamic":
            md = ModelData(
                transfer_strategy=soft_prob_model_strat,
                predict_project={target_incubator: proj}
            )
        
        ## optimal is to use train one model, but we don't train on the target
        if (not train_inc) and (route_strat == "static"):
            md = ModelData(
                transfer_strategy=soft_prob_model_strat
            )
        
        if train_inc and route_strat == "static":
            md = ModelData(
                transfer_strategy=soft_prob_model_strat
            )
        
            
        ## training
        model = TimeSeriesModel(
            model_arch=model_arch,
            hyperparams=hyperparams
        )
        model.train_model(md=md, attempt_load=attempt_load)
        
        # export
        return model
        
    
    # unpack global arguments
    ## user specified
    strat = str(args_dict["strategy"])
    soft_prob_model_strat = str(args_dict["soft-prob-strat"])
    model_arch = args_dict.get("model-arch", "BLSTM")
    projects = set(args_dict.get("projects", set()))
    hyperparams = args_dict.get("hyperparams", dict())
    attempt_load = args_dict.get("load-model", True)
    
    ## inferred
    train_strat, test_strat = strat.split("-->", maxsplit=1)
    train_strat, test_strat = train_strat.strip(), test_strat.strip()
    target_incubator_abbrv = test_strat[0]
    target_incubator = ([
        inc for inc, abbrv in params_dict["abbreviations"].items() 
        if abbrv == target_incubator_abbrv
    ])[0]
    
    ## routing inference
    route_strat = "dynamic" if "^" in train_strat else "static"
    train_inc = target_incubator_abbrv in train_strat
        
    ## reporting progress
    util.log(
        f"\n\n<Using `{train_strat}` to train>", log_type="none", output="file"
    )
    
    # load the data we need for training and soft probs
    md = ModelData(transfer_strategy=strat, soft_prob=0)
    augments = next(iter(md.options["test"].items()))[1]
    nd = NetData(target_incubator, options=augments)
    
    ## ensure the projects set is valid
    if len(projects) == 0:
        projects = nd.base_projects
        
    ### ensure fits within downsampled, etc. trials
    projects &= nd.projects_set
    
    # setup before trials begin
    ## setup tracking for performance
    perf_db = PerfData(
        args_dict.get("perf-path", "../model-reports/databases/performance_db")
    )
    full_preds = dict()
    full_targets = dict()
    confusion_matrix = dict(zip(
        ["true-positive", "false-positive", "false-negative", "true-negative", "incubating"],
        [list(), list(), list(), list(), list()]
    ))
    
    ## setup model
    ts_model = None
    hyperparams["input_size"] = md.tensors["train"]["x"][0].shape[1]

    ## setup dir to ensure we can output without accidentally forgetting to
    ## overwrite
    dir = f"../predictions/{target_incubator}/"
    util.clear_dir(dir, skip_input=True)
    util.check_dir(dir)
    
    # for each project, iteratively generate and export the soft probabilities
    for proj in tqdm(projects):
        ## setup data to push into the soft prob model
        tensors = nd._regular_tensors(subset=proj)
        X, y = tensors["x"][0], tensors["y"][0]
        
        ## generate intervals ourselves without having to associate all pseudo 
        ## projects
        X_dict = {i: X[:i, ...] for i in range(1, X.shape[0] + 1)}
        
        ## get the soft prob model
        ts_model = route_model(
            ts_model, train_strat, route_strat, soft_prob_model_strat, train_inc,
            attempt_load
        )
        
        ## soft probs
        soft_probs = ts_model.soft_probs(X_dict)
        soft_probs = ({
            timestep: proba.cpu().detach().numpy().tolist()
            for timestep, proba in soft_probs.items()
        })
        
        ## export soft probabilities
        ### create a dataframe for easy augmentation
        df = pd.DataFrame.from_dict(soft_probs, orient="index")
        df.reset_index(inplace=True)
        df.rename(columns={"index": "month", 0: "close"}, inplace=True)
        
        ### skip any short projects
        if df.shape[0] == 0:
            continue
        
        ### save forecasts
        proj_name = proj.replace("/", "_")
        df.to_csv(f"{dir}{proj_name}_f_data.csv", index=False)
        
        ## export performance data
        ### unpack predictions vs target
        final_pred = df.iloc[-1, 1]
        rounded_pred = float(round(final_pred))
        final_target = int(y.cpu().detach().numpy()[0])

        ### update confusion matrix & log file
        if final_target == -1:
            util.log(f"Incubating: {proj}, predicted {rounded_pred} w/ {final_pred}", "log", output="file")
            confusion_matrix["incubating"].append(proj)
        elif rounded_pred != final_target:
            if (final_target == 0.0) and (final_pred >= 0.5):
                confusion_matrix["false-positive"].append(proj)
            elif (final_target == 1.0) and (final_pred < 0.5):
                confusion_matrix["false-negative"].append(proj)
            util.log(f"MIS-PREDICTION: expected {final_target}, got {rounded_pred} with {final_pred} for {proj}", "warning", output="file")
        else:
            if (final_target == 1.0) and (final_pred >= 0.5):
                confusion_matrix["true-positive"].append(proj)
            if (final_target == 0.0) and (final_pred < 0.5):
                confusion_matrix["true-negative"].append(proj)
            util.log(f"Correct Prediction: expected {final_target}, got {final_pred} for {proj}", "log", output="file")

        ### performance logging in perfdata; first round to hard probs then 
        ### track
        hard_probs = {int(m): round(pred[0]) for m, pred in soft_probs.items()}

        ### compile into final 
        for m, p in hard_probs.items():
            if m not in full_preds:
                full_preds[m] = list()
                full_targets[m] = list()
            full_preds[m].append(p)
            full_targets[m].append(final_target)
            
    # export the final report
    count_confusion_matrix = {k: len(v) for k, v in confusion_matrix.items()}
    print(json.dumps(count_confusion_matrix, indent=4))
    util.log(json.dumps(count_confusion_matrix, indent=4), output="file")
    with open(Path(dir) / "__confusion_matrix.json", "w") as f:
        json.dump(confusion_matrix, f, indent=4)
    with open(Path(dir) / "__confusion_matrix.json", "a") as f:
        json.dump(count_confusion_matrix, f, indent=4)
    
    # print(json.dumps(full_preds, indent=4))
    # print(json.dumps(full_targets, indent=4))
    # print(type(next(iter(full_preds))))
    # exit()
    # perf_db.add_entry(
    #     strat,
    #     model_arch=model_arch,
    #     preds=full_preds,
    #     targets=full_targets,
    #     intervaled=True
    # )
    # perf_db.perf_vs_time(
    #     transfer_strategy=strat, model_arch=model_arch, stop_month=1000
    # )
    
    # end function
    return


def full_trials(params_dict: dict[str, Any], trials: int=10, model_arch: str="BLSTM", **kwargs) -> None:
    """
        Generates all trials.
    """

    # schema
    versions = {
        "apache": [
            ("A", 0, 0),
            ("A", 1, 1)
        ],
        "github": [
            ("G", 0, 0),
            ("G", 1, 1),
            ("G", 1, 2),
            ("G", 2, 3),
            ("G", 3, 4)
        ],
        "eclipse": [
            ("E", 0, 0),
            ("E", 1, 1)
        ]
    }
    frameworks = {
        "self": "{}-{}-{}^ --> {}-{}-{}^^",
        "split": "{}-{}-{} --> {}-{}-{}",
        "even-mix": "{}-{}-{}^ + {}-{}-{}^ --> {}-{}-{}^^ + {}-{}-{}^^",
        "uneven-mix": "{}-{}-{}^ + {}-{}-{} --> {}-{}-{}^^"
    }

    # generate all trials
    version_combos = list(product(*versions.values()))
    version_combos_extended = version_combos + [(t[1], t[0]) for t in version_combos]

    # setup trials
    hyperparams = args_dict.get("hyperparams", dict())

    # define wrappers
    def self_trials():
        self_combos = [v for version in versions.values() for v in version]
        for version_combo in self_combos:
            # run modeling program
            args_dict = {
                "strategy": frameworks["self"].format(
                    version_combo[0],
                    version_combo[1],
                    version_combo[2],
                    version_combo[0],
                    version_combo[1],
                    version_combo[2]
                ),
                "model-arch": model_arch,
                "hyperparams": hyperparams
            }
            
            for i in range(trials):
                modeling(params_dict=params_dict, args_dict=args_dict)

    def split_trials():
        for version_combo in version_combos_extended:
            # run modeling program
            args_dict = {
                "strategy": frameworks["split"].format(
                    version_combo[0][0],
                    version_combo[0][1],
                    version_combo[0][2],
                    version_combo[1][0],
                    version_combo[1][1],
                    version_combo[1][2],
                    version_combo[0][0],
                    version_combo[0][1],
                    version_combo[0][2],
                    version_combo[1][0],
                    version_combo[1][1],
                    version_combo[1][2]
                ),
                "model-arch": model_arch,
                "hyperparams": hyperparams
            }

            for i in range(trials):
                modeling(params_dict=params_dict, args_dict=args_dict)

    def even_mix_trials():
        for version_combo in version_combos:
            # run modeling program
            args_dict = {
                "strategy": frameworks["even-mix"].format(
                    version_combo[0][0],
                    version_combo[0][1],
                    version_combo[0][2],
                    version_combo[1][0],
                    version_combo[1][1],
                    version_combo[1][2],
                    version_combo[0][0],
                    version_combo[0][1],
                    version_combo[0][2],
                    version_combo[1][0],
                    version_combo[1][1],
                    version_combo[1][2]
                ),
                "model-arch": model_arch,
                "hyperparams": hyperparams
            }
            
            for i in range(trials):
                modeling(params_dict=params_dict, args_dict=args_dict)
    
    def uneven_mix_trials():
        for version_combo in version_combos_extended:
            # run modeling program
            args_dict = {
                "strategy": frameworks["uneven-mix"].format(
                    version_combo[0][0],
                    version_combo[0][1],
                    version_combo[0][2],
                    version_combo[1][0],
                    version_combo[1][1],
                    version_combo[1][2],
                    version_combo[0][0],
                    version_combo[0][1],
                    version_combo[0][2]
                ),
                "model-arch": model_arch,
                "hyperparams": hyperparams
            }

            for i in range(trials):
                modeling(params_dict=params_dict, args_dict=args_dict)

    # calls
    self_trials()
    split_trials()
    even_mix_trials()
    uneven_mix_trials()


def breakdown(params_dict: dict[str, Any], args_dict: dict[str, Any]) -> None:
    """
        Given a set of options, generates a breakdown surrounding those options.
        NOTE: follows a very similar structure to the full trials

        self: <incubator> --> <same incubator>
        split: <k - 1 incubators except one> --> <kth incubator>
        even-mix: <k incubators> --> <k incubators>
        uneven-mix: <k - 1 incubators, kth is train-test split> --> <kth test split>

        Notice we'll check k = 2, 3, . . ., len(versions) for the mixes and 
        split.

        @param params_dict: centralized parameter structure
        @param args_dict: arguments passed in; should include some specification 
                          of the options selected (as a str of characters to 
                          append)
    """

    # unpack args
    options_str = args_dict["options"]
    model_arch = args_dict.get("model-arch", "BLSTM")
    trials = args_dict.get("trials", 2)
    hyperparams = args_dict.get("hyperparams", dict())

    # infer arguments
    versions = params_dict["default-versions"]
    versions = [f"{params_dict['abbreviations'][k]}-{v[0]}-{v[1]}" for k, v in versions.items()]
    num_versions = len(versions)
    num_mixes = range(2, num_versions + 1)
    placeholder = "{}"

    # define wrappers
    def self_trials(versions):
        # define schema
        schema = f"{placeholder}^{options_str} --> {placeholder}^^{options_str}"

        # every self trial
        for version in versions:
            # run modeling program
            args_dict = {
                "strategy": schema.format(
                    version,
                    version
                ),
                "model-arch": model_arch,
                "hyperparams": hyperparams
            }
            
            for i in range(trials):
                modeling(params_dict=params_dict, args_dict=args_dict)

    def split_trials(versions):
        # for each number of mixes
        for num_mix in num_mixes:
            # define schema
            schema = (
                " + ".join([f"{placeholder}{options_str}" for _ in range(num_mix - 1)]) + 
                f" --> {placeholder}{options_str}"
            )

            # for each permutation
            for permutation in permutations(versions, num_mix):
                # run modeling program
                args_dict = {
                    "strategy": schema.format(
                        *permutation
                    ),
                    "model-arch": model_arch,
                    "hyperparams": hyperparams
                }

                for i in range(trials):
                    modeling(params_dict=params_dict, args_dict=args_dict)

    def even_mix_trials(versions):
        # for each number of mixes
        for num_mix in num_mixes:
            # generate schema
            schema = (
                " + ".join([f"{placeholder}^{options_str}" for _ in range(num_mix)]) + 
                " --> " + 
                " + ".join([f"{placeholder}^^{options_str}" for _ in range(num_mix)])
            )

            # for each permutation unless every permutation is equal (i.e. following edge case)
            if num_mix == num_versions:
                args_dict = {
                    "strategy": schema.format(
                        *(versions + versions)
                    ),
                    "model-arch": model_arch,
                    "hyperparams": hyperparams
                }
                
                for i in range(trials):
                    modeling(params_dict=params_dict, args_dict=args_dict)
                
                break

            for permutation in permutations(versions, num_mix):
                # versions rotation
                permutation = permutation + permutation

                # run modeling program
                args_dict = {
                    "strategy": schema.format(
                        *permutation
                    ),
                    "model-arch": model_arch,
                    "hyperparams": hyperparams
                }
                
                for i in range(trials):
                    modeling(params_dict=params_dict, args_dict=args_dict)

    def uneven_mix_trials(versions):
        # for each k
        for num_mix in num_mixes:
            # schema
            schema = (
                f"{placeholder}^{options_str} + " + 
                " + ".join([f"{placeholder}{options_str}" for _ in range(num_mix - 1)]) + 
                f" --> {placeholder}^^{options_str}"
            )

            for permutation in permutations(versions, num_mix):
                # versions rotation
                permutation = permutation + permutation

                # run modeling program
                args_dict = {
                    "strategy": schema.format(
                        *permutation
                    ),
                    "model-arch": model_arch,
                    "hyperparams": hyperparams
                }

                for i in range(trials):
                    modeling(params_dict=params_dict, args_dict=args_dict)

    # run trials
    self_trials(versions)
    split_trials(versions)
    even_mix_trials(versions)
    uneven_mix_trials(versions)

    # generate reports
    p = PerfData()
    p.subset_breakdown(options=options_str)
    p.comparison(field="transfer_strategy")


def incubator_breakdown(params_dict: dict[str, Any], args_dict: dict[str, Any],
                        augmentations: list[str]=None) -> None:
    """
        Goes through all augmentations and generates a table for easy viewing 
        and augmentation comparison. The end goal is a short-list of strategies 
        for a given incubator's transfer.

        @param params_dict: centralized parameter structure
        @param args_dict: arguments passed in; should include some specification 
                          of the incubator needed (e.g. "apache").
        @param augmentations: list of options strings to feed into the training 
                              scheme; note, all incubators will have this 
                              augmentation regime applied, including the test to 
                              ensure transfer
    """

    # auxiliary functions
    def powerset(iterable):
        """Generates powerset from iteratable object."""
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    
    def all_subsets(lst):
        """Generate all subsets of a list."""
        return list(chain.from_iterable(combinations(lst, r) for r in range(len(lst) + 1)))

    def all_combinations_of_subsets(lists_of_lists):
        """Generate all combinations of subsets for a list of lists."""
        # Generate all subsets for each list
        subsets = [all_subsets(lst) for lst in lists_of_lists]
        
        # Generate the Cartesian product of these subsets
        return list(product(*subsets))

    def convert_combinations_to_strings(combinations):
        """Convert combinations of subsets to concatenated strings."""
        result = []
        for combination in combinations:
            # Filter out empty tuples and concatenate non-empty ones
            concatenated_string = "".join("".join(subset) for subset in combination)
            result.append(concatenated_string)
        return result

    # unpack args
    incubator = args_dict["incubator"]
    model_arch = args_dict.get("model-arch", "BLSTM")
    trials = args_dict.get("trials", 2)
    hyperparams = args_dict.get("hyperparams", dict())

    # infer arguments
    versions = params_dict["default-versions"]
    versions = {k: f"{params_dict['abbreviations'][k]}-{v[0]}-{v[1]}" for k, v in versions.items()}
    num_incubators = len(versions) - 1

    if augmentations is None:
        augmentations = params_dict["network-aug-groups"]
        augmentations = all_combinations_of_subsets(augmentations)
        augmentations = convert_combinations_to_strings(augmentations)
    
    num_augmentations = len(augmentations)

    # report prior to continuing
    print(f"Breakdown for {incubator}", "new")
    print(f"Testing {num_augmentations} augmentations")
    print(f"Trying every combination of incubators, i.e. {num_incubators * 2 * trials} trials per augmentation")

    # split into train section, test section
    train_versions = [v for k, v in versions.items() if k != incubator]
    test_version = versions[incubator]

    # for every subset of training 
    for train_subset in powerset(train_versions):
        # for every augmentation we want to test
        for augment_str in augmentations:
            # trial without training using the target incubator
            train_without_target = " + ".join(list(train_subset))
            test_target = f""
            trial_without_target = (
                " + ".join()
            )
            
            # trial with training using the target incubator


    # generate reports
    p = PerfData()


def icse_25_breakdown(params_dict: dict[str, Any], args_dict: dict[str, Any]) -> None:
    """
        Temporary utility for the ICSE '25 Paper table generation; saves all 
        results in a separate ICSE db
    
        @param params_dict: centralized parameter structure
        @param args_dict: arguments passed in; should include some specification 
                          of the incubator needed (e.g. "apache").
    """
    
    # trials to do
    options_structs = [
        "cbn"
        # "cbna",
        # "cbnd",
    ]
    trial_structs = [
        "A{opt}^ --> A{t_opt}^^",
        "E{opt} --> A{t_opt}",
        "A{opt}^ + E{opt} --> A{t_opt}^^",
        "A{opt} --> E{t_opt}",
        "E{opt}^ --> E{t_opt}^^",
        "A{opt} + E{opt}^ --> E{t_opt}^^",
        "A{opt} --> G{t_opt}",
        "E{opt} --> G{t_opt}",
        "A{opt} + E{opt} --> G{t_opt}",
        "A{opt}^ + E{opt}^ --> A{t_opt}^^ + E{t_opt}^^",
        "A{opt}^ + E{opt}^ --> A{t_opt}^^ + E{t_opt}^^ + G{t_opt}"
    ]
    
    # setup vars
    num_trials = args_dict.get("trials", 5)
    perf_db_path = "../model-reports/icse-trials/final_icse_perf_db"
    
    # try every set of trials
    for option in options_structs:
        for trial in trial_structs:
            # get args dict, implement new options; note the test options don't 
            # use balancing
            new_args_dict = {
                "strategy": trial.format(opt=option, t_opt=option.replace("b", "")),
                "perf-path": perf_db_path,
                **args_dict
            }
            
            # generate results
            for _ in range(num_trials):
                modeling(params_dict, args_dict=new_args_dict)

    # generate breakdown
    perf_db = PerfData(perf_source=perf_db_path)
    
    ## get only trials with this model architecture
    perf_db.data = perf_db.data[perf_db.data["model_arch"] == args_dict.get("model_arch", "BLSTM")]
    perf_db.summary(metrics=["accuracy", "macro-f1-score"], export=True, verbose=False)
    perf_db.comparison()
    icse_25_experiments()
    
    ## generate overall comparison across all architectures
    regex_matches = [re.sub(r"\{\}", r".*", s) for s in trial_structs]
    perf_db = PerfData(perf_source=perf_db_path)
    best_df = perf_db.best_perfs(transfer_strats=trial_structs)
    

if __name__ == "__main__":
    # forward parameters to main
    params_dict = util.load_params()
    args_dict = util.parse_input(sys.argv)

    trial_type = args_dict.get("trial-type", "regular")
    match trial_type:
        case "regular":
            for i in range(args_dict.get("trials", 1)):
                modeling(params_dict=params_dict, args_dict=args_dict)
        
        case "full":
            full_trials(
                params_dict=params_dict,
                **args_dict
            )

        case "monthly-preds":
            monthly_predictions(
                params_dict=params_dict,
                args_dict=args_dict
            )

        case "breakdown":
            breakdown(
                params_dict=params_dict,
                args_dict=args_dict
            )
        
        case "icse":
            icse_25_breakdown(
                params_dict=params_dict,
                args_dict=args_dict
            )
        
        case _:
            print(":(")

