import os
import string
from datetime import datetime
from random import random


def replace_ambigous_title(title_str: str):
    title_str = title_str.replace('_training', '')
    title_str = title_str.replace('_train', '')
    title_str = title_str.replace('/training', '')
    title_str = title_str.replace('training', '')
    title_str = title_str.replace('train', '')
    title_str = title_str.replace('train_cgp', '')
    title_str = title_str.replace('_cgp', '')
    title_str = title_str.replace('_large', '_lg')
    title_str = title_str.replace('_small', '_sm')
    title_str = title_str.replace('\"', '')
    return title_str


def get_dataset_identifier(dataset_path):
    # Split the path by the OS-specific separator
    parts = dataset_path.split(os.sep)
    del parts[-1]
    del parts[-1]

    parts[-1] = replace_ambigous_title(parts[-1])

    # Ensure no part is an empty string
    parts = [part for part in parts if part]

    # Determine how to construct the identifier
    if len(parts) > 3:
        # Combine the third and second-to-last parts for a more descriptive name
        identifier = f"{parts[-3]}_{parts[-1]}"
    elif len(parts) > 2:
        # Fall back to combining the second-to-last and last parts
        identifier = f"{parts[-2]}_{parts[-1]}"
    elif parts[-1] != "D:" and parts[-1] != "C:":
        identifier = parts[-1]
    else:
        # avoid empty identifier
        identifier = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

    return identifier


def log_message(function: str, dataset: str, msg: str):
    logdir = os.path.join(os.getcwd(), "log")
    current_date = datetime.now().strftime("%Y%m%d")
    curlogfile = os.path.join(logdir, f"log{current_date}.txt")

    log_entry = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ", " \
                + msg + " in " \
                + function + " for dataset " \
                + dataset + "\n"

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    with open(curlogfile, "a") as logfile:
        logfile.write(log_entry)
