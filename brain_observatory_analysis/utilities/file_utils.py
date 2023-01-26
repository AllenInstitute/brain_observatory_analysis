import os
from pathlib import Path


def find_files_with_string(base_dir, sub_folders=True, string_list=["average"]):
    """Find all files with string in name in all subfolders of base_dir.

    Parameters:
    -----------
    base_dir : str
            path to base directory
    string : str
            string to search for in file names

    Returns:
    --------
    all_files : list of str
    """
    all_files = []
    # search all subfolders for files with string in name
    if sub_folders:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if all(x in file for x in string_list):
                    all_files.append(Path(os.path.join(root, file)))
    else:
        for file in os.listdir(base_dir):
            if all(x in file for x in string_list):
                all_files.append(Path(os.path.join(base_dir, file)))
    return all_files

    # get all files with string in name
    # all_files = [f for f in os.listdir(base_dir) if string in f]


def find_files_with_string_OLD(base_dir, string="average"):
    """Find all files with string in name in all subfolders of base_dir.

    Parameters:
    -----------
    base_dir : str
            path to base directory
    string : str
            string to search for in file names

    Returns:
    --------
    all_files : list of str
    """
    all_files = []
    # search all subfolders for files with string in name
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if string in file:
                all_files.append(Path(os.path.join(root, file)))
    return all_files


def windows_path(path):
    wp = "\\" + str(path).replace("//", "\\").replace("/", "\\")
    return wp
