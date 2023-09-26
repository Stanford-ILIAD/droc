import os
import pickle
from collections import defaultdict
import numpy as np

HISTORY_TMP_PATH = "cache/history_tmp.pkl"
USER_LOG = 'user_log.txt'


def read_py(path_to_py):
    f = open(path_to_py, "r")
    return f.read()


def open_file(filename, mode):
    try:
        file = open(filename, mode)
    except FileNotFoundError:
        create_folder(os.path.dirname(filename))
        file = open(filename, 'w')
        file.close()
        file = open(filename, mode)
    return file


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Folder created:", folder_path)
    else:
        print("Folder already exists:", folder_path)


def load_file(cache_file):
    if not os.path.exists(os.path.dirname(cache_file)):
        # add_to_log(f"Creating directory for {cache_file}")
        os.makedirs(os.path.dirname(cache_file))
    cache: defaultdict[str, dict] = defaultdict(dict)
    if os.path.exists(cache_file):
        # add_to_log(f"loading cache from {cache_file}")
        cache = pickle.load(open(cache_file, "rb"))
    return cache


def add_to_log(*text, file_path="log.txt", also_print=False):
    if len(text) == 1:
        text = text[0]
    text = str(text)
    with open(file_path, "a+") as file:
        file.seek(0)
        existing_content = file.read()
        file.seek(0, 2)
        if existing_content:
            file.write("\n" + text)
        else:
            file.write(text)
    if also_print:
        print(text)


def delete_file(file_path):
    try:
        # Use os.remove() to delete the file
        os.remove(file_path)
        add_to_log(f"File '{file_path}' has been deleted successfully.")
    except FileNotFoundError:
        add_to_log(f"File '{file_path}' not found.")
    except PermissionError:
        add_to_log(f"Permission denied to delete '{file_path}'.")
    except Exception as e:
        add_to_log(f"An error occurred: {e}")


def save_information_perm(var_dict):
    history_tmp = load_file(HISTORY_TMP_PATH)
    language_instruction = list(history_tmp.keys())[0]
    step_name = list(history_tmp[language_instruction].keys())[-1]
    try:
        previous_tasks_dict = pickle.load(open('cache/task_history/task_history.pkl','rb'))
        newtask_idx = int(list(previous_tasks_dict.keys())[-1]) + 1
    except:
        previous_tasks_dict = {}
        newtask_idx = 1
    previous_tasks_dict[newtask_idx] = step_name
    pickle.dump(previous_tasks_dict, open('cache/task_history/task_history.pkl', "wb"))
    filename = f'./cache/task_history/{newtask_idx}.pkl'
    for var_name, var_value in var_dict.items():
        var_name = var_name.replace(" ", "_")
        save_info(filename, var_name, var_value)


def save_information(var_name, var_value, inter_round):
    filename = 'cache/tmp_info.pkl'
    if type(var_value) != np.ndarray and type(var_value) != int and type(var_value) != str:
        return
    else:
        if type(var_value) == str:
            if any(char.isdigit() for char in var_value):
                save_info(filename, var_name, var_value)
            else:
                return
        save_info(filename, var_name, var_value)


def save_info(filename, info_name, info_value):
    try:
        open(filename)
        # TODO: rb or wb?
        info_dict: dict = pickle.load(open(filename, "rb"))
        info = {info_name: info_value}
        info_dict.update(info)
        pickle.dump(info_dict, open(filename, "wb"))
    except:
        info = {info_name: info_value}
        pickle.dump(info, open(filename, "wb"))


def get_previous_tasks():
    from utils.string_utils import from_dict_to_str
    if not os.path.exists('cache/task_history'):
        os.makedirs('cache/task_history')
    if not os.path.exists('cache/task_history/task_history.pkl'):
        initial_data = {}
        with open('cache/task_history/task_history.pkl', 'wb') as file:
            pickle.dump(initial_data, file)
    try:
        previous_tasks_dict = pickle.load(open('cache/task_history/task_history.pkl','rb'))
    except:
        initial_data = {}
        with open('cache/task_history/task_history.pkl', 'wb') as file:
            pickle.dump(initial_data, file)
        previous_tasks_dict = pickle.load(open('cache/task_history/task_history.pkl','rb'))
    previous_tasks = from_dict_to_str(previous_tasks_dict)
    return previous_tasks

def save_plan_info(constraint, vis_feature=None):
    try:
        previous_dict = pickle.load(open('cache/task_history/constraints.pkl','rb'))
        newtask_idx = int(list(previous_dict.keys())[-1]) + 1
    except:
        previous_dict = {}
        newtask_idx = 1
    if vis_feature is not None:
        previous_dict[newtask_idx] = (constraint, vis_feature)
    else:
        previous_dict[newtask_idx] = constraint
    pickle.dump(previous_dict, open('cache/task_history/constraints.pkl', "wb"))