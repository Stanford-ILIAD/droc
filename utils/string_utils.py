import numpy as np
import re
import json
import pickle
from utils.io.io_utils import add_to_log, read_py, HISTORY_TMP_PATH, load_file
from utils.LLM_utils import query_LLM

prompt_replace_description_with_value = read_py('prompts/replace_des_with_val.txt')

def format_code(response):
    if "'''" in response.text:
        code = response.text.split("'''")[1]
        if 'Perception error' in response.text:
            error_type = 'Perception error'
        elif 'Planning error' in response.text:
            error_type = 'Planning error'
        else:
            error_type = None
        return error_type, code
    else:
        code = response.text
        error_type = None
        return error_type, code


def format_plan(response, prefix='Response:'):
    lines = response.text.split("\n")
    non_empty_lines = [line for line in lines if line.strip() != ""]
    result = "\n".join(non_empty_lines)
    if prefix is not None:
        plan_raw = result.split(prefix)[1]
        lines = plan_raw.split("\n")
        non_empty_lines = []
        for line in lines:
            if line != '}':
                non_empty_lines.append(line)
            else:
                non_empty_lines.append(line)
                break
        result = "\n".join(non_empty_lines)
        plan_dict = json.loads(result)
    else:
        plan_dict = json.loads(result)
    return plan_dict


def from_dict_to_str(res_dict):
    ret_str = ''
    for k, v in res_dict.items():
        ret_str += str(k)
        ret_str += '. '
        ret_str += str(v)
        ret_str += ' '
    return ret_str


def get_lines_starting_with(text, prefix, first=True):
    lines = text.splitlines()
    move_to_lines = [line for line in lines if line.strip().startswith(prefix)]
    if first:
        result = move_to_lines[0].strip()
        if result[-1] == ";":
            result = result[:-1]
        return result
    else:
        result = [i.strip() for i in move_to_lines]
        return result


def break_plan_into_steps(code_as_policies):
    if '# ' in code_as_policies:
        lines = code_as_policies.strip().split('\n')
        # Initialize variables to store each step's code
        step_codes = []
        current_step = []
        # Iterate through the lines and identify each step's code
        for line in lines:
            if line.startswith("# "):
                if current_step:
                    step_codes.append('\n'.join(current_step))
                    current_step = []
            current_step.append(line)
        # Append the last step's code
        if current_step:
            step_codes.append('\n'.join(current_step))
    else:
        step_codes = [code_as_policies]
    return step_codes


def extract_array_from_str(text):
    matches = re.findall(r'\[([\d\.\s-]+)\]', text)
    if matches:
        numbers_str = matches[0]
        numbers_list = [float(num) for num in numbers_str.split()]
        num_array = np.array(numbers_list)
        return num_array
    else:
        add_to_log("No matching numbers found.")
        return None


def replace_strarray_with_str(input_string, replace_str):
    new_text = re.sub(r'\[([\d\.\s-]+)\]', replace_str, input_string)
    return new_text


def replace_brackets_in_file(file_path, replacement_string):
    with open(file_path, 'r') as file:
        content = file.read()
    modified_content = content.replace("[]", replacement_string)
    return modified_content


def replace_description_with_value(code, pos_description):
    value_line = code.splitlines()[-1]
    if any(char.isdigit() for char in value_line):
        number_matches = str(int(float(re.findall(r'\d+\.\d+', value_line)[0]) * 100)) + 'cm'
    else:
        number_matches = '10cm'
    whole_prompt = prompt_replace_description_with_value + '\n' + '"' + pos_description + '", "' + number_matches + '":'
    response = query_LLM(whole_prompt, [], "cache/llm_replace_des_with_val.pkl")
    new_description = response.text
    add_to_log('old_pos_des:' + pos_description + ', new_pos_des:' + new_description)
    return new_description


def str_to_dict(string):
    paragraphs = string.split('\n\n')
    paragraph_dict = {index: paragraph.strip() for index, paragraph in enumerate(paragraphs)}
    return paragraph_dict


def dict_to_str(dictionary):
    ret_str = ''
    for k, v in dictionary.items():
        if k != len(dictionary)-1:
            ret_str = ret_str + v + '\n\n'
        else:
            ret_str = ret_str + v
    return ret_str


def replace_code_with_no_description(new_pos_description, corr_rounds):
    history_tmp = load_file(HISTORY_TMP_PATH)
    language_instruction = list(history_tmp.keys())[0]
    step_name = list(history_tmp[language_instruction].keys())[-1]
    correction_code = history_tmp[language_instruction][step_name][f'code response {corr_rounds}']
    lines = correction_code.split('\n')
    for i,line in enumerate(lines):
        if "parse_pos" in line:
            if any(char.isdigit() for char in line):
                pass
            else:
                parts = line.split('"')
                assert len(parts) == 3
                parts[1] = new_pos_description
                lines[i] = '"'.join(parts)
                break
    history_tmp[language_instruction][step_name][f'code response {corr_rounds}'] = '\n'.join(lines)
    pickle.dump(history_tmp, open(HISTORY_TMP_PATH, "wb"))


def format_dictionary(dict_input):
    formatted_string = "{\n"
    last_key = list(dict_input.keys())[-1]
    for key, value in dict_input.items():
        if key != last_key:
            formatted_string += f'  "{key}": "{value}",\n'
        else:
            formatted_string += f'  "{key}": "{value}"\n'
    formatted_string += "}"
    return formatted_string