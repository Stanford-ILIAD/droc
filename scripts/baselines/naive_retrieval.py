import numpy as np
import os, pickle, json, atexit, re, argparse
from collections import defaultdict
from utils.modulable_prompt import modulable_prompt
from utils.LLM_utils import query_LLM
from utils.io.io_utils import read_py, delete_file, load_file, HISTORY_TMP_PATH, USER_LOG, add_to_log, save_information_perm, save_information, get_previous_tasks
from utils.perception.perception_utils import _parse_pos, _correct_past_detection, _get_grasp_pose, get_detected_feature, compare_feature, _get_task_detection, _change_reference_frame, initialize_detection, _update_object_state, get_initial_state, set_policy_and_task, clear_saved_detected_obj
from utils.transformation_utils import calculate_relative_pose, get_real_r, quaternion_to_rotation_matrix, r_to_quat
from utils.string_utils import format_code, format_plan, get_lines_starting_with, extract_array_from_str, replace_strarray_with_str, replace_brackets_in_file, replace_description_with_value, replace_code_with_no_description, format_dictionary
from utils.exception_utils import InterruptedByHuman, GraspError, RobotError, PlanningError, WrongDetection, interruption_handler, robot_error_handler, grasp_error_handler, detection_error_handler, no_exception_handler, other_exception_handler

from sentence_transformers import SentenceTransformer  # pip install this
from sentence_transformers import util as st_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
masked_LM = SentenceTransformer('all-MiniLM-L6-v2').to(device)

def get_emb(text):
  return masked_LM.encode(text, convert_to_tensor=True, device=device)

def match(emb1, emb2):
  scores = np.abs(st_utils.pytorch_cos_sim(emb1, emb2).cpu().numpy().squeeze())
  return scores

prompt_plan_instance = modulable_prompt('prompts/newnew_prompts/prompt_plan_backbone.txt', 'prompts/newnew_prompts/prompt_plan_content.txt')
prompt_codepolicy_instance = modulable_prompt('prompts/newnew_prompts/prompt_codepolicy_backbone.txt', 'prompts/newnew_prompts/prompt_codepolicy_content.txt')
rel_pos, rel_ori, gripper_opened = None, None, False

# ------------------------------------------ Primitives ------------------------------------------

def get_current_state():
    if realrobot:
        pose = policy.robot_env.robot.get_ee_pose()
        ee_pos = pose[0].numpy()
        ee_ori = pose[1].numpy()
        return (ee_pos, ee_ori)
    else:
        return (np.array((1.,0.,0.)), np.array((1.,0.,0.,0.)))

def get_horizontal_ori():
    return policy.get_horizontal_ori()

def get_vertical_ori():
    return policy.get_vertical_ori()

def open_gripper(width=1):
    global gripper_opened
    if type(width) is not int and type(width) is not float:
        width = 1
    gripper_opened = True
    policy.open_gripper(width)

def close_gripper(width=None):
    global gripper_opened
    # if gripper_opened:
    #     check_grasp = True
    # else:
    #     check_grasp = False
    check_grasp = False
    if width is None:
        policy.close_gripper(check_grasp=check_grasp)
    else:
        if type(width) is not int and type(width) is not float:
            width = 1
        policy.open_gripper(width)

def get_ori(degrees, axis):
    return policy.rotate_gripper(degrees, axis)

def move_gripper_to_pose(pos, rot):
    return policy.move_to_pos(pos, rot)

def parse_pos(pos_description, reference_frame='object'):
    global corr_rounds
    reference_frame = str(reference_frame)
    if '20cm' in pos_description:
        pos_description = pos_description.replace('20cm', '23cm')
    if reference_frame != "object" and reference_frame != "absolute":
        reference_frame = 'absolute'
    numpy_array = extract_array_from_str(pos_description)
    if numpy_array is None:
        ret_val, code_as_policies = _parse_pos(pos_description, reference_frame)
        if not any(char.isdigit() for char in pos_description):
            new_pos_description = replace_description_with_value(code_as_policies, pos_description)
            replace_code_with_no_description(new_pos_description, corr_rounds)
        return ret_val
    else:
        current_pos, _ = get_current_state()
        replace = False
        if np.linalg.norm(current_pos-numpy_array) < 0.008:
            pos_description = replace_strarray_with_str(pos_description, "current position")
            replace = True
        ret_val, code_as_policies = _parse_pos(pos_description, reference_frame)
        if replace:
            if not any(char.isdigit() for char in pos_description):
                new_pos_description = replace_description_with_value(code_as_policies, pos_description)
                replace_code_with_no_description(new_pos_description, corr_rounds)
        else:
            if not any(char.isdigit() for char in pos_description.split('[')[0]):
                new_pos_description = replace_description_with_value(code_as_policies, pos_description)
                replace_code_with_no_description(new_pos_description, corr_rounds)
        return ret_val

def parse_ori(ori_description):
    prompt_parse_ori = read_py('prompts/newnew_prompts/prompt_parse_ori.py')
    ori_description = ori_description.split(' relative')[0]
    whole_prompt = prompt_parse_ori + '\n' + '\n' + f"# Query: {ori_description}" + "\n"
    response = query_LLM(whole_prompt, ["# Query:"], "cache/llm_response_parse_ori.pkl")
    code_as_policies = response.text
    add_to_log('-'*80)
    add_to_log('*at parse_ori*')
    add_to_log(code_as_policies)
    add_to_log('-'*80)
    localss = {}
    exec(code_as_policies, globals(), localss)
    return localss['ret_val']

def reset_to_default_pose(reset_gripper=False):
    return policy.reset(reset_gripper=reset_gripper)

def delete_last_detection(obj_name, corrected_pos=None):
    _correct_past_detection(obj_name, corrected_pos)

def get_task_pose(task):
    global rel_pos, rel_ori
    if rel_pos is None:
        pos, ori =  _get_grasp_pose(task, visualize=True)
        return pos, ori
    else:
        pos, ori =  _get_grasp_pose(task, visualize=False)
        print(pos)
        print(ori)
        assert len(ori) == 4
        ori_r = quaternion_to_rotation_matrix(ori)
        real_r = get_real_r(rel_ori, ori_r)
        real_ori = r_to_quat(real_r)
        real_pos = pos + rel_pos
        rel_pos, rel_ori = None, None
        print(real_pos)
        print(real_ori)
        return real_pos, real_ori

def get_task_detection(task):
    return _get_task_detection(task)

def change_reference_frame(wrong_direction, correct_direction):
    _change_reference_frame(wrong_direction, correct_direction)

def execute_post_action(step_name):
    if "open" in step_name.lower() or "close" in step_name.lower():
        current_pos, current_ori = get_current_state()
        target_pos = parse_pos(f"a point 8cm back to {current_pos}.", reference_frame='object')
        move_gripper_to_pose(target_pos, current_ori)
        reset_to_default_pose()
    if "put" in step_name.lower():
        current_pos, current_ori = get_current_state()
        target_pos = parse_pos(f"a point 60cm above {current_pos}.", reference_frame='absolute')
        move_gripper_to_pose(target_pos, current_ori)
        reset_to_default_pose()
    if "pick" in step_name.lower():
        current_pos, current_ori = get_current_state()
        target_pos = parse_pos(f"a point 60cm above {current_pos}.", reference_frame='absolute')
        move_gripper_to_pose(target_pos, current_ori)

# ------------------------------------------ Main function ------------------------------------------

def main():
    global corr_rounds, use_interrupted_code, gripper_opened

    # Infinite outmost loop
    while True:

        # Receive instructions and generate the initial plan
        li = input('\n\n\n' + "I'm ready to take instruction." + '\n' + 'Input your instruction:')
        obj_state = get_initial_state()
        prompt_plan_instance.set_object_state(obj_state)
        prompt_plan = prompt_plan_instance.get_prompt()
        whole_prompt = prompt_plan + '\n' + '\n' + f"Instruction: {li}" + "\n"
        response = query_LLM(whole_prompt, ["Instruction:"], "cache/llm_response_planning_high.pkl")
        code_step = format_plan(response)
        plan_success = False

        # Loop for handling plan failures
        while not plan_success:
            try:
                add_to_log(prompt_plan_instance.get_prompt(), also_print=False)
                add_to_log(code_step, also_print=False)
                tmp = load_file(HISTORY_TMP_PATH)
                tmp[li] = {}
                clear_saved_detected_obj()

                # Loop for executing each sub-step
                for _, step_name in code_step.items():
                    parsed_step_name = step_name.lower()[:-1] if step_name[-1] == '.' else step_name.lower()
                    add_to_log("****Step name: " + parsed_step_name + '****', file_path=USER_LOG)
                    add_to_log(f"I am performing the task: {parsed_step_name}.", also_print=True)
                    use_interrupted_code, gripper_opened, corr_rounds = False, False, 0
                    locals = defaultdict(dict)
                    initialize_detection()

                    # Retrieve relavant task info and generate code
                    task_related_knowledge, localss = retrieve_task_info(step_name)
                    prompt_codepolicy = prompt_codepolicy_instance.get_prompt()
                    whole_prompt = prompt_codepolicy + '\n' + '\n' + f"# Task: {step_name}" + "\n" + task_related_knowledge + "\n"
                    response = query_LLM(whole_prompt, ["# Task:", "# Outcome:"], "cache/llm_response_planning_low.pkl")
                    _, code_as_policies = format_code(response)

                    # Add info to history temporary file for later interaction history retrieval
                    tmp[li][step_name] = {}
                    tmp[li][step_name]['code response 0'] = code_as_policies
                    add_to_log('-'*80 + '\n' + '*whole prompt*' + '\n' + whole_prompt)
                    add_to_log(f"# Task: {step_name}" + "\n" + task_related_knowledge + "\n" + '-'*80)
                    pickle.dump(tmp, open(HISTORY_TMP_PATH, "wb"))

                    # Main loop for code execution, correction and code regeneration
                    while True:
                        try:
                            add_to_log('-'*80 + '\n' + '*original code*' + '\n' + code_as_policies + '\n' + '-'*80, also_print=True)
                            exec(code_as_policies, globals(), localss)
                            _break, corr_rounds, code_as_policies, use_interrupted_code = no_exception_handler(localss, locals, corr_rounds, li, step_name, failure_reasoning, use_interrupted_code, code_as_policies)
                            if _break:
                                break
                        except InterruptedByHuman:
                            _break = interruption_handler(localss, locals, corr_rounds, li, step_name, code_as_policies)
                            if _break:
                                break
                            code_as_policies, corr_rounds = failure_reasoning(step_name, li, corr_rounds)
                        except RobotError:
                            _break = robot_error_handler(localss, locals, corr_rounds, li, step_name, code_as_policies)
                            if _break:
                                break
                            code_as_policies, corr_rounds = failure_reasoning(step_name, li, corr_rounds)
                        except GraspError:
                            open_gripper(width=1)
                            _break = grasp_error_handler(localss, locals, corr_rounds, li, step_name, code_as_policies)
                            if _break:
                                break
                            code_as_policies, corr_rounds = failure_reasoning(step_name, li, corr_rounds)
                        except WrongDetection:
                            code_as_policies = detection_error_handler(localss, locals, corr_rounds, li, step_name, code_as_policies)
                        except PlanningError as pe:
                            raise PlanningError(pe)
                        # except Exception:
                        #     _break = other_exception_handler(localss, locals, corr_rounds, li, step_name)
                        #     if _break:
                        #         break
                        #     code_as_policies, corr_rounds = failure_reasoning(step_name)
                    add_to_log(f'********************Success! "{parsed_step_name}" is fulfilled !!!!!********************', also_print=True)
                    print('# of corrections: ',corr_rounds)
                    save_task_info(locals, step_name, corr_rounds, li)
                    obj_state = update_object_state(obj_state, parsed_step_name)
                    print(obj_state)
                plan_success = True
                delete_file(HISTORY_TMP_PATH)
            except PlanningError:
                code_step, obj_state = replan(corr_rounds, li, step_name, code_step, obj_state)
        add_to_log("---------------------------Ready to move to next instruction...---------------------------", also_print=True)

# ------------------------------------------ LLM reasoning functions ------------------------------------------

def is_plan_error(step_name, li, corr_rounds):
    prompt_is_planning_error = read_py('prompts/newnew_prompts/prompt_is_planning_error.txt')
    tmp = load_file(HISTORY_TMP_PATH)
    correction = tmp[li][step_name][f'Correction {corr_rounds}']
    prompt = prompt_is_planning_error + '\n\n' + f'Task: {step_name}' + '\n' +f'Correction: {correction}' + '\n' + 'Output: '
    response = query_LLM(prompt, [], 'cache/llm_is_planning_error.pkl')
    add_to_log(prompt)
    add_to_log(response.text)
    if 'yes' in response.text.lower():
        is_planning_error = True
        really_error = input('Is this really a planning error? (y/n)')
        if 'y' in really_error:
            pass
        else:
            is_planning_error = False
    elif 'no' in response.text.lower():
        is_planning_error = False
    return is_planning_error

def retrieve_task_info(step_name):
    global rel_pos, rel_ori

    previous_tasks = get_previous_tasks()

    if previous_tasks == '':
        print('Not retrieving anything...')
        return "# Task-related knowledge: None.", {}
    else:
        with open("prompts/newnew_prompts/prompt_get_object_feature.txt", "r") as template_file:
            template_content = template_file.read()
        filled_content = template_content.replace("[]", previous_tasks, 1)
        prompt = filled_content.replace("[]", step_name, 1)
        response = query_LLM(prompt, [], 'cache/llm_get_object_feature.pkl')
        add_to_log("previous tasks:"+previous_tasks, also_print=False)
        add_to_log(prompt)
        add_to_log(response.text, also_print=False)
        response_dict = json.loads(response.text)
        if response_dict["1"] == "Yes":
            info_dict : dict = pickle.load(open(f'cache/task_history/{int(response_dict["2"])}.pkl', "rb"))
            # image_feature = info_dict['image_feature']
            image_feature = info_dict['dino_image_feature']
            compare_feature(image_feature, step_name)

        with open("prompts/newnew_prompts/prompt_check_done.txt", "r") as template_file:
            template_content = template_file.read()
        filled_content = template_content.replace("[]", previous_tasks, 1)
        prompt = filled_content.replace("[]", step_name, 1)
        response = query_LLM(prompt, [], 'cache/llm_check_done.pkl')
        add_to_log(prompt,also_print=True)
        add_to_log(response.text, also_print=True)
        response_dict = json.loads(response.text)

        if response_dict["1"] == 'Yes' and response_dict["2"] == 'Yes':
            # NOTE: If we want to use dino feature, make sure get_task_detection return .dino_img_feature instead of .img_feature
            task_visual_feature, already_have = get_task_detection(step_name)
            if already_have:
                compare_feature(task_visual_feature, step_name, threshold=1, visualize=True)
            max_diff = 10000
            for task_index in response_dict["3"]:
                assert type(task_index) == int
                info_dict : dict = pickle.load(open(f'cache/task_history/{task_index}.pkl', "rb"))
                # NOTE: If we want to use dino feature, change 'image_feature' to 'dino_image_feature'
                # vis_feature = info_dict['image_feature']
                vis_feature = info_dict['dino_image_feature']
                diff = np.linalg.norm(vis_feature-task_visual_feature)
                if diff<=max_diff:
                    max_diff = diff
                    ind = task_index
            print(f'Retrieving from task {ind}')
            info_dict : dict = pickle.load(open(f'cache/task_history/{ind}.pkl', "rb"))
            rel_pos = info_dict['relative_pose'][0]
            rel_ori = info_dict['relative_pose'][1]
            other_keys = [key for key in info_dict.keys() if key != 'relative_pose' and key != 'image_feature' and key != 'dino_image_feature' and key != 'code']
            ret_str = "# Task-related knowledge: "
            locals = {}
            for i in range(len(other_keys)):
                k = other_keys[i].replace(" ", "_")
                ret_str += k
                locals[k] = info_dict[other_keys[i]]
                if i != len(other_keys)-1:
                    ret_str += ', '
            if ret_str == "# Task-related knowledge: ":
                ret_str += 'None.'
            print(f'Retrieving these knowledge: {ret_str}')
            if 'code' in info_dict.keys():
                code = info_dict['code']
                update_code = '# Task: ' + step_name + '\n' + ret_str + '\n' + code
                prompt_codepolicy_instance.update_content(update_code)
            return ret_str, locals
        else:
            print('Not retrieving anything...')
            return "# Task-related knowledge: None.", {}

def save_task_info(locals, step_name, corr_rounds, li):
    prompt_saveinfo = read_py('prompts/newnew_prompts/prompt_saveinfo.txt')
    prompt_get_task_feature_file = 'prompts/newnew_prompts/prompt_task_features.txt'
    if "close" in step_name.lower() or 'put' in step_name.lower():
        return
    parsed_step_name = step_name.lower()[:-1] if step_name[-1] == '.' else step_name.lower()
    with open(prompt_get_task_feature_file, "r") as template_file:
        template_content = template_file.read()
    prompt_get_task = template_content.replace("{}", parsed_step_name, 1)
    response = query_LLM(prompt_get_task, ['The information'], "cache/llm_get_task_feature.pkl")
    _, task_related_info = format_code(response)
    tmp = load_file(HISTORY_TMP_PATH)
    history_prompt = "Interaction history:" + '\n' + "'''" + '\n'
    history_prompt = history_prompt + f'Task: {step_name}' + '\n'
    history_prompt = history_prompt + f'Task-related knowledge: {task_related_info}' + '\n' + '\n'
    history_prompt = history_prompt + 'Response 1:' + '\n' + "'''" + '\n' + tmp[li][step_name]['code response 0'] + '\n' + "'''"
    history_prompt = history_prompt + '\n' + 'Outcome: ' + tmp[li][step_name]['Outcome 0']
    history_prompt = history_prompt + '\n' + 'Human feedback: ' + tmp[li][step_name]['Correction 0']
    for i in range(1, corr_rounds+1):
        history_prompt = history_prompt + '\n' + '\n'
        history_prompt = history_prompt + f'Response to feedback {i+1}:' + '\n' + "'''" + '\n' + tmp[li][step_name][f'code response {i}'] + '\n' + "'''"
        history_prompt = history_prompt + '\n' + 'Outcome: ' + tmp[li][step_name][f'Outcome {i}']
        history_prompt = history_prompt + '\n' + 'Human feedback: ' + tmp[li][step_name][f'Correction {i}']
    history_prompt = prompt_saveinfo + '\n' + '\n' + history_prompt + '\n' + "'''" + '\n' + '\n' + 'Your response:' + '\n'
    response = query_LLM(history_prompt, ['Your response:', 'Interaction'], "cache/llm_response_save_info.pkl")
    codes = get_lines_starting_with(response.text, 'save_information', first=False)
    _, code_as_policies = format_code(response)
    if code_as_policies == response.text:
        has_code = False
    else:
        has_code = True
        add_to_log(code_as_policies)
    add_to_log('-'*80)
    add_to_log('*at save_info*')
    add_to_log(history_prompt, also_print=True)
    add_to_log(response.text, also_print=True)
    add_to_log(locals)
    add_to_log('-'*80)
    localss = locals.copy()
    
    for code in codes:
        try:
            int(code[-3])
            locals = localss[int(code[-3:-1])-1]
        except:
            locals = localss[int(code[-2])-1]
        exec(code, globals(), locals)
    try:
        info_dict : dict = pickle.load(open('cache/tmp_info.pkl', "rb"))
        if has_code:
            info_dict['code'] = code_as_policies
        keys_with_ori = [key for key in info_dict.keys() if 'ori' in key]
        real_ori = info_dict[keys_with_ori[0]]
        keys_with_pos = [key for key in info_dict.keys() if 'pos' in key]
        real_pos = info_dict[keys_with_pos[0]]
        image_feature, dino_image_feature, detected_pose = get_detected_feature()
        d_pos, d_ori = detected_pose
        assert len(d_ori) == 4
        assert len(real_ori) == 4
        relative_pose = calculate_relative_pose((real_pos,real_ori), (d_pos, d_ori), is_quat=True)
        info_to_save = ({'relative_pose': relative_pose,'image_feature': image_feature, 'dino_image_feature': dino_image_feature})
        other_keys = [key for key in info_dict.keys() if 'ori' not in key and 'pos' not in key]
        for key in other_keys:
            info_to_save[key] = info_dict[key]
        save_information_perm(info_to_save)
        task_related_info = re.sub(r'\b\w+(_pos|_ori)\b', '', task_related_info)
        task_related_info = re.sub(r',+', ',', task_related_info)
        task_related_info = task_related_info.strip(', ')
        # update_code = '# Task: ' + step_name + '\n' + f'Task-related knowledge: {task_related_info}' + code_as_policies
        # prompt_codepolicy_instance.update_content(update_code)
        # execute_post_action(step_name)
        delete_file('cache/tmp_info.pkl')
    except:
        pass

def replan(corr_rounds, li, step_name, original_plan, object_state):
    prompt_plan_correction = read_py('prompts/newnew_prompts/prompt_plan_correction.txt')
    # prompt_replan = 'prompts/newnew_prompts/prompt_replan.txt'
    parsed_step_name = step_name.lower()[:-1] if step_name[-1] == '.' else step_name.lower()
    tmp = load_file(HISTORY_TMP_PATH)
    correction = tmp[li][step_name][f'Correction {corr_rounds}']
    whole_prompt = prompt_plan_correction + '\n' + '\n' + f"Task: {li}" + "\n" + f"Plan:\n{format_dictionary(original_plan)}" + '\n' + f'Outcome: Interrupted by human at the step "{parsed_step_name}.'
    whole_prompt = whole_prompt + '\n' + 'Correction: ' + correction + '.' + '\n' + f'Object state: {object_state}'
    response = query_LLM(whole_prompt, ["Task:"], "cache/llm_planning_correction.pkl")
    task_constraint = get_lines_starting_with(response.text, 'Task constraint:', first=True).split('Task constraint: ')[1]
    robot_constraint = get_lines_starting_with(response.text, 'Robot constraint:', first=True).split('Robot constraint: ')[1]
    updated_object_state = get_lines_starting_with(response.text, 'Updated object state:', first=True).split('Updated object state: ')[1]
    code_step = format_plan(response, "Modified original plan:")
    add_to_log(code_step, also_print=False)
    print(response.text)
    if 'none' in task_constraint.lower():
        pass
    else:
        prompt_plan_instance.add_constraints(task_constraint)
    if 'none' in robot_constraint.lower():
        pass
    else:
        prompt_plan_instance.add_constraints(robot_constraint)
    add_to_log(task_constraint, also_print=False)
    add_to_log(robot_constraint, also_print=False)
    update_plan = f'Instruction: {li}' + '\n' + 'Response:' + '\n' + format_dictionary(code_step)
    prompt_plan_instance.update_content(update_plan)
    prompt_plan_instance.set_object_state(updated_object_state)
    # with open(prompt_replan, "r") as template_file:
    #     template_content = template_file.read()
    # values = {"original_plan": format_dictionary(original_plan), "new_plan": format_dictionary(code_step), "step_when_interrupted": parsed_step_name}
    # prompt = template_content.format(**values)
    # response = query_LLM(prompt, [], "cache/llm_replan.pkl")
    # re_plan = format_plan(response, None)
    re_plan = format_plan(response, "Replan:")
    add_to_log(re_plan, also_print=False)
    delete_file(HISTORY_TMP_PATH)
    return re_plan, updated_object_state

def update_object_state(obj_state, task_name):
    new_obj_state = _update_object_state(obj_state, task_name)
    prompt_plan_instance.set_object_state(new_obj_state)
    return new_obj_state

def failure_reasoning(step_name, li, corr_rounds):
    tmp = load_file(HISTORY_TMP_PATH)
    _is_plan_error = is_plan_error(step_name, li, corr_rounds)
    if _is_plan_error:
        add_to_log("**Error Type: Planning error", also_print=True)
        raise PlanningError('')
    history_type, prompt_or_code = retrieve_interaction_history(step_name, li, corr_rounds)
    if history_type == 1:
        response = query_LLM(prompt_or_code, ["Outcome:", "# Outcome:", "# Task:"], 'cache/llm_response_correction.pkl')
        _, correction_code = format_code(response)
        error_type = "Position Inaccuracy"
    elif history_type == 2:
        correction_code = prompt_or_code
        error_type = 'None'
    elif history_type == 3:
        response = query_LLM(prompt_or_code, ["Outcome:", "# Outcome:", "# Task:"], 'cache/llm_response_correction.pkl')
        _, correction_code = format_code(response)
        error_type = "Position Inaccuracy"
    elif history_type == 4:
        response = query_LLM(prompt_or_code, ["Outcome:", "# Outcome:", "# Task:"], 'cache/llm_response_correction.pkl')
        _, correction_code = format_code(response)
        error_type = "Position Inaccuracy"
    add_to_log("**Error Type: " + error_type, also_print=True)
    add_to_log("**Response Code: " + correction_code)
    tmp = load_file(HISTORY_TMP_PATH)
    tmp[li][step_name][f'error type {corr_rounds}'] = error_type
    corr_rounds += 1
    tmp[li][step_name][f'code response {corr_rounds}'] = correction_code
    pickle.dump(tmp, open(HISTORY_TMP_PATH, "wb"))
    return correction_code, corr_rounds

def retrieve_interaction_history(step_name, li, corr_rounds):
    global use_interrupted_code
    prompt_correction_file = 'prompts/newnew_prompts/prompt_correction.txt'
    prompt_correction_no_history = read_py('prompts/newnew_prompts/prompt_correction_no_history.txt')
    prompt_retrieve = read_py('prompts/newnew_prompts/prompt_retrieve.txt')
    tmp = load_file(HISTORY_TMP_PATH)
    correction = tmp[li][step_name][f'Correction {corr_rounds}']
    retrieve_prompt = prompt_retrieve + '\n' + f'"{correction}":'
    response = query_LLM(retrieve_prompt, ['"'], 'cache/llm_retrieve.pkl')
    if len(response.text) >= 4:
        answer = response.text[:4]
    else:
        answer = response.text
    if 'a' in answer:
        history_type = 1
    elif 'b' in answer:
        history_type = 2
    elif 'c' in answer:
        history_type = 3
    elif 'd' in answer:
        history_type = 4
    add_to_log(history_type)
    assert history_type in [1,2,3,4]
    prompt_correction = replace_brackets_in_file(prompt_correction_file, step_name)
    if history_type == 1:
        add_to_log("*** retrieve answer: Last Round Code ***")
        response_code = tmp[li][step_name][f'code response {corr_rounds}']
        add_to_log(response_code)
        correction_prompt = prompt_correction + '\n' + '\n' + "Last round code:" + '\n' + "'''" + '\n' + response_code + '\n' + "'''"
        correction_prompt = correction_prompt + '\n' + 'Outcome: ' + tmp[li][step_name][f'Outcome {corr_rounds}']
        whole_prompt = correction_prompt + '\n' + 'Human feedback: ' + correction
    elif history_type == 3:
        a = input('Only restarting from beginning can solve this error. Should I restart? (y/n)')
        if 'y' in a.lower():
            raise KeyboardInterrupt
        else:
            add_to_log("*** retrieve answer: Last Round Code ***")
            response_code = tmp[li][step_name][f'code response {corr_rounds}']
            correction_prompt = prompt_correction + '\n' + '\n' + "Last round code:" + '\n' + "'''" + '\n' + response_code + '\n' + "'''"
            correction_prompt = correction_prompt + '\n' + 'Outcome: ' + tmp[li][step_name][f'Outcome {corr_rounds}']
            whole_prompt = correction_prompt + '\n' + 'Human feedback: ' + correction
    elif history_type == 2:
        add_to_log("*** retrieve answer: Code When Interrupted ***")
        whole_prompt = tmp[li][step_name]['code when interrupted']
        use_interrupted_code = True
    elif history_type == 4:
        add_to_log("*** retrieve answer: No ***")
        whole_prompt = prompt_correction_no_history + '\n' + '\n' + f"# Task: {correction}" + "\n"       
    else:
        raise NotImplementedError
    return history_type, whole_prompt

if __name__ == '__main__':
    global policy, realrobot
    def delete_tmp():
        delete_file(HISTORY_TMP_PATH)
        with open('new_codeprompt.txt', 'w') as file:
            file.write(prompt_codepolicy_instance.get_prompt())
        with open('new_planprompt.txt', 'w') as file:
            file.write(prompt_plan_instance.get_prompt())
    atexit.register(delete_tmp)
    user_id = 'lihan'
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='drawer')
    parser.add_argument("--realrobot", type=bool, default=False)
    args = parser.parse_args()
    task = args.task
    realrobot = args.realrobot
    set_policy_and_task(realrobot, task)
    if realrobot:
        from utils.robot.robot_policy import KptPrimitivePolicy
        from utils.vision.shared_devices import multi_cam
        policy = KptPrimitivePolicy(multi_cam)
    else:
        from utils.robot.dummy_policy import DummyPolicy
        policy = DummyPolicy()
    if os.path.exists(f"log/{task}/{user_id}/"):
        pass
    else:
        os.makedirs(f"log/{task}/{user_id}/")
    main()