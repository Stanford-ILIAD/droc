import numpy as np
import os, atexit, argparse
import utils.perception.perception_utils
from utils.modulable_prompt import modulable_prompt
from utils.LLM_utils import query_LLM
from utils.io.io_utils import read_py, USER_LOG, add_to_log
from utils.perception.perception_utils import _parse_pos, _correct_past_detection, _get_grasp_pose, _get_task_detection, _change_reference_frame, initialize_detection, set_policy_and_task, clear_saved_detected_obj
from utils.string_utils import format_code, format_plan, extract_array_from_str, replace_strarray_with_str
from utils.exception_utils import InterruptedByHuman, GraspError, RobotError, PlanningError, WrongDetection

prompt_plan_instance = modulable_prompt('prompts/prompt_plan_backbone.txt', 'prompts/prompt_plan_content.txt')
prompt_codepolicy_instance = modulable_prompt('prompts/prompt_codepolicy_backbone.txt', 'prompts/prompt_codepolicy_content.txt')
prompt_correction_no_history = read_py('prompts/prompt_correction_no_history.txt')
gripper_opened = False

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
    if gripper_opened:
        check_grasp = True
    else:
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
    reference_frame = str(reference_frame)
    if reference_frame != "object" and reference_frame != "absolute":
        reference_frame = 'absolute'
    numpy_array = extract_array_from_str(pos_description)
    if numpy_array is None:
        ret_val, _ = _parse_pos(pos_description, reference_frame)
        return ret_val
    else:
        current_pos, _ = get_current_state()
        if np.linalg.norm(current_pos-numpy_array) < 0.008:
            pos_description = replace_strarray_with_str(pos_description, "current position")
        ret_val, _ = _parse_pos(pos_description, reference_frame)
        return ret_val

def parse_ori(ori_description):
    prompt_parse_ori = read_py('prompts/prompt_parse_ori.py')
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

def reset_to_default_pose():
    return policy.reset()

def delete_last_detection(obj_name, corrected_pos=None):
    _correct_past_detection(obj_name, corrected_pos)

def get_task_pose(task):
    pos, ori =  _get_grasp_pose(task, visualize=True)
    return pos, ori

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
    global gripper_opened
    # Infinite outmost loop
    while True:

        # Receive instructions and generate the initial plan
        li = input('\n\n\n' + "I'm ready to take instruction." + '\n' + 'Input your instruction:')
        prompt_plan = prompt_plan_instance.get_prompt()
        whole_prompt = prompt_plan + '\n' + '\n' + f"Instruction: {li}" + "\n"
        response = query_LLM(whole_prompt, ["Instruction:"], "cache/llm_response_planning_high.pkl")
        code_step = format_plan(response)
        plan_success = False

        # Loop for handling plan failures
        while not plan_success:
            try:
                add_to_log(prompt_plan, also_print=False)
                add_to_log(code_step, also_print=True)
                clear_saved_detected_obj()

                # Loop for executing each sub-step
                for _, step_name in code_step.items():
                    parsed_step_name = step_name.lower()[:-1] if step_name[-1] == '.' else step_name.lower()
                    add_to_log("****Step name: " + parsed_step_name + '****', file_path=USER_LOG)
                    add_to_log(f"I am performing the task: {parsed_step_name}.", also_print=True)
                    gripper_opened = False
                    initialize_detection()

                    # Retrieve relavant task info and generate code
                    prompt_codepolicy = prompt_codepolicy_instance.get_prompt()
                    whole_prompt = prompt_codepolicy + '\n' + '\n' + f"# Task: {step_name}" + "\n" + 'Task-related knowledge: None' + "\n"
                    response = query_LLM(whole_prompt, ["# Task:", "# Outcome:"], "cache/llm_response_planning_low.pkl")
                    _, code_as_policies = format_code(response)

                    # Add info to history temporary file for later interaction history retrieval
                    add_to_log('-'*80 + '\n' + '*whole prompt*' + '\n' + whole_prompt)
                    add_to_log(f"# Task: {step_name}" + "\n" + 'Task-related knowledge: None' + "\n" + '-'*80)

                    # Main loop for code execution, correction and code regeneration
                    while True:
                        try:
                            add_to_log('-'*80 + '\n' + '*original code*' + '\n' + code_as_policies + '\n' + '-'*80, also_print=True)
                            exec(code_as_policies, globals())
                            correction = input(f'Please issue further corrections, or is the task "{parsed_step_name}" done: ')
                            if 'done' in correction.lower() or 'yes' in correction.lower() or correction.lower() == 'y':
                                break
                            else:
                                whole_prompt = prompt_correction_no_history + '\n' + '\n' + f"# Task: {correction}."
                                response = query_LLM(whole_prompt, ["# Task:", "# Outcome:"], "cache/llm_voice_teleop.pkl")
                                _, code_as_policies = format_code(response)
                        except InterruptedByHuman:
                            correction = input(f'Please issue corrections: ')
                            if 'done' in correction.lower() or 'yes' in correction.lower() or correction.lower() == 'y':
                                break
                            else:
                                whole_prompt = prompt_correction_no_history + '\n' + '\n' + f"# Task: {correction}."
                                response = query_LLM(whole_prompt, ["# Task:", "# Outcome:"], "cache/llm_voice_teleop.pkl")
                                _, code_as_policies = format_code(response)
                        except RobotError:
                            correction = input(f'Please issue corrections: ')
                            if 'done' in correction.lower() or 'yes' in correction.lower() or correction.lower() == 'y':
                                break
                            else:
                                whole_prompt = prompt_correction_no_history + '\n' + '\n' + f"# Task: {correction}."
                                response = query_LLM(whole_prompt, ["# Task:", "# Outcome:"], "cache/llm_voice_teleop.pkl")
                                _, code_as_policies = format_code(response)
                        except GraspError:
                            open_gripper(width=1)
                            correction = input(f'Please issue corrections: ')
                            if 'done' in correction.lower() or 'yes' in correction.lower() or correction.lower() == 'y':
                                break
                            else:
                                whole_prompt = prompt_correction_no_history + '\n' + '\n' + f"# Task: {correction}."
                                response = query_LLM(whole_prompt, ["# Task:", "# Outcome:"], "cache/llm_voice_teleop.pkl")
                                _, code_as_policies = format_code(response)
                        except WrongDetection:
                            correction = 'Wrong detection. Please delete past detections.'
                            whole_prompt = prompt_correction_no_history + '\n' + '\n' + f"# Task: {correction}."
                            response = query_LLM(whole_prompt, ["# Task:", "# Outcome:"], "cache/llm_voice_teleop.pkl")
                            _, code_as_policies = format_code(response)
                        except PlanningError as pe:
                            raise PlanningError(pe)
                        # except Exception:
                        #     _break = other_exception_handler(localss, locals, corr_rounds, li, step_name)
                        #     if _break:
                        #         break
                        #     code_as_policies, corr_rounds = failure_reasoning(step_name)
                    add_to_log(f'********************Success! "{parsed_step_name}" is fulfilled !!!!!********************', also_print=True)
                plan_success = True
            except PlanningError:
                prompt_plan = prompt_plan_instance.get_prompt()
                whole_prompt = prompt_plan + '\n' + '\n' + f"Instruction: {li}" + "\n"
                response = query_LLM(whole_prompt, ["Instruction:"], "cache/llm_response_planning_high.pkl")
                code_step = format_plan(response)
        add_to_log("---------------------------Ready to move to next instruction...---------------------------", also_print=True)
        reset_to_default_pose()


if __name__ == '__main__':
    global policy, realrobot
    def delete_tmp():
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