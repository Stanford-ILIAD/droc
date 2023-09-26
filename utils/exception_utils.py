from utils.io.io_utils import HISTORY_TMP_PATH, USER_LOG, load_file, add_to_log
from utils.string_utils import get_lines_starting_with
import utils.perception.perception_utils as putils
import pickle

class PlanningError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class CodeError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class InterruptedByHuman(Exception):

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class WrongDetection(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class GraspError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class RobotError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class NotFinished(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


def interruption_handler(localss, locals, corr_rounds, li, step_name, code_as_policies):
    for k, v in localss.items():
        locals[corr_rounds][k] = v
    putils.save_current_image('cache/image_for_training', visualize=False)
    tmp = load_file(HISTORY_TMP_PATH)
    correction = input('Please tell me how to correct: ')
    add_to_log(correction, file_path=USER_LOG)
    if 'done' in correction.lower() or 'yes' in correction.lower() or correction.lower() == 'y':
        tmp[li][step_name][f'Outcome {corr_rounds}'] = 'Done.'
        tmp[li][step_name][f'Correction {corr_rounds}'] = 'Done'
        pickle.dump(tmp, open(HISTORY_TMP_PATH, "wb"))
        return True
    current_codeline = get_lines_starting_with(code_as_policies, "move_gripper")
    if 'code when interrupted' not in tmp[li][step_name].keys():
        tmp[li][step_name]['code when interrupted'] = code_as_policies.split(current_codeline)[1]
    add_to_log('current_codeline:' + current_codeline)
    tmp[li][step_name][f'Outcome {corr_rounds}'] = 'Interrupted by human at codeline "' + current_codeline + '".'
    tmp[li][step_name][f'Correction {corr_rounds}'] = correction
    pickle.dump(tmp, open(HISTORY_TMP_PATH, "wb"))
    return False


def robot_error_handler(localss, locals, corr_rounds, li, step_name, code_as_policies):
    for k, v in localss.items():
        locals[corr_rounds][k] = v
    tmp = load_file(HISTORY_TMP_PATH)
    current_codeline = get_lines_starting_with(code_as_policies, "move_gripper")
    add_to_log('current_codeline:' + current_codeline)
    while True:
        correction = input('I cannot reach that tatget. The solution is: 1. Modify your correction; 2. Delete that detection.')
        if '1' in correction:
            correction = input('Please tell me how to correct: ')
            add_to_log(correction, file_path=USER_LOG)
            try:
                tmp[li][step_name][f'Outcome {corr_rounds}'] = tmp[li][step_name][f'Outcome {corr_rounds-1}']
            except:
                tmp[li][step_name][f'Outcome {corr_rounds}'] = 'Interrupted'
            tmp[li][step_name][f'Correction {corr_rounds}'] = correction
            tmp[li][step_name]['code when interrupted'] = code_as_policies.split(current_codeline)[1]
            break
        elif '2' in correction:
            tmp[li][step_name]['code when interrupted'] = code_as_policies.split(current_codeline)[1]
            tmp[li][step_name][f'Outcome {corr_rounds}'] = 'Wrong detection. Interrupted at codeline "' + current_codeline + '".'
            tmp[li][step_name][f'Correction {corr_rounds}'] = "Please correct past detection."
        break
    if 'done' in correction.lower() or 'yes' in correction.lower() or correction.lower() == 'y':
        tmp[li][step_name][f'Outcome {corr_rounds}'] = 'Done.'
        tmp[li][step_name][f'Correction {corr_rounds}'] = 'Done'
        pickle.dump(tmp, open(HISTORY_TMP_PATH, "wb"))
        return True
    pickle.dump(tmp, open(HISTORY_TMP_PATH, "wb"))
    return False


def grasp_error_handler(localss, locals, corr_rounds, li, step_name, code_as_policies):
    for k, v in localss.items():
        locals[corr_rounds][k] = v
    tmp = load_file(HISTORY_TMP_PATH)
    try:
        current_codeline = get_lines_starting_with(code_as_policies, "move_gripper")
        if 'code when interrupted' not in tmp[li][step_name].keys():
            tmp[li][step_name]['code when interrupted'] = code_as_policies.split(current_codeline)[1]
        tmp[li][step_name][f'Outcome {corr_rounds}'] = 'Grasp failure at codeline "' + current_codeline + '".'
    except:
        tmp[li][step_name][f'Outcome {corr_rounds}'] = 'Grasp failure.'
    correction = input('Please tell me how to correct: ')
    add_to_log(correction, file_path=USER_LOG)
    if 'done' in correction.lower() or 'yes' in correction.lower() or correction.lower() == 'y':
        tmp[li][step_name][f'Outcome {corr_rounds}'] = 'Done.'
        tmp[li][step_name][f'Correction {corr_rounds}'] = 'Done'
        pickle.dump(tmp, open(HISTORY_TMP_PATH, "wb"))
        return True
    tmp[li][step_name][f'Correction {corr_rounds}'] = correction
    pickle.dump(tmp, open(HISTORY_TMP_PATH, "wb"))
    return False


def detection_error_handler(localss, locals, corr_rounds, li, step_name, code_as_policies):
    for k, v in localss.items():
        locals[corr_rounds][k] = v
    tmp = load_file(HISTORY_TMP_PATH)
    current_codeline = get_lines_starting_with(code_as_policies, "move_gripper")
    add_to_log('current_codeline:' + current_codeline)
    if "target" in current_codeline or "current" in current_codeline:
        lines = tmp[li][step_name][f'Correction {corr_rounds-1}'].splitlines()
    else:
        tmp[li][step_name]['code when interrupted'] = code_as_policies.split(current_codeline)[1]
        lines = tmp[li][step_name][f'code response 0'].splitlines()                
    obj = putils.parse_obj_name(lines[0])
    if "target" in current_codeline or "current" in current_codeline:
        code_as_policies = f'delete_last_detection("{obj}")'
    else:
        code_as_policies = f'delete_last_detection("{obj}")' + '\n' + tmp[li][step_name]['code response 0']
    print("**Error Type: ", "WrongDetection")
    add_to_log("**Response Code: " + code_as_policies)
    pickle.dump(tmp, open(HISTORY_TMP_PATH, "wb"))
    return code_as_policies


def no_exception_handler(localss, locals, corr_rounds, li, step_name, failure_reasoning, use_interrupted_code, code_as_policies):
    for k, v in localss.items():
        locals[corr_rounds][k] = v
    tmp = load_file(HISTORY_TMP_PATH)
    parsed_step_name = step_name.lower()[:-1] if step_name[-1] == '.' else step_name.lower()
    correction = input(f'Please issue further corrections, or is the task "{parsed_step_name}" done: ')
    # For dummy tests
    if correction == 'wrong detection':
        raise WrongDetection('')
    add_to_log(correction, file_path=USER_LOG)
    if 'done' in correction.lower() or 'yes' in correction.lower() or correction.lower() == 'y':
        if use_interrupted_code:
            tmp[li][step_name][f'Outcome {corr_rounds}'] = 'Done.'
            tmp[li][step_name][f'Correction {corr_rounds}'] = 'Done'
            pickle.dump(tmp, open(HISTORY_TMP_PATH, "wb"))
            return True, corr_rounds, None, use_interrupted_code
        else:
            a = input(f'Are you sure I have finish the task: "{parsed_step_name}" ? (y/n)')
            if 'y' in a.lower():
                tmp[li][step_name][f'Outcome {corr_rounds}'] = 'Done.'
                tmp[li][step_name][f'Correction {corr_rounds}'] = 'Done'
                pickle.dump(tmp, open(HISTORY_TMP_PATH, "wb"))
                return True, corr_rounds, None, use_interrupted_code
            else:
                tmp[li][step_name][f'Outcome {corr_rounds}'] = 'Done.'
                tmp[li][step_name][f'Correction {corr_rounds}'] = 'Please continue from interrupted code'
                correction_code = tmp[li][step_name]['code when interrupted']
                tmp[li][step_name][f'error type {corr_rounds}'] = 'None'
                corr_rounds += 1
                tmp[li][step_name][f'code response {corr_rounds}'] = correction_code
                pickle.dump(tmp, open(HISTORY_TMP_PATH, "wb"))
                code_as_policies = correction_code
                use_interrupted_code = True
    elif correction.lower() == 'no' or correction.lower() == 'n':
        correction = input(f'Please issue further corrections: ')
        add_to_log(correction, file_path=USER_LOG)
        tmp[li][step_name][f'Outcome {corr_rounds}'] = 'Done.'
        tmp[li][step_name][f'Correction {corr_rounds}'] = correction
        pickle.dump(tmp, open(HISTORY_TMP_PATH, "wb"))
        code_as_policies, corr_rounds = failure_reasoning(step_name, li, corr_rounds)
    else:   
        tmp[li][step_name][f'Outcome {corr_rounds}'] = 'Done.'
        tmp[li][step_name][f'Correction {corr_rounds}'] = correction
        try:
            if 'code when interrupted' not in tmp[li][step_name].keys():
                current_codeline = get_lines_starting_with(code_as_policies, "move_gripper")
                tmp[li][step_name]['code when interrupted'] = code_as_policies.split(current_codeline)[1]
        except:
            pass
        pickle.dump(tmp, open(HISTORY_TMP_PATH, "wb"))
        code_as_policies, corr_rounds = failure_reasoning(step_name, li, corr_rounds)
    return False, corr_rounds, code_as_policies, use_interrupted_code


def other_exception_handler(localss, locals, corr_rounds, li, step_name):
    correction = input("I cannot execute that correction. Please give me another correction:")
    for k, v in localss.items():
        locals[corr_rounds][k] = v
    tmp = load_file(HISTORY_TMP_PATH)
    add_to_log(correction, file_path=USER_LOG)
    if 'done' in correction.lower() or 'yes' in correction.lower() or correction.lower() == 'y':
        tmp[li][step_name][f'Outcome {corr_rounds}'] = 'Done.'
        tmp[li][step_name][f'Correction {corr_rounds}'] = 'Done'
        pickle.dump(tmp, open(HISTORY_TMP_PATH, "wb"))
        return True
    tmp[li][step_name][f'Outcome {corr_rounds}'] = 'Done'
    tmp[li][step_name][f'Correction {corr_rounds}'] = correction
    pickle.dump(tmp, open(HISTORY_TMP_PATH, "wb"))
    return False
