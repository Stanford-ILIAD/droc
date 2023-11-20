import pickle, json
import time
import cv2
import os
import numpy as np
from PIL import Image
import open3d as o3d
import shutil
import math
from utils.io.io_utils import open_file, delete_file, add_to_log, read_py
from utils.transformation_utils import extract_z_axis
from utils.LLM_utils import query_LLM

prompt_parse_pos = 'prompts/parse_pos.py'
prompt_get_task_pose = read_py('prompts/get_task_pose_str.txt')
prompt_get_pose_from_str = read_py('prompts/get_pose_from_str.py')
prompt_parse_name_str = 'prompts/parse_name.txt'
prompt_change_frame_str = 'prompts/change_frame.txt'
prompt_get_pos_scale = read_py('prompts/get_pos_scale.txt')
prompt_get_obj_name_from_task = read_py('prompts/get_obj_name_from_task.txt')
prompt_update_obj_state_file = 'prompts/update_obj_state.txt'
prompt_get_initial_obj_state = read_py('prompts/get_ini_obj_state.txt')

# global variables
load_detected_objs = None
popped_detected_objs = []
saved_detected_obj = {}
queried_obj = None
reference_frame = 'object'
RADIUS = 0.0135
TASK = None
REAL_ROBOT = None
policy = None
clip_model = None

def get_considered_classes():
    if TASK == 'drawer':
        considered_classes = [["drawer", "drawer handle"], ["pen"], ["cup", "mug"], ["tape"], ["scissors", "scissors handle"], ["apple"]]
        other_classes = [["table cloth", "wooden table", "wooden floor", "gray cloth", "gray background", "gray board"],["tripod"], ["unrecoginized object"]]
    elif TASK == 'coffee':
        considered_classes = [["coffee maker", "coffee machine", "keurig coffee maker"], ["red cup", "red mug"], ["coffee capsule", "cone"], ["pink cup", "pink mug"], ["white spoon"]]
        other_classes = [["unrecognizable object"], ["robot arm"], ["table cloth"], ["work surface"], ["tripod"]]
    elif TASK == 'cup':
        considered_classes = [["pink cup"], ["green cup", "blue cup"], ["gray rack", "gray oval object"]]
        other_classes = [["robot arm"], ["table cloth"], ["work surface"], ["tripod"]]
    elif TASK == 'lego':
        considered_classes = [["white drawer", "white drawer handle"], ["red block"], ["yellow block"], ["blue block"]]
        other_classes = [["robot arm"], ["table cloth"], ["work surface"], ["tripod"]]
    else:
        raise NotImplementedError
    return considered_classes, other_classes


class ToScaledFloat:
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, pic):
        # assert pic.dtype == torch.uint8, f"{pic.dtype}"
        pic = pic.float() / 255.0
        return pic

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class DinoImageTransform:
    def __init__(self):
        self.transforms = pt_transforms.Compose(
            [
                ToScaledFloat(),
                pt_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __call__(self, pic):
        return self.transforms(pic)

class detected_object: 
    def __init__(self):
        self.pcd = None
        self.prob = None
        self.quat = None
        self.position = None
        self.vec = None
        self.clip_feature = None
        self.dino_feature = None
        self.detected_pose = None

# ---------------------------------- Initialization ----------------------------------

def set_policy_and_task(real_robot, task):
    global TASK, REAL_ROBOT, policy
    global torch, open_clip, F, pt_transforms, device
    global SamAutomaticMaskGenerator, sam_model_registry, multi_cam, realsense_serial_numbers, clip_with_owl
    REAL_ROBOT = real_robot
    TASK = task   
    if REAL_ROBOT:
        import torch
        from torchvision import transforms as pt_transforms
        import torch.nn.functional as F
        import open_clip
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        from utils.perception.shared_devices import multi_cam, realsense_serial_numbers
        import utils.robot.robot_policy as robot_policy
        from utils.perception.owl_vit import clip_with_owl
        policy = robot_policy.KptPrimitivePolicy(multi_cam)
        device = "cuda" if torch.cuda.is_available() else "cpu"    
    else:
        from utils.robot.dummy_policy import DummyPolicy
        import open_clip
        import torch
        from torchvision import transforms as pt_transforms
        import torch.nn.functional as F 
        policy = DummyPolicy()

def initialize_detection(first=False, load_image=False, folder_path='cache/image_for_retrieval', image_pattern='*.png', label=['cup', 'drawer']):
    global load_detected_objs, saved_detected_obj, queried_obj, clip_model, clip_preprocess
    queried_obj = detected_object()
    if REAL_ROBOT:
        if first:
            a = input("Can I use last round detection? (y/n)")
        else:
            a = 'y'
        if 'n' in a:
            detect_objs(load_from_cache=False, save_to_cache=False, visualize=False)
            saved_detected_obj = {}
        load_detected_objs = pickle.load(open(f"log/{TASK}/detected_objs.pkl", "rb"))
    else:
        if first:
            a = input("Can I use last round detection? (y/n)")
        else:
            a = 'y'
        if 'n' in a:
            saved_detected_obj = {}
        if first:
            if load_image:
                # Only consider clip feature here
                if clip_model is None:
                    print("Loading CLIP...")
                    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k')
                image_files = glob.glob(os.path.join(folder_path, image_pattern))
                img_dict = {}
                for l in label:
                    img_files = [file for file in image_files if l in os.path.basename(file).lower()]
                    img_files = sorted(img_files, key=lambda x: int(x.split(l)[1].split('.png')[0]))
                    img_dict[l] = img_files
                files = {}
                clip_model.to(device=device)
                for l in label:
                    img_files = img_dict[l]
                    all_img = []
                    _files = []
                    for file in img_files:
                        _files.append(file)
                        img = Image.open(file)
                        img = clip_preprocess(img).to(device)
                        all_img.append(img)
                    preprocessed_images = torch.stack(all_img)    
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        image_features = clip_model.encode_image(preprocessed_images)
                    image_features = image_features.cpu().numpy()
                    img_dict[l] = image_features
                    files[l] = _files
                load_detected_objs = create_loaded_objs(img_dict)
            else:
                load_detected_objs = create_loaded_objs(None)
        else:
            load_detected_objs = pickle.load(open(f"log/{TASK}/detected_objs.pkl", "rb"))

def create_loaded_objs(img_dict):
    considered_classes, _ = get_considered_classes()
    load_detected_objs = {}
    for cls in considered_classes:
        if img_dict is None:
            obj_name = cls[0]
            load_detected_objs[obj_name] = [(np.ones((10,3)), 0.5, np.random.randn(1024),np.random.randn(512)), (np.ones((10,3)), 0.3, np.random.randn(1024),np.random.randn(512))]
        else:
            if obj_name not in list(img_dict.keys()):
                n = 3
                for i in range(n):
                    obj = obj_name + f'_{i+1}'
                    load_detected_objs[obj] = (np.ones((10,3)), 0.5, np.random.randn(1024),np.random.randn(512))
            else:
                image_feature = img_dict[obj_name]
                n = len(image_feature)
                for i in range(n):
                    obj = obj_name + f'_{i+1}'
                    load_detected_objs[obj] = (np.ones((n+2,3)), 0.5, image_feature[i], np.random.randn(512))
    pickle.dump(load_detected_objs, open(f"log/{TASK}/detected_objs.pkl", "wb"))
    return load_detected_objs

def get_objs():
    global load_detected_objs
    return load_detected_objs

def get_initial_state():
    global load_detected_objs
    # obj_list_str = ''
    # for obj_name in list(load_detected_objs.keys()):
    #     obj_list_str += obj_name + ', '
    # obj_list_str = obj_list_str[:-2]
    # whole_prompt = prompt_get_initial_obj_state + '\n' + 'Object name: ' + obj_list_str +'\n' + 'Object state:'
    # response = query_LLM(whole_prompt, [], "cache/llm_get_initial_state.pkl")
    # initial_state = response.text
    initial_state = 'Object state: N/A'
    return initial_state

def get_extra_classes():
    if TASK == 'drawer':
        return ["scissors", "scissors handle"], 2
    else:
        return None, None

def setup_clip_words():
    considered_classes, other_classes = get_considered_classes()
    clip_candidates = []
    for word in considered_classes + other_classes:
        clip_candidates.extend(word)
    return clip_candidates

def clear_saved_detected_obj():
    global saved_detected_obj
    saved_detected_obj = {}

def set_saved_detected_obj(task_name, task_feature):
    obj_name = get_obj_name_from_task(task_name)
    saved_detected_obj[obj_name] = task_feature

# ---------------------------------- Primitives ----------------------------------

def _change_reference_frame(wrong_direction, right_direction):
    global queried_obj
    with open(prompt_change_frame_str, "r") as template_file:
        template_content = template_file.read()
    values = {"wrong_direction": wrong_direction, "correct_direction": right_direction}
    prompt = template_content.format(**values)
    response = query_LLM(prompt, ['.'], "cache/llm_change_frame.pkl")
    add_to_log(prompt)
    add_to_log(response.text)
    if "left" in response.text:
        correct_forward_vec = get_directional_vec("left")
    if "right" in response.text:
        correct_forward_vec = get_directional_vec("right")
    if "forward" in response.text:
        correct_forward_vec = get_directional_vec("forward")
    if "back" in response.text:
        correct_forward_vec = get_directional_vec("back")
    queried_obj.vec = correct_forward_vec

def get_directional_vec(direction, *kwargs):
    global queried_obj, reference_frame
    if reference_frame == 'absolute':
        if direction == 'forward' or direction == 'along' or direction == 'towards':
            return np.array((1,0,0))
        elif direction == 'backward' or direction == 'back' or direction == 'away':
            return np.array((-1,0,0))
        elif direction == 'left':
            return np.array((0,1,0))
        elif direction == 'right':
            return np.array((0,-1,0))
        elif direction == 'up' or direction == 'above':
            return np.array((0,0,1))
        elif direction == 'down' or direction == 'downward':
            return np.array((0,0,-1))
        else:
            return np.array((1,0,0))
    else:
        if REAL_ROBOT:
            vec = queried_obj.vec.squeeze()
        else:
            vec = np.array((1.,0.,0.))
        vec = vec/np.linalg.norm(vec)
        if direction == 'forward' or direction == 'along' or direction == 'towards':
            vec = vec if vec[0] >=0 else -vec
            return vec
        elif direction == 'backward' or direction == 'back' or direction == 'away':
            vec = -vec if vec[0] >=0 else vec
            return vec
        elif direction == 'left':
            normal_vec = np.array((-vec[1], vec[0], vec[2])) if vec[0] >=0 else np.array((vec[1], -vec[0], -vec[2]))
            return normal_vec
        elif direction == 'right':
            normal_vec = np.array((-vec[1], vec[0], vec[2])) if vec[0] <=0 else np.array((vec[1], -vec[0], -vec[2]))
            return normal_vec
        elif direction == 'up' or direction == 'above':
            vec = np.array((0,0,1))
            return vec
        elif direction == 'down' or direction == 'downward':
            vec = np.array((0,0,-1))
            return vec
        else:
            return np.array((1.,0.,0.))

def get_current_pos(*kwargs):
    if REAL_ROBOT:
        pose = policy.robot_env.robot.get_ee_pose()
        ee_pos = pose[0].numpy()
        return ee_pos
    else:
        return np.array((1.,0.,0.))

def get_current_ori(*kwargs):
    if REAL_ROBOT:
        pose = policy.robot_env.robot.get_ee_pose()
        ee_ori = pose[1].numpy()
        return ee_ori
    else:
        return np.array((1.,0.,0.,0.))

def get_horizontal_ori(get_vec=False):
    if get_vec == False:
        return policy.get_horizontal_ori()
    else:
        quat = policy.get_horizontal_ori()
        vec = extract_z_axis(quat)
        vec[2] = 0.
        vec = -vec if vec[0]<0 else vec
        vec /= np.linalg.norm(vec)
        return vec
    
def get_vertical_ori(get_vec=False):
    if get_vec==False:
        return policy.get_vertical_ori()
    else:
        return np.array((0.,0.,-1.))

def _correct_past_detection(obj_name, obj):
    global load_detected_objs, popped_detected_objs
    obj_name = parse_obj_name(obj_name)
    for i in range(len(load_detected_objs[obj_name])):
        if (obj.pcd.shape == load_detected_objs[obj_name][i][0].shape):
            popped_elem = load_detected_objs[obj_name].pop(i)
            popped_detected_objs.append(popped_elem)
            break

# ---------------------------------- Parsing ----------------------------------

def get_obj_name_from_task(task_name):
    prompt = prompt_get_obj_name_from_task + '\n' + f'Input: {task_name}' + '\n' + 'Output: '
    response = query_LLM(prompt, [], "cache/llm_parse_task_name.pkl")
    obj_name = response.text
    return obj_name

def parse_obj_name(obj_name):
    considered_classes, _ = get_considered_classes()
    with open(prompt_parse_name_str, 'r') as file:
        template_content = file.read()
    values = {"object_class": considered_classes, "object_name": obj_name}
    prompt = template_content.format(**values)
    response = query_LLM(prompt, [], "cache/llm_parse_obj_name.pkl")
    obj_class_idx = int(response.text)
    parsed_obj_name = considered_classes[obj_class_idx][0]
    return parsed_obj_name

def _parse_pos(pos_description, r_frame='object'):
    global reference_frame
    reference_frame = r_frame
    mapping_dict, little_left, little_up, little_forward, more_back = calculate_pos_scale()
    with open(prompt_parse_pos, "r") as template_file:
        template_content = template_file.read()
    values = {"mapping_dict": mapping_dict, "little_left": little_left, "little_up": little_up, "little_forward": little_forward, "more_back": more_back}
    prompt = template_content.format(**values)
    whole_prompt = prompt + '\n' + '\n' + f"""# Query: {pos_description}.""" + "\n"
    response = query_LLM(whole_prompt, ["# Query:"], "cache/llm_response_parse_pos.pkl")
    code_as_policies = response.text
    add_to_log('-'*80)
    add_to_log('*at parse_pos*')
    add_to_log(code_as_policies)
    add_to_log('-'*80)
    locals = {}
    exec(code_as_policies, globals(), locals)
    return locals['ret_val'], code_as_policies

def calculate_pos_scale():
    global queried_obj
    if REAL_ROBOT:
        whole_prompt = prompt_get_pos_scale + '\n\n' + 'Bounding box: ' + str(queried_obj.bounding_box)
    else:
        whole_prompt = prompt_get_pos_scale + '\n\n' + 'Bounding box: (1, 8, 3)'
    response = query_LLM(whole_prompt, ["Bounding box:"], "cache/llm_get_pos_scale.pkl")
    response_json = response.text
    mapping_dict = json.loads(response_json)
    add_to_log(mapping_dict)
    little_forward = mapping_dict["'A little bit' for 'forward' or 'backward'"]/100
    little_left = mapping_dict["'A little bit' for 'left' or 'right'"]/100
    little_up = mapping_dict["'A little bit' for 'up' or 'down'"]/100
    more_back = mapping_dict["'More' for 'forward' or 'back'"]/100
    return mapping_dict, little_left, little_up, little_forward, more_back

# ---------------------------------- Detection ----------------------------------

def _get_task_detection(task_name):
    """Given task name, find corresponding feature"""
    global saved_detected_obj

    obj_name = get_obj_name_from_task(task_name)

    if obj_name in saved_detected_obj.keys():
        compare_feature(saved_detected_obj[obj_name], obj_name, threshold=1)
        return saved_detected_obj[obj_name]
    
    whole_prompt = prompt_get_pose_from_str + '\n' + "# Query: " + f"""the centroid of "{obj_name}"."""
    response = query_LLM(whole_prompt, ["INPUT:"], "cache/llm_response_get_pose_from_str.pkl")
    code_as_policies = response.text
    add_to_log('-'*80 + '*at get_pose_from_str*' + code_as_policies + '-'*80)
    while True:
        locals = {}
        exec(code_as_policies, globals(), locals)
        obj = locals['ret_val']
        if REAL_ROBOT:
            pcd_merged = o3d.io.read_point_cloud(f"log/{TASK}/pcd_merged.pcd")    
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            visualized_geometries = [pcd_merged, mesh_frame]
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=RADIUS); sphere.paint_uniform_color([0.0, 1.0, 0.0])
            sphere.translate(calculate_centroid(obj.pcd))
            visualized_geometries.append(sphere)
            o3d.visualization.draw_geometries(visualized_geometries)
        correct_detection = input("Is this detection correct? (y/n)")
        if 'y' in correct_detection.lower():
            saved_detected_obj[obj_name] = obj.dino_feature
            compare_feature(obj.dino_feature, obj_name, threshold=1)
            return obj.dino_feature
        else:
            _correct_past_detection(obj_name, obj)

def detect(obj_name, visualize=False):
    global load_detected_objs, saved_detected_obj

    parsed_obj_name = parse_obj_name(obj_name)
    obj_name = parsed_obj_name

    if load_detected_objs[obj_name] == [] or load_detected_objs[obj_name] is None:
        while True:
            a = input("It seems that the correct object was not detected, please make sure to place the drawer in a correct position. Reply 'y' once you finish it.")
            if "y" in a:
                break
        if REAL_ROBOT:
            detect_objs(load_from_cache=False, save_to_cache=False, visualize=visualize)
            load_detected_objs = pickle.load(open(f"log/{TASK}/detected_objs.pkl", "rb"))
        else:
            load_detected_objs = create_loaded_objs()

    all_pcds = []
    probs = []
    clip_features = []
    dino_features = []

    for pcd, prob, clip_feature, dino_feature in load_detected_objs[obj_name]:
        all_pcds.append(pcd)
        probs.append(prob)
        clip_features.append(clip_feature)
        dino_features.append(dino_feature)
    if visualize:
        pcd_merged = o3d.io.read_point_cloud(f"log/{TASK}/pcd_merged.pcd")
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        visualized_geometries = [pcd_merged, mesh_frame]
        for pcd in all_pcds:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=RADIUS)
            sphere.paint_uniform_color([0.0, 1.0, 0.0])
            sphere.translate(np.mean(pcd,axis=0))
            visualized_geometries.append(sphere)
        o3d.visualization.draw_geometries(visualized_geometries)

    all_obj = detected_object()
    all_obj.pcd = all_pcds
    all_obj.prob = probs
    all_obj.clip_feature = clip_features
    all_obj.dino_feature = dino_features
    return all_obj

def sort_from_high_to_low(all_obj, key):
    all_obj_pos = [np.mean(i, axis=0) for i in all_obj.pcd]
    probs = all_obj.prob
    clip_features = all_obj.clip_feature
    dino_features = all_obj.dino_feature
    if key == 'probability':
        order = np.argsort(np.array(probs))[::-1]
    elif key == 'x':
        x_pos = [x[0] for x in all_obj_pos]
        order = np.argsort(np.array(x_pos))[::-1]
    elif key == 'y':
        y_pos = [x[1] for x in all_obj_pos]
        order = np.argsort(np.array(y_pos))[::-1]
    elif key == 'z':
        z_pos = [x[2] for x in all_obj_pos]
        order = np.argsort(np.array(z_pos))[::-1]
    obj_list = []
    for ind in order:
        obj = detected_object()
        obj.pcd = all_obj.pcd[ind]
        obj.prob = probs[ind]
        obj.clip_feature = clip_features[ind]
        obj.dino_feature = dino_features[ind]
        obj_list.append(obj)
    return obj_list

def detect_objs(load_from_cache=False, run_on_server=False, save_to_cache=True, visualize=False):
    global clip_model, clip_preprocess
    parent_folder = 'cache'
    items = os.listdir(parent_folder)
    for item in items:
        item_path = os.path.join(parent_folder, item)    
        if os.path.isdir(item_path) and (item[0].isdigit() or item == TASK):
            try:
                shutil.rmtree(item_path)
                print(f"Folder '{item_path}' and its contents deleted successfully.")
            except OSError as e:
                print(f"Error deleting '{item_path}': {e}")
    try:
        shutil.rmtree(f'log/{TASK}')
        print(f"Folder 'log/{TASK}' and its contents deleted successfully.")
    except OSError as e:
        print(f"Error deleting '{item_path}': {e}")

    print("Loading CLIP model...")
    init_time = time.time()
    if clip_model is None:
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k')
    print("time taken:", time.time() - init_time)

    print("Loading DINOv2 model...")
    init_time = time.time()
    dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dinov2_model.train(False)
    for param in dinov2_model.parameters():
        param.requires_grad = False
    dinov2_preprocess = DinoImageTransform()
    print("time take: ", time.time() - init_time)

    clip_candidates = setup_clip_words()
    text_features = get_text_features(clip_candidates, clip_model)
    considered_classes, other_classes = get_considered_classes()
    detected_objs = {words[0]: [] for words in considered_classes}
    if load_from_cache:
        print('Loading data...')
        init_time = time.time()
        bgr_images = pickle.load(open(f"cache/{TASK}/bgr_images.pkl", "rb"))
        pcd_merged = o3d.io.read_point_cloud(f"cache/{TASK}/pcd_merged.pcd")
        detected_objs = pickle.load(open(f"cache/{TASK}/detected_objs.pkl", "rb"))
        print("time taken:", time.time() - init_time)
    else:
        print("Computing point cloud...")
        init_time = time.time()
        used_camera_idx = 0
        bgr_images, pcd_merged, raw_points, _ = multi_cam.take_bgrd(visualize=visualize)
        image = bgr_images[realsense_serial_numbers[used_camera_idx]]
        coord2point_3d = raw_points[used_camera_idx]
        del raw_points
        print("time taken:", time.time() - init_time)
     
        print("Loading SAM model...")
        init_time = time.time()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        sam = sam_model_registry["vit_h"](checkpoint="cache/sam_vit_h_4b8939.pth")
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32, 
            pred_iou_thresh=0.95,
            stability_score_thresh=0.95,
            box_nms_thresh=0.2,
            crop_n_layers=1,
            crop_overlap_ratio=0.6,
            min_mask_region_area=100
        )
        print("time taken:", time.time() - init_time)
    
        print("Cropping picture...")
        init_time = time.time()
        height, width = image.shape[:2]
        print(width, height)
        coord2point_3d = np.reshape(coord2point_3d, (height, width, -1))
        print("time taken:", time.time() - init_time)

        print("Generating masks...")
        init_time = time.time()
        sam.to(device=device)
        masks = mask_generator.generate(image)
        mask_points = []
        mask_images = []
        mask_preprocessed_images = []
        extra_classes, num = get_extra_classes()
        if extra_classes is not None:
            extra_masks, boxes = clip_with_owl(image, num, sam, obj_name=extra_classes[0], visualize=False)
            for mask, box in zip(extra_masks, boxes):
                tmp_mask = {'segmentation': mask, 'bbox': box, 'area': mask.sum()}
                masks.append(tmp_mask)
            if extra_classes not in considered_classes:
                considered_classes.append(extra_classes)
        masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        del sam
        del mask_generator
        print("time taken:", time.time() - init_time)

        print("Processing masks...")
        init_time = time.time()
        for i, mask in enumerate(masks):
            # if mask['area'] < 600:
            #     break
            duplicate = False
            for prev_mask in masks[:i]:
                intersection = np.sum(mask['segmentation'] & prev_mask['segmentation'])
                union = np.sum(mask['segmentation'] | prev_mask['segmentation'])
                if intersection / union > 0.1:
                    duplicate = True
                    break
            if duplicate:
                continue

            x, y, w, h = map(int, mask['bbox'])
            print(x, y, w, h)
            # bbox = (y,x,y+h,x+w)
            # project(used_camera_idx,bbox,max_num=300)
            background = 255
            im = np.ones((h, w, 3), dtype=np.uint8) * background
            mask_points.append(coord2point_3d[mask['segmentation']])
            new_mask = mask['segmentation'][y:min(y+h, height), x:min(x+w, width)]
            print(new_mask.shape)
            im[new_mask] = image[y:min(y+h, height), x:min(x+w, width), :][new_mask]
            square_size = max(h, w) + 6
            im = np.pad(im, (((square_size - h) // 2, (square_size - h) // 2), ((square_size - w) // 2, (square_size - w) // 2), (0, 0)), 'constant', constant_values=background)
            # im = image[y:y+h, x:x+w, :]
            mask_images.append(im)
            mask_preprocessed_images.append(clip_preprocess(Image.fromarray(im)).to(device))
        mask_preprocessed_images = torch.stack(mask_preprocessed_images)
        print("time taken:", time.time() - init_time)

        print("Getting labels with CLIP...")
        init_time = time.time()
        clip_model.to(device=device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = clip_model.encode_image(mask_preprocessed_images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            raw_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu()
            print(image_features.shape, text_features.shape)
        clip_model.to(device="cpu")
        image_features = image_features.cpu().numpy()
        print("time taken:", time.time() - init_time)

        print("Compute DINOv2 features...")
        init_time = time.time()
        for i in range(len(mask_images)):
            mask_images[i] = cv2.resize(mask_images[i], (224, 224))
        mask_dino_preprocessed_images = torch.from_numpy(np.transpose(np.stack(mask_images), (0, 3, 1, 2))).to(device)
        dinov2_model.to(device=device)
        dino_image_features = dinov2_model.forward_features(dinov2_preprocess(mask_dino_preprocessed_images))["x_norm_clstoken"]
        dino_image_features = F.normalize(dino_image_features, dim=-1)
        dinov2_model.to(device="cpu")
        dino_image_features = dino_image_features.cpu().numpy()
        print("time taken:", time.time() - init_time)

        print("Postprocessing raw CLIP probabilities...")
        init_time = time.time()
        len_list = [len(words) for words in considered_classes + other_classes]
        probs = torch.zeros(*raw_probs.shape[:1], len(considered_classes) + len(other_classes), dtype=raw_probs.dtype, device=raw_probs.device)
        for i in range(len(considered_classes)):
            probs[..., i] = (raw_probs[..., sum(len_list[:i]) : sum(len_list[:i + 1])].max(dim=-1).values)
        for i in range(len(other_classes)):
            probs[..., i + len(considered_classes)] = (raw_probs[..., sum(len_list[:len(considered_classes) + i]) : sum(len_list[:len(considered_classes) + i + 1])].max(dim=-1).values)
        del raw_probs
        del clip_model
        del clip_preprocess
        print("time taken:", time.time() - init_time)

        print("Filtering detection results...")
        init_time = time.time()
        probs = probs.numpy()
        for i, p in enumerate(probs):
            predicted_class_idx = np.argmax(p)
            predicted_class_name = considered_classes[predicted_class_idx][0] if predicted_class_idx < len(considered_classes) else other_classes[predicted_class_idx - len(considered_classes)][0]
            if detected_objs.get(predicted_class_name) is not None:
                # detected_objs[predicted_class_name].append((mask_points[i], p[predicted_class_idx], norm_vec[i]))
                detected_objs[predicted_class_name].append((mask_points[i], p[predicted_class_idx], image_features[i], dino_image_features[i]))
            if not os.path.exists(f"cache/{predicted_class_idx}_{predicted_class_name}"):
                os.makedirs(f"cache/{predicted_class_idx}_{predicted_class_name}")
            cv2.imwrite(f"cache/{predicted_class_idx}_{predicted_class_name}/{i}_{p[predicted_class_idx]:.4f}.png", mask_images[i][:, :, ::-1])
        print("time taken:", time.time() - init_time)
        
        if save_to_cache:
            pickle.dump(bgr_images, open_file(f"cache/{TASK}/bgr_images.pkl", "wb"))
            o3d.io.write_point_cloud(f"cache/{TASK}/pcd_merged.pcd", pcd_merged)
            pickle.dump(detected_objs, open_file(f"cache/{TASK}/detected_objs.pkl", "wb"))

    delete_file(f"log/{TASK}/bgr_images.pkl")
    delete_file(f"log/{TASK}/pcd_merged.pcd")
    delete_file(f"log/{TASK}/detected_objs.pkl")
    pickle.dump(bgr_images, open_file(f"log/{TASK}/bgr_images.pkl", "wb"))
    o3d.io.write_point_cloud(f"log/{TASK}/pcd_merged.pcd", pcd_merged)
    pickle.dump(detected_objs, open_file(f"log/{TASK}/detected_objs.pkl", "wb"))

    if visualize:
        print("Plotting point cloud...")
        init_time = time.time()
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        detected_objects_geomertries = o3d.geometry.TriangleMesh()
        for key, detected_points in detected_objs.items():
            for pcd, _, _ in detected_points:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=RADIUS)
                sphere.translate(np.mean(pcd, axis=0))
                detected_objects_geomertries += sphere
        detected_objects_geomertries.paint_uniform_color([0.0, 1.0, 0.0])
        o3d.visualization.draw_geometries([pcd_merged, mesh_frame, detected_objects_geomertries])
        print("time taken:", time.time() - init_time)

def get_text_features(clip_candidates, clip_model):
    global tokenizer
    try:
        text_features = pickle.load(open(f"cache/{TASK}/text_features.pkl", "rb"))
        return text_features
    except:
        if tokenizer is None:
            tokenizer = open_clip.get_tokenizer('ViT-g-14')
        texts = tokenizer([f"A photo of {c}." for c in clip_candidates]).to(device)
        clip_model.to(device=device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = clip_model.encode_text(texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        clip_model.to(device="cpu")
        pickle.dump(text_features, open_file(f"cache/{TASK}/text_features.pkl", "wb"))
        return text_features

# ---------------------------------- Get task pose ----------------------------------

def _get_task_pose(task_name, visualize=True):
    task_name = task_name[:-1] if task_name[-1] == '.' else task_name
    whole_prompt = prompt_get_task_pose + '\n' + '\n' + f"INPUT: the robot's task is to {task_name.lower()}." + "\n"
    add_to_log(whole_prompt)
    response = query_LLM(whole_prompt, ["INPUT:"], "cache/llm_response_get_task_pose.pkl")
    if "response" in response.text.lower():
        response_json = response.text.split('\n', 1)[1]
        add_to_log(response_json)
        pose_dict = json.loads(response_json)
    else:
        pose_dict = json.loads(response.text)
    add_to_log('-'*80)
    add_to_log('*at get_task_pose*')
    add_to_log('get_task_pose:', pose_dict)
    add_to_log('-'*80)
    pos_str = pose_dict['gripper position']
    ori_str = pose_dict['gripper orientation']
    _get_task_detection(task_name)
    pos, ori = get_pose_from_str(pos_str, ori_str, visualize=visualize)
    return pos, ori

def get_pose_from_str(pos_str, ori_str, visualize=False):
    global queried_obj
    pos_feature = pos_str.split('.')[1]
    obj_name = pos_str.split('.')[0]
    if 'absolute_vertical' in ori_str or 'absolute_horizontal' in ori_str:
        ori_feature = ori_str
        whole_prompt = prompt_get_pose_from_str + '\n' + "# Query: " + f"""the {pos_feature} of "{obj_name}"."""
    else:
        ori_feature = ori_str.split('.')[1]
        whole_prompt = prompt_get_pose_from_str + '\n' + "# Query: " + f"""the {pos_feature} and {ori_feature} of "{obj_name}"."""
    response = query_LLM(whole_prompt, ["INPUT:"], "cache/llm_response_get_pose_from_str.pkl")
    code_as_policies = response.text
    add_to_log('-'*80)
    add_to_log('*at get_pose_from_str*')
    add_to_log(code_as_policies)
    add_to_log('-'*80)
    locals = {}
    exec(code_as_policies, globals(), locals)
    obj = locals['ret_val']
    if ori_feature == 'absolute_vertical':
        if 'drawer' in obj_name:
            vertical_quat = get_vertical_ori()
            minor_axis = get_minor_axis_from_pcd(obj.pcd)
            rotate_angle = math.degrees(-math.atan2(minor_axis[1], minor_axis[0]))
            tar_quat = policy.rotate_around_gripper_z_axis(rotate_angle, vertical_quat)
            obj.rotation = tar_quat
            obj.vec = minor_axis
        elif 'shelf' in obj_name:
            obj.rotation = get_current_ori()
            minor_axis = get_minor_axis_from_pcd(obj.pcd)
            obj.vec = minor_axis
        else:
            vertical_quat = get_vertical_ori()
            main_axis = get_x_axis_from_pcd(obj.pcd)
            rotate_angle = math.degrees(-math.atan2(main_axis[1], main_axis[0]))
            tar_quat = policy.rotate_around_gripper_z_axis(rotate_angle, vertical_quat)
            obj.rotation = tar_quat
            obj.vec = main_axis
    elif ori_feature == 'absolute_horizontal':
        obj.rotation = get_horizontal_ori(get_vec=False)
        obj.vec = get_horizontal_ori(get_vec=True)
    else:
        obj.vec = obj.rotation.copy()
        obj.rotation = policy.align_z_axis_with_vector(obj.vec)
    if REAL_ROBOT:
        pcd_merged = o3d.io.read_point_cloud(f"log/{TASK}/pcd_merged.pcd")    
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        visualized_geometries = [pcd_merged, mesh_frame]
        for i in range(50):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=RADIUS/3)
            sphere.paint_uniform_color([0.0, 1.0, 0.0])
            sphere.translate(calculate_centroid(obj.pcd)+i/100*obj.vec)
            visualized_geometries.append(sphere)
        o3d.visualization.draw_geometries(visualized_geometries)
    obj.position = policy.robot_fingertip_pos_to_ee(obj.position, obj.rotation)
    obj.bounding_box = get_bounding_box_from_pcd(obj.pcd, obj.vec)
    queried_obj.bounding_box = obj.bounding_box
    queried_obj.vec = obj.vec
    if queried_obj.detected_pose is None:
        queried_obj.clip_feature = obj.clip_feature.copy()
        queried_obj.dino_feature = obj.dino_feature.copy()
        queried_obj.detected_pose = (obj.position, obj.rotation)
    return obj.position, obj.rotation

def calculate_centroid(point_cloud):
    centroid = np.mean(point_cloud, axis=0)
    return centroid

def calculate_normal_vector(point_cloud):
    cov_matrix = np.cov(point_cloud, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    min_eigenvalue_index = np.argmin(eigenvalues)
    normal_vector = eigenvectors[:, min_eigenvalue_index]
    if normal_vector[0]<0:
        normal_vector = -normal_vector
    normal_vector[2] = 0.
    normal_vector /= np.linalg.norm(normal_vector)
    return normal_vector
    
def calculate_major_axis(point_cloud):
    covariance_matrix = np.cov(point_cloud, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    main_axis = eigenvectors[:, np.argmax(eigenvalues)]
    return main_axis
    # return calculate_normal_vector(point_cloud)

def calculate_minor_axis(point_cloud):
    return calculate_normal_vector(point_cloud)

def calculate_tail_point(point_cloud):
    x_axis = get_x_axis_from_pcd(point_cloud)
    centroid = calculate_centroid(point_cloud)
    tail_point = centroid - 0.5 * x_axis
    return tail_point

def get_x_axis_from_pcd(point_cloud):
    if REAL_ROBOT:
        o3d_point_cloud = o3d.geometry.PointCloud()
        o3d_point_cloud.points = o3d.utility.Vector3dVector(point_cloud)
        bbox = o3d_point_cloud.get_oriented_bounding_box()
        eight_vertices = np.asarray(bbox.get_box_points())
        sorted_indices = np.argsort(eight_vertices[:,2])[::-1]
        sorted_array_z = eight_vertices[sorted_indices]
        sorted_array_z = sorted_array_z[:4,:]
        len_xy = np.linalg.norm(sorted_array_z[0]-sorted_array_z[1])
        axis_xy = sorted_array_z[0]-sorted_array_z[1]
        len_yx = np.linalg.norm(sorted_array_z[0]-sorted_array_z[2])
        axis_yx = sorted_array_z[0]-sorted_array_z[2]
        if len_xy > len_yx:
            main_axis = axis_xy
        else:
            main_axis = axis_yx
        main_axis[2] = 0.
        main_axis /= np.linalg.norm(main_axis)
        main_axis = main_axis if main_axis[0]>0 else -main_axis
        return main_axis
    else:
        return np.array((1.,0.,0.))

def get_minor_axis_from_pcd(point_cloud):
    if REAL_ROBOT:
        o3d_point_cloud = o3d.geometry.PointCloud()
        o3d_point_cloud.points = o3d.utility.Vector3dVector(point_cloud)
        bbox = o3d_point_cloud.get_oriented_bounding_box()
        eight_vertices = np.asarray(bbox.get_box_points())
        sorted_indices = np.argsort(eight_vertices[:,2])[::-1]
        sorted_array_z = eight_vertices[sorted_indices]
        sorted_array_z = sorted_array_z[:4,:]
        len_xy = np.linalg.norm(sorted_array_z[0]-sorted_array_z[1])
        axis_xy = sorted_array_z[0]-sorted_array_z[1]
        len_yx = np.linalg.norm(sorted_array_z[0]-sorted_array_z[2])
        axis_yx = sorted_array_z[0]-sorted_array_z[2]
        if len_xy > len_yx:
            minor_axis = axis_yx
        else:
            minor_axis = axis_xy
        minor_axis[2] = 0.
        minor_axis /= np.linalg.norm(minor_axis)
        minor_axis = minor_axis if minor_axis[0]>0 else -minor_axis
        return minor_axis
    else:
        return (1., 0., 0.)

def get_bounding_box_from_pcd(point_cloud, forward_vec):
    if REAL_ROBOT:
        o3d_point_cloud = o3d.geometry.PointCloud()
        o3d_point_cloud.points = o3d.utility.Vector3dVector(point_cloud)
        bbox = o3d_point_cloud.get_oriented_bounding_box()
        eight_vertices = np.asarray(bbox.get_box_points())
        sorted_indices = np.argsort(eight_vertices[:,2])[::-1]
        sorted_array_z = eight_vertices[sorted_indices]
        len_z = sorted_array_z[:4,2].mean() - sorted_array_z[4:,2].mean()
        sorted_array_z = sorted_array_z[:4,:]
        len_xy = np.linalg.norm(sorted_array_z[0]-sorted_array_z[1])
        axis_xy = sorted_array_z[0]-sorted_array_z[1]
        len_yx = np.linalg.norm(sorted_array_z[0]-sorted_array_z[2])
        axis_yx = sorted_array_z[0]-sorted_array_z[2]
        if len_xy > len_yx:
            main_axis = axis_xy
            len_x = len_xy
            len_y = len_yx
        else:
            main_axis = axis_yx
            len_x = len_yx
            len_y = len_xy
        main_axis[2] = 0.
        main_axis /= np.linalg.norm(main_axis)
        minor_axis = np.array((main_axis[1], -main_axis[0], 0.))
        if np.abs(forward_vec.dot(main_axis)) > np.abs(forward_vec.dot(minor_axis)):
            return (round(len_x*100), round(len_y*100), round(len_z*100))
        else:
            return (round(len_y*100), round(len_x*100), round(len_z*100))
    else:
        return (1., 8., 3.)

# ---------------------------------- Image feature ----------------------------------

def compare_text_image_sim(text, vis_feature):
    global clip_model, tokenizer
    vis_feature = torch.tensor(vis_feature, device=device)
    if tokenizer is None:
        tokenizer = open_clip.get_tokenizer('ViT-g-14')
    if clip_model is None:
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k')
    clip_model = clip_model.to(device=device)
    text = tokenizer(f"A photo of {text}.").to(device=device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_feature = clip_model.encode_text(text).squeeze()
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        vis_feature /= vis_feature.norm(dim=-1, keepdim=True)
        raw_probs = vis_feature.dot(text_feature).item()
    del clip_model
    return raw_probs

def get_detected_feature():
    global queried_obj
    if REAL_ROBOT:
        image_feature = queried_obj.clip_feature.copy()
        dino_image_feature = queried_obj.dino_feature.copy()
        detected_pose = queried_obj.detected_pose
        return (image_feature, dino_image_feature, detected_pose)
    else:
        return (np.random.randn(1024), np.random.randn(512), (np.array((1.,0.,0.)), np.array((1.,0.,0.,0.))))

def compare_feature(img_feature, task_name, visualize=False, threshold=10, metric='L2'):
    """ Delete all irrelevant features """
    global load_detected_objs
    obj_name = get_obj_name_from_task(task_name)
    obj_name = parse_obj_name(obj_name)
    objs_specified = load_detected_objs[obj_name]
    similarity = []
    for _, _, _, new_dino_feature in objs_specified:
        if metric == 'L2':
            similarity.append(((new_dino_feature-img_feature)**2).sum())
        elif metric == 'cosine':
            similarity.append(1-new_dino_feature.dot(img_feature)/(np.linalg.norm(new_dino_feature)*np.linalg.norm(img_feature)))
        else:
            raise NotImplementedError
    add_to_log(similarity)
    order = np.argsort(np.array(similarity))
    picked_obj_idx = list(order[:threshold])
    picked_obj = [objs_specified[i] for i in picked_obj_idx]
    load_detected_objs[obj_name] = picked_obj
    if visualize:
        pcd_merged = o3d.io.read_point_cloud(f'log/{TASK}/pcd_merged.pcd')
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        detected_objects_geomertries = o3d.geometry.TriangleMesh()
        for obj in picked_obj:
            pcd, _, _, _ = obj
            p = calculate_centroid(pcd)
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=RADIUS)
            sphere.translate(p)
            detected_objects_geomertries += sphere
            detected_objects_geomertries.paint_uniform_color([0.0, 1.0, 0.0])
        o3d.visualization.draw_geometries([pcd_merged, mesh_frame, detected_objects_geomertries])

# ---------------------------------- Others ----------------------------------

def _update_object_state(ini_state, task_name):
    with open(prompt_update_obj_state_file, "r") as template_file:
        template_content = template_file.read()
    values = {"initial_state": ini_state, "task_name": task_name}
    prompt = template_content.format(**values)
    response = query_LLM(prompt, [], "cache/llm_update_obj_name.pkl")
    new_obj_state = response.text
    return new_obj_state

def segment_image(image):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("Loading SAM model...")
    init_time = time.time()
    sam = sam_model_registry["vit_h"](checkpoint="cache/sam_vit_h_4b8939.pth")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32, 
        pred_iou_thresh=0.95,
        stability_score_thresh=0.95,
        box_nms_thresh=0.2,
        crop_n_layers=1,
        crop_overlap_ratio=0.6,
        min_mask_region_area=100
    )
    print("time taken:", time.time() - init_time)
    print("Generating masks...")
    sam.to(device="cuda")
    masks = mask_generator.generate(image)
    masks = sorted(masks, key=(lambda x: x['area']), reverse=False)
    print("Processing masks...")
    init_time = time.time()
    mask_images = []
    for i, mask in enumerate(masks):
        duplicate = False
        for prev_mask in masks[:i]:
            intersection = np.sum(mask['segmentation'] & prev_mask['segmentation'])
            union = np.sum(mask['segmentation'] | prev_mask['segmentation'])
            if intersection / union > 0.1:
                duplicate = True
                break
        if duplicate:
            continue
        x, y, w, h = map(int, mask['bbox'])
        background = 255
        im = np.ones((h, w, 3), dtype=np.uint8) * background
        new_mask = mask['segmentation'][y:y+h, x:x+w]
        im[new_mask] = image[y:y+h, x:x+w, :][new_mask]
        im = np.pad(im, ((3, 3), (3, 3), (0, 0)), 'constant', constant_values=background)
        # im = image[y:y+h, x:x+w, :]
        mask_images.append(im)

def save_current_image(directory_path, visualize=False):
    if REAL_ROBOT:
        pass

def test_image(text):
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k')
    files = ['cache/image_for_retrieval/cup1.png', 'cache/image_for_retrieval/cup2.png', 'cache/image_for_retrieval/cup3.png', 'cache/image_for_retrieval/cup4.png', 'cache/image_for_retrieval/cup5.png']
    images = [clip_preprocess(Image.open(file)).to(device).to(torch.float) for file in files]
    preprocessed_images = torch.stack(images)
    clip_model = clip_model.to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = clip_model.encode_image(preprocessed_images)
    tokenizer = open_clip.get_tokenizer('ViT-g-14')
    text = tokenizer(f"A photo of {text}.").to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_feature = clip_model.encode_text(text).squeeze()
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
    for i in image_features:
        i /= i.norm(dim=-1, keepdim=True)
        raw_probs = i.dot(text_feature).item()
        print(raw_probs)
    return raw_probs