from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from transformers.image_utils import ImageFeatureExtractionMixin
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  


def clip_with_owl(image, num_objs, sam, obj_name='drawer', visualize=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    predictor = SamPredictor(sam)

    img = image.copy()
    img = Image.fromarray(np.uint8(img)).convert("RGB")
    text_queries = [f"a photo of a {obj_name}"]
    inputs = processor(text=text_queries, images=img, return_tensors="pt").to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    mixin = ImageFeatureExtractionMixin()

    # Load example image
    image_size = model.config.vision_config.image_size
    original_image = img.copy()
    img = mixin.resize(img, image_size)
    input_image = np.asarray(img).astype(np.float32) / 255.0

    # Threshold to eliminate low probability predictions
    score_threshold = 0.01

    # Get prediction logits
    logits = torch.max(outputs["logits"][0], dim=-1)
    scores = torch.sigmoid(logits.values).cpu().detach().numpy()

    # Get prediction labels and boundary boxes
    labels = logits.indices.cpu().detach().numpy()
    boxes = outputs["pred_boxes"][0].cpu().detach().numpy()
    if visualize:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(input_image, extent=(0, 1, 1, 0))
        ax.set_axis_off()

    # s_list = []
    # for i in range(len(scores)):
    #     box = boxes[i]
    #     cx, cy, w, h = box
    #     s_list.append(w*h)
    # s_order = np.argsort(np.array(s_list))[::-1]
    # boxes = boxes[s_order]
    # scores = scores[s_order]
    # labels = labels[s_order]
    # boxes = boxes[:5]
    # scores = scores[:5]
    # labels = labels[:5]

    order = np.argsort(scores)[::-1]
    boxes = boxes[order]
    scores = scores[order]
    labels = labels[order]
    
    img = np.asarray(img)
    original_image = np.asarray(original_image)
    height, width = img.shape[:2]
    old_height, old_width = original_image.shape[:2]
    h_ratio = old_height/height
    w_ratio = old_width/width

    # ret_points = []
    # p3d = point3d.copy()
    # p3d = np.reshape(p3d, (old_height, old_width, -1))
    masks = []
    ret_boxes = []
    for i in range(num_objs):
        label = labels[i]
        score = scores[i]
        box = boxes[i]
        cx, cy, w, h = box
        if visualize:
            ax.plot([cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2],
                    [cy-h/2, cy-h/2, cy+h/2, cy+h/2, cy-h/2], "r")
            ax.text(
                cx - w / 2,
                cy + h / 2 + 0.015,
                f"{text_queries[label]}: {score:1.2f}",
                ha="left",
                va="top",
                color="red",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "red",
                    "boxstyle": "square,pad=.3"
                })

        cx *= width
        w *= width
        cy *= height
        h *= height

        sam_box = np.array([int((cx-w/2)*w_ratio), int((cy-h/2)*h_ratio), int((cx+w/2)*w_ratio), int((cy+h/2)*h_ratio)])
        predictor.set_image(original_image)
        mask, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=sam_box[None, :],
            multimask_output=False,
        )
        masks.append(mask[0].copy())
        ret_boxes.append((int((cx-w/2)*w_ratio), int((cy-h/2)*h_ratio), int(w*w_ratio), int(h*h_ratio)))
        if visualize:
            plt.show()
        
            plt.figure(figsize=(10, 10))
            plt.imshow(original_image)
            show_mask(mask[0], plt.gca())
            show_box(sam_box, plt.gca())
            plt.axis('off')
            plt.show()

        # ret_points.append(p3d[int((cy-h/2)*h_ratio): int((cy+h/2)*h_ratio), int((cx-w/2)*w_ratio): int((cx+w/2)*w_ratio)])

    return masks, ret_boxes