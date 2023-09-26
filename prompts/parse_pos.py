from perception_utils import get_directional_vec, get_current_pos
import numpy as np

# Choose direction from "left", "right", "forward", "backward", "up", "down"
{mapping_dict}

# Query: a point a little bit left to current_pos.
current_pos = get_current_pos()
direction = get_directional_vec("left")
ret_val = current_pos + {little_left} * direction

# Query: a point a little bit up to adjusted_pos.
current_pos = get_current_pos()
direction = get_directional_vec("right")
ret_val = current_pos + {little_up} * direction

# Query: a point a little bit forward from [x y z].
current_pos = np.array([x, y, z])
direction = get_directional_vec("forward")
ret_val = current_pos + {little_forward} * direction

# Query: a point more back from current_pos.
current_pos = get_current_pos()
direction = get_directional_vec("back")
ret_val = current_pos + {more_back} * direction