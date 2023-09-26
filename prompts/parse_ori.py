from perception_utils import get_ori, get_horizontal_ori, get_vertical_ori

# Query: tilt up a little bit
ret_val = get_ori(degrees=10, axis='x')

# Query: rotate clockwise 45 degrees
ret_val = get_ori(45, 'z')

# Query: tilt right more
ret_val = get_ori(-30, 'y')

# Query: vertical
ret_val = get_vertical_ori()

# Query: rotate to horizontal
ret_val = get_horizontal_ori()