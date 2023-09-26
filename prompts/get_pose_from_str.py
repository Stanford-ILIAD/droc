import numpy as np
from perception_utils import detect, sort_from_high_to_low
from geometric_utils import calculate_centroid, calculate_major_axis, calculate_minor_axis, calculate_tail_point

# Query: the centroid and minor axis of "top black hat"
black_hats = detect('black hat')
black_hats = sort_from_high_to_low(black_hats, key='z')
top_black_hat = black_hats[0]
top_black_hat.position = calculate_centroid(top_black_hat.pcd)
top_black_hat.rotation = calculate_minor_axis(top_black_hat.pcd)
ret_val = top_black_hat

# Query: the tail point of "cup"
cups = detect('cup')
cups = sort_from_high_to_low(cups, key='probability')
cup = cups[0]
cup.position = calculate_tail_point(cup.pcd)
ret_val = cup

# Query: the major axis of "middle shelf"
shelves = detect('shelf')
shelves = sort_from_high_to_low(shelves, key='z')
middle_shelf = shelves[len(shelves)//2]
middle_shelf.rotation = calculate_major_axis(middle_shelf.pcd)
ret_val = middle_shelf