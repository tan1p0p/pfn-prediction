import numpy as np

def joint_accuracy(y, t):
    image_num, joint_num, height, width = y.shape

    y_flat = y.array.reshape((image_num * joint_num, height, width))
    y_h = y_flat.argmax(1).argmax(1)
    y_w = y_flat.argmax(2).argmax(1)
    
    t_flat = t.reshape((image_num * joint_num, height, width))
    t_h = t_flat.argmax(1).argmax(1)
    t_w = t_flat.argmax(2).argmax(1)

    max_dist = (height - 1) ** 2 + (width - 1) ** 2
    return ((y_h - t_h) ** 2 + (y_w - t_w) ** 2).mean() / max_dist
