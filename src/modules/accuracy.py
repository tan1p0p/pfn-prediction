import cv2

def get_h_w_labels(labels):
    [cv2.GaussianBlur(label,(5,5),0) for label in labels]

    labels_flat = labels.reshape((labels.shape[0] * labels.shape[1], labels.shape[2], labels.shape[3]))
    labels_h = labels_flat.argmax(1).argmax(1)
    labels_w = labels_flat.argmax(2).argmax(1)
    return labels_h, labels_w

def joint_accuracy(y, t):
    image_num, joint_num, height, width = y.shape

    y_h, y_w = get_h_w_labels(y.array)
    t_h, t_w = get_h_w_labels(t)
    print(y_h, y_w)

    max_dist = (height - 1) ** 2 + (width - 1) ** 2
    return ((y_h - t_h) ** 2 + (y_w - t_w) ** 2).mean() / max_dist
