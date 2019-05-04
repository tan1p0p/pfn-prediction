import glob
import json
import os

import chainer
import numpy as np
from PIL import Image

from modules.accuracy import get_h_w_labels
from modules.network import GeneratingStage, RefinementStage

def main():
    # load images
    dir_path = os.path.abspath('./data/hands')
    file_list = []
    for extension in ['jpg', 'jpeg', 'png']:
        file_list.append(glob.glob('{}/*.{}'.format(dir_path, extension)))
    file_list = sum(file_list, [])

    image_tensor = np.zeros((len(file_list), 3, 224, 224), dtype='float32')
    for idx, image_path in enumerate(file_list):
        pil_image = Image.open(image_path).convert('RGB')
        np_image = np.asarray(pil_image.resize((224, 224)))
        image_tensor[idx] = np.moveaxis(np_image, 2, 0)

    # load networks
    first_stage = GeneratingStage()
    second_stage = RefinementStage()
    third_stage = RefinementStage()
    chainer.serializers.load_npz('./model/stage1.npz', first_stage)
    chainer.serializers.load_npz('./model/stage2.npz', second_stage)
    chainer.serializers.load_npz('./model/stage3.npz', third_stage)
    networks = [first_stage, second_stage, third_stage]

    # predict
    batch_size = 2
    h_list = []
    w_list = []
    for x in np.array_split(image_tensor, image_tensor.shape[0] / batch_size):
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            output_1 = networks[0](x)
            output_2 = networks[1](x, output_1)
            output_3 = networks[2](x, output_2)
            h, w = get_h_w_labels(output_3.array)
            h_list.append(h)
            w_list.append(w)

    h_np = np.array(h_list).reshape((-1, 16)) / 56 * 600
    w_np = np.array(w_list).reshape((-1, 16)) / 56 * 600

    pos_list = np.array([[w_np[i], h_pos] for i, h_pos in enumerate(h_np)])
    pos_list = np.array([h_w.T for h_w in pos_list], dtype=np.int32)

    out_json = {
        'fileList': file_list,
        'posList': pos_list.tolist()
    }

    json_file = open('./pred/predicted_label.json','w')
    json.dump(out_json, json_file)
    print('save predicted data to ./pred/predicted_label.json')

if __name__ == "__main__":
    main()