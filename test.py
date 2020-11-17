import os, sys
import numpy as np
import cv2
from curve_text_rectification import AutoRectifier

def txt_reader(txt_path):
    if not os.path.exists(txt_path):
        raise FileNotFoundError

    points_list = []
    with open(txt_path, 'r') as f:
        data = f.readlines()
        for item in data:
            item = item.strip('\r\n').strip('\r').strip('\n')
            item = item.replace(' ', '').replace('\t', '')
            info = item.split(',')
            box = [float(i) for i in info]
            points_list.append(box)
    return points_list

if __name__ == '__main__':
    # test an image
    print('begin')

    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Assign the image path')
    parser.add_argument('--txt', type=str, help='Assign the path of .txt to get points',
                        default=None)
    args = parser.parse_args()

    image_path = os.path.abspath(args.image)
    txt_path = os.path.abspath(args.txt)

    if not os.path.exists(image_path):
        raise FileNotFoundError

    with open(image_path, "rb") as f:
        image = np.array(bytearray(f.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError

    points_list = txt_reader(txt_path)

    autoRectifier = AutoRectifier()
    res, visualized_image = autoRectifier.run(image, points_list, interpolation=cv2.INTER_LINEAR,
                                              ratio_width=1.0, ratio_height=1.0,
                                              loss_thresh=5.0)

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_path = os.path.basename(image_path)
    profix = image_path.split('.')[-1]
    basename = image_path[:-(len(profix)+1)]
    cv2.imwrite(os.path.join(save_path, 'vis_{}.jpg'.format(basename)), visualized_image)
    for i, rectified_image in enumerate(res):
        cv2.imwrite(os.path.join(save_path, 'vis_{}_{}.jpg'.format(basename, i+1)), rectified_image)
    print('saved in {}'.format(save_path))
    print('done!')