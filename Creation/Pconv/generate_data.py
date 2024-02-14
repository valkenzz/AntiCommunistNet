import argparse
import numpy as np
import random
from PIL import Image

action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]


def random_walk(canvas, ini_x, ini_y, length):
 #   x = ini_x
 #   y = ini_y
 #   img_size = canvas.shape[-1]
 #   x_list = []
 #   y_list = []
 #   for i in range(length):
#        r = random.randint(0, len(action_list) - 1)
   #     x = np.clip(x + action_list[r][0], a_min=0, a_max=img_size - 1)
  #      y = np.clip(y + action_list[r][1], a_min=0, a_max=img_size - 1)
   #     x_list.append(x)
   #     y_list.append(y)
    #canvas[np.array(x_list), np.array(y_list)] = 0
    mask_width=100
    mask_height=100
    width=256
    height=256
    mask_x = random.randint(0, width - mask_width)
    mask_y = random.randint(0, height - mask_height)
    canvas[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width]=0
    return canvas


if __name__ == '__main__':
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='masks')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for i in range(args.N):
        canvas = np.ones((args.image_size, args.image_size)).astype("i")
        ini_x = random.randint(0, args.image_size - 1)
        ini_y = random.randint(0, args.image_size - 1)
        mask = random_walk(canvas, ini_x, ini_y, args.image_size ** 2)
        print("save:", i, np.sum(mask))

        img = Image.fromarray(mask * 255).convert('1')
        img.save('{:s}/{:06d}.jpg'.format(args.save_dir, i))
