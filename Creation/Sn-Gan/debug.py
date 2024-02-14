import sys
import os
import glob

import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


if __name__ == "__main__":
    # training data
    FLAGS = ng.Config('inpaint.yml')
    img_shapes = FLAGS.img_shapes
    with open(FLAGS.data_flist[FLAGS.dataset][0]) as f:
        fnames = f.read().splitlines()
    c=0
    b=0
    for file in fnames:
        try:
            image = cv2.resize(cv2.imread(file), (10, 10))
            b=b+1
        except:
            print(file)
            c=c+1

print(c)
print(b)
