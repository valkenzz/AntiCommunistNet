import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel

from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')


if __name__ == "__main__":
    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(1)
    args, unknown = parser.parse_known_args()



    mypath=args.image
    
    onlyfiles = [join(mypath, f) for f in listdir(mypath)]

    mymask=args.mask
    
    onlymask = [join(mymask, f) for f in listdir(mymask)]
    
    myRestult=args.output
    
    nom=[join(myRestult, f) for f in listdir(mypath)]
    
    for i in range(len(onlymask)):
        
    
        model = InpaintCAModel()
        image = cv2.imread(onlyfiles[i])
        mask = cv2.imread(onlymask[i])
        # mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)
        image=cv2.resize(image, (256,256), interpolation = cv2.INTER_AREA)
    
        assert image.shape == mask.shape
    
        h, w, _ = image.shape
        grid = 8
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]
        print('Shape of image: {}'.format(image.shape))
    
        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)
    
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            input_image = tf.constant(input_image, dtype=tf.float32)
            output = model.build_server_graph(FLAGS, input_image, reuse=tf.AUTO_REUSE)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)
            # load pretrained model
            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
                assign_ops.append(tf.assign(var, var_value))
            sess.run(assign_ops)
            print('Model loaded.')
            result = sess.run(output)
            
            cv2.imwrite(nom[i], result[0][:, :, ::-1])
            
