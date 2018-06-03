import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import cv2
from plyfile import PlyData, PlyElement

import models

def predict(model_data_path, image_path):

    
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
   
    # Read image
    img = Image.open(image_path)
    img = img.resize([width,height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)
   
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
        
    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        #net.load(model_data_path, sess) 

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})
        
        # Plot result
        fig = plt.figure()
        ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
        fig.colorbar(ii)
        plt.show()
        
        return pred

def create_point_Cloud(pred):
    # Depth Intrinsic Parameters
    fx_d = 5.8262448167737955e+02
    fy_d = 5.8269103270988637e+02
    cx_d = 3.1304475870804731e+02
    cy_d = 2.3844389626620386e+02

    imgDepthAbs = pred[0,:,:,:]
    imgDepthAbs = cv2.resize(imgDepthAbs, (640,480))
    print("imgDepthAbs shape: ", imgDepthAbs.shape)
    [H, W] = imgDepthAbs.shape
    assert H == 480
    assert W == 640
    
    [xx, yy] = np.meshgrid(range(0, W), range(0, H))
    X = (xx - cx_d) * imgDepthAbs / fx_d
    Y = (yy - cy_d) * imgDepthAbs / fy_d
    Z = imgDepthAbs

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # for c, m, zlow, zhigh in [('r', 'o', -50, -25)]:
    #     ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=Z.flatten(), cmap=plt.cm.coolwarm, marker=m)
    
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')            
    # plt.show()

    points = np.array([X.flatten(), Y.flatten(), Z.flatten()]).transpose()
    points = [tuple(x) for x in points]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    print("Vertex shape: ", vertex.shape)

    ply = PlyData([PlyElement.describe(vertex, 'vertex')], text=True)

    ply.write('point_cloud.ply')   
                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    # Predict the image
    pred = predict(args.model_path, args.image_paths)
    create_point_Cloud(pred)

    os._exit(0)

if __name__ == '__main__':
    main()

        



