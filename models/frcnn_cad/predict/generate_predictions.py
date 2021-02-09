import numpy as np
import cv2
import pydicom
from collections import Counter
import pickle
import matplotlib.pyplot as plt
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import caffe
import sys
import pandas as pd
from tqdm import tqdm
import argparse

cfg.TEST.SCALES=(1700,)
cfg.TEST.MAX_SIZE=2100
cfg.TEST.HAS_RPN = True

# Need to test CPU mode
def load_net(ptxt, w, use_gpu, device=0):
    """Load model."""
    if use_gpu:
        caffe.set_mode_gpu()
        caffe.set_device(device)
        cfg.GPU_ID = device
        net = caffe.Net(ptxt, w, caffe.TEST)
    else:
        caffe.set_mode_cpu()
        print("using cpu")
        net = caffe.Net(ptxt, w, caffe.TEST)
    return net


# Try resizing before and after scaling
def preprocess_image(image_path):
    img = cv2.imread(image_path, -1)   
   
    img=255.*img/img.max()


    im=np.zeros((img.shape[0], img.shape[1]) + (3,), dtype=np.uint8)
    im[:,:,0], im[:,:,1], im[:,:,2]=img, img, img
    return im

def nyu_preprocess_image(image_path, std, mean):
    parr = cv2.imread(image_path, -1)
    parr=255.*parr/parr.max()
    nonzero_indices = parr != 0
    nonzero_values = parr[nonzero_indices]
    transformed_values = std * ((nonzero_values - np.mean(nonzero_values)) / np.sqrt(np.var(nonzero_values))) + mean
    parr[nonzero_indices] = transformed_values
    im = np.zeros((parr.shape[0], parr.shape[1]) + (3,), dtype=np.uint8)
    im[:,:,0], im[:,:,1], im[:,:,2] = parr, parr, parr
    return im

def eval_net(net,pkl_file,image_path,n=None,NMS_THRESH = 0.1):
    """Evaluate net on all images."""
    
    # Set random number generator
    random_number_generator = np.random.RandomState(0)
 
    # Load information about exams
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    image_indexes = []
    malignant_pred = []
    malignant_label = []    
    # Iterate over exams in data
    for d in tqdm(data):
        for v in ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']:
            if len(d[v]) == 0:
                continue
            else:
                index = random_number_generator.randint(low=0, high=len(d[v]))
                image_id = d[v][index]    
                image_indexes.append(image_id)
                im_path = image_path + '/' + image_id + '.png'
                im = preprocess_image(im_path)
                bboxes_and_scores = score_im(net,im,NMS_THRESH)
                scores = bboxes_and_scores[:, -1]
                max_score = np.max(scores)
                malignant_pred.append(max_score)
            
                if v[0] == 'L':
                    malignant_label.append(d['cancer_label']['left_malignant'])
                else:
                    malignant_label.append(d['cancer_label']['right_malignant'])
      
    # Create pandas dataframe
    df = pd.DataFrame()
    df["image_index"] = image_indexes
    df["malignant_pred"] = malignant_pred
    df["malignant_label"] = malignant_label
   

    return df

def score_im(net,im,NMS_THRESH):
    """Score one image."""
    scores,boxes=im_detect(net,im)
    cls_ind=2
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep,:]
    return dets

def main(pkl_file, image_path, prediction_file, use_gpu):
    net = load_net(ptxt='vgg16_frcnn_cad_test.prototxt',w='/home/frcnn/frcnn_cad/weights/vgg16_frcnn_cad.caffemodel',use_gpu=use_gpu, device=0)
    df = eval_net(net, pkl_file, image_path)
    # Save predictions to csv file
    df.to_csv(prediction_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate frcnn_cad model on a data set")
    parser.add_argument('--input-data-folder', required=True)
    parser.add_argument('--exam-list-path', required=True)
    parser.add_argument('--prediction-file', required=True)
    parser.add_argument('--use-gpu', required=True) 

    args = parser.parse_args()
    if args.use_gpu == 'gpu':
        args.use_gpu = True
    else:
        args.use_gpu = False

    main(args.exam_list_path, args.input_data_folder, args.prediction_file, args.use_gpu)



