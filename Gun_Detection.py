import numpy as np
import sys
import tensorflow as tf
import cv2
import time


sys.path.append('..')

from utils import label_map_util

cap = cv2.VideoCapture("./test_images/video.mp4")
status = False
count =0

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
labels_path = "./training/labelmap.pbtxt"


ckpt_path = './inference_graph/frozen_inference_graph.pb'
NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(ckpt_path, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(labels_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

with detection_graph.as_default():
    print ("Graph loaded")
    with tf.Session(graph=detection_graph) as sess:
        print ("Session Loaded")
        
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
       
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        # Loading images using openCV imread method
        #frame = cv2.imread("/root/Anton/tensorflow-repo/models/research/object_detection/test_images/image1.jpg")
        
        #Determining the loaded image height and width
        cur_frames = 0
        track_count = 0
        cv2.namedWindow("Camera frame",cv2.WINDOW_NORMAL);
        while True:
            
            ret,frame = cap.read()
            
            
            bounding_box = (0,0,0,0)
            if ret == True:
                cur_frames+=1
                if not cur_frames % 3 == 0:
                    continue
                
                img_height, img_width, channels = frame.shape
                
                image_np_expanded = np.expand_dims(frame, axis=0)
                
            
                
                
                det_start_time = time.time()
                (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                det_stop_time = time.time()
                print ("Detection rate: ",det_stop_time-det_start_time)
                
                if(len(classes) > 0):
                    
                    object_index = int(classes[0][0])
                    
                    y1 = int(boxes[0][0][0]*img_height)
                    x1 = int(boxes[0][0][1]*img_width)
                    y2 = int(boxes[0][0][2]*img_height)
                    x2 = int(boxes[0][0][3]*img_width)
                    p1 = (x1,y1)
                    p2 = (x2,y2)
                    
                    if(object_index == 1 and scores[0][0]*100 > 70):
                        cv2.rectangle(frame,p1,p2,(0,255,0),3)
                        
                        print ("Detection Done")
                        
                        
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                cv2.imshow("Camera frame", frame)    
            
            else:
                print("Image not loaded for Tracking and Detection")
                
            
        cv2.destroyAllWindows()
