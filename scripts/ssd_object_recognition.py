
#!/usr/bin/env python
from __future__ import print_function, unicode_literals
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import cv2
import numpy as np
import tf
import yaml
 
# モデルの中の訓練されたクラス
classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
 
# 物体情報のpublisher
pub_objinfo = None

 
# モデルの読み込み
model = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb',
                                      'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

# 物体情報を送信
def send_objects_info(rects, positions, labels):
    global pub_objinfo
    br = tf.TransformBroadcaster()
    object_info = []
    for i, p in enumerate(positions):
        br.sendTransform(p,
                tf.transformations.quaternion_from_euler(0, 0, 0),
                rospy.Time.now(), "obj:%d,id:%d"%(i, labels[i]), "camera_depth_optical_frame")
        object_info.append(
            { 
                "lefttop" : [ int(i) for i in rects[i][0]],
                "rightbottom" : [ int(i) for i in rects[i][1]],
                "position" : [ float(p) for p in positions[i]],
                "label" : int(labels[i])
             }
        )
    pub_objinfo.publish( yaml.dump( object_info )  )

    print(yaml.dump( object_info ) )


def pointcloud_cb( pc2 ):
    lag = rospy.get_time()-pc2.header.stamp.secs
    if lag>0.5:
        print("discard queue")
        return 

    xyz = np.frombuffer(pc2.data, dtype=np.float32).reshape(pc2.height, pc2.width, 8)[:,:,0:3]
    img = np.frombuffer(pc2.data, dtype=np.uint8).reshape(pc2.height, pc2.width, 32)[:, :,16:19]

    h, w = img.shape[:2]
    
    model.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True))
    output = model.forward()
    detections = output[0, 0, :, :]
    
    # detection=[?,id, confidence, top, left, right, bottom]
    rects = []
    positions = []
    labels = []
    N = 0
    for detection in detections:
        confidence = detection[2]
        if confidence > 0.5:
            idx = detection[1]
            class_name = classNames[idx]
    
            rect = detection[3:7] * (w, h, w, h)
            (left, top, right, bottom) = rect.astype(np.int)[:4]

            pos = xyz[ int((top+bottom)/2) , int((left+right)/2) ]

            print(idx, confidence, class_name, pos )

            labels.append( idx )
            rects.append( ((left, top), (right, bottom)) )
            positions.append( pos )
            N += 1
    
    send_objects_info( rects, positions, labels )

    img_display = np.copy( img )
    for i in range(N):
        cv2.rectangle(img_display, rects[i][0], rects[i][1], (0, 0, 255), thickness=2)
        cv2.putText(img_display, class_name, rects[i][0], cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('image', img_display)    
    cv2.waitKey(10)

def main():
    global pub_objinfo
    rospy.init_node('ssd_object_rec', anonymous=True)
    rospy.Subscriber("/camera/depth_registered/points", PointCloud2, pointcloud_cb, queue_size=1)
    pub_objinfo = rospy.Publisher('/ssd_object_rec/object_info', String, queue_size=1)
    rospy.spin()


if __name__ == '__main__':
    main()