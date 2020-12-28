#!/usr/bin/env python
from __future__ import print_function, unicode_literals
import rospy, rosnode
from sensor_msgs.msg import Image
import numpy as np
import cv2
import os
import time

SAVE_DIR = "tmp_img"


pt1 = None
pt2 = None
lbottun_down = False
def mouse_event(event, x, y, flags, param):
    global pt1, pt2, lbottun_down
    # 左クリックイベント
    if event == cv2.EVENT_LBUTTONDOWN:
        pt1 = (x, y)
        lbottun_down = True
    elif event==cv2.EVENT_LBUTTONUP:
        lbottun_down = False

    if lbottun_down:
        pt2 = (x, y)

def save_img( img, pt1, pt2 ):
    x1 = min( pt1[0], pt2[0] )
    y1 = min( pt1[1], pt2[1] )
    x2 = max( pt1[0], pt2[0] )
    y2 = max( pt1[1], pt2[1] )

    rect_img = img[y1:y2, x1:x2, :]

    if not os.path.exists( SAVE_DIR ):
        os.mkdir( SAVE_DIR )

    for i in range(100):
        fname = os.path.join( SAVE_DIR, "%03d.png"%i )

        if not os.path.exists(fname):
            cv2.imwrite( fname, rect_img )
            print( "saved as ", fname  )
            break

def image_cb( img ):
    global pt1, pt2
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", mouse_event)

    img = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)
    img_display = cv2.cvtColor( img, cv2.COLOR_RGB2BGR )

    if pt1!=None and pt2!=None:
        cv2.rectangle( img_display, pt1, pt2, (0,0,255), 1 )

    cv2.imshow("img", img_display)
    c = cv2.waitKey(10)

    if c==115: # s
        if pt1!=None and pt2!=None:
            save_img( cv2.cvtColor( img, cv2.COLOR_RGB2BGR ), pt1, pt2 )
        else:
            print("画像をドラッグして保存範囲を選択してください")
    elif c==113: # q
        cv2.destroyAllWindows()
        rosnode.kill_nodes([rospy.get_name() ])


def main():
    rospy.init_node('image_saver', anonymous=True)
    rospy.Subscriber("camera/color/image_raw", Image, image_cb)

    time.sleep(1)
    print( "s:save image, q:quit\r" )
    rospy.spin()
        
if __name__ == '__main__':
    main()