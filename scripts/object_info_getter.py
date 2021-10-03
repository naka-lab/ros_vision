#!/usr/bin/env python
from __future__ import print_function, unicode_literals
import rospy
from std_msgs.msg import String
import yaml

def callback(data):
    obj_info = yaml.load(data.data)

    for o in obj_info:
        print( "label:", o["label"] )
        print( "lefttop:", o["lefttop"] )
        print( "rightbottom:", o["rightbottom"] )
        print( "position:", o["position"] )
        #print( "position_mindepth:", o["position_mindepth"] )
        #print( "position_center:", o["position_center"] )
        #print( "position_bottom:", o["position_bottom"] )
        print( "-----------" )
    
def main():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/object_rec/object_info", String, callback)
    #rospy.Subscriber("/ssd_object_rec/object_info", String, callback)
    #rospy.Subscriber("/ar_marker_rec/object_info", String, callback)
    rospy.spin()
        
if __name__ == '__main__':
    main()