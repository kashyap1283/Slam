#!/usr/bin/env python3
from platform import node

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')   #initialise the node with name 'image_subscriber'
        self.bridge = CvBridge()               #create a bridge object to convert ROS image messages to OpenCV format
        self.subscription = self.create_subscription(Image,
            '/serf01/nav_rgbd_1/rgb/image_raw', self.listener_callback, 10)   #subscribe to the topic '/serf/rgb/image_raw' with message type Image and callback function listener_callback, queue size 10
        self.depth_sub = self.create_subscription(Image,
            '/serf01/nav_rgbd_1/depth/image_raw', self.depth_subscription, 10)  
        

    #def listener_callback(self,msg):
     #  frame=self.bridge.imgmsg_to_cv2(msg,'bgr8')  #convert the ROS image message to OpenCV format using the bridge object "bgr8 for 8 bits per channel and BGR color order"
      # cv2.imshow("Image",frame)                    #display the image in a window named "Image" using OpenCV
       #cv2.waitKey(1) 

    def depth_subscription(self,msg):
       self.depth=self.bridge.imgmsg_to_cv2(msg,desired_encoding="passthrough")  #convert the ROS image message to OpenCV format using the bridge object "bgr8 for 8 bits per channel and BGR color order"
    #    print("Frame shape:", display.shape)
    #    cv2.imshow("Image2",display)                    #display the image in a window named "Image" using OpenCV
    #    cv2.waitKey(1)   

    def listener_callback(self,msg):
        frame=self.bridge.imgmsg_to_cv2(msg,'bgr8') 
        orb=cv2.ORB_create()                        #create an ORB feature detector object
        kp = orb.detect(frame,None)                      #detect keypoints in the input image using the ORB feature detector
        kp, des = orb.compute(frame, kp)                   #compute the descriptors for the detected keypoints using the ORB feature detector  
        img=cv2.drawKeypoints(frame,kp,None,color=(0,255,0),flags=0)  #draw the detected keypoints on the original image using OpenCV
        cv2.imshow("ORB Features",img)              #display the image with detected keypoints in a window named "ORB Features" using OpenCV
        cv2.waitKey(1)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  #create a brute-force matcher object with Hamming distance and cross-checking enabled
        matches = bf.match(des, des)  #match the descriptors of the detected keypoints against themselves using the brute-force matcher
        matches = sorted(matches, key = lambda x:x.distance)
        img3 = cv2.drawMatches(frame, kp, frame, kp, matches[:10], None, flags=2)  #draw the top 10 matches between the keypoints on the original image using OpenCV
        cv2.imshow("ORB Matches", img3)  #display the image with matches in a window named "ORB Matches" using OpenCV
        cv2.waitKey(1)


        for keypoint in kp:
            x,y=keypoint.pt #.pt gives the (x,y) coordinates of the keypoint
            x=int(x) #float to int conversion for indexing the depth image (column)
            y=int(y) #row
            distance = self.depth[y,x] #indexing works row column (y,x)
            text =f"{distance/1000:.1f} m" 
            cv2.putText(img,text,[x,y],cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1) 
            
        cv2.imshow("Depth at Keypoints", img) 
        cv2.waitKey(1)


def main():     
    rclpy.init()
    node=ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()