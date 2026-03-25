#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image,'/serf01/nav_rgbd_1/rgb/image_raw', self.listener_callback, 10)
        self.depth = self.create_subscription(Image,'/serf01/nav_rgbd_1/depth/image_raw', self.listener_callback_depth, 10)

    def listener_callback(self,msg):
        frame=self.bridge.imgmsg_to_cv2(msg,'bgr8')

        orb = cv2.ORB_create()
        kp = orb.detect(frame,None)
        kp, des = orb.compute(frame,kp)
        img = cv2.drawKeypoints(frame, kp, None, color=(0,255,0),flags=0)
        # cv2.imshow("Image",img)
        # cv2.waitKey(1)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des,des)
        matches = sorted(matches, key = lambda x:x.distance)
        img1 = cv2.drawMatches(frame,kp,frame,kp,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("ORB",img1)
        cv2.waitKey(1)

        for keypoints in kp:
            x,y = keypoints.pt
            int_x = int(x)
            int_y = int(y)
            distance = self.display[int_y,int_x]
            text = f"{distance/1000}"
            cv2.putText(img,text,(int_x,int_y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)

        cv2.imshow("Distance",img)
        cv2.waitKey(1)
        
        
    def listener_callback_depth(self,msg):
        self.display=self.bridge.imgmsg_to_cv2(msg,desired_encoding="passthrough")
        # cv2.imshow("depth",self.display)
        # cv2.waitKey(1)


def main():
    rclpy.init()
    node=ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()