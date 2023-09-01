import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image

#############################
#       Requirements        #
#############################

# REQUIRES OPENCV 4.8.0
# If using an earlier version, please change
"""cv2.VideoWriter.fourcc(*"mp4v")"""
# to
"""cv2.VideoWriter_fourcc(*"mp4v")"""

# REQUIRES FFMPEG (or suitable codec) IF WRITE_FLAG IS TRUE

######################
#       Flags        #
######################

DEBUG_DISPLAY_FLAG = False  # For HSV and mask output
OUTPUT_DISPLAY_FLAG = True  # For published image
WRITE_FLAG = False  # To write video to see accuracy

##############################################
#       Generic Functions and Classes        #
##############################################


def get_HSV_Value(RGB_Value):
    """
    Convert tuple of RGB to tuple of HSV
    """
    RGB_Pixel = np.array(RGB_Value).reshape((1, 1, 3))
    RGB_Pixel = RGB_Pixel.astype(np.uint8)
    HSV_Pixel = cv2.cvtColor(RGB_Pixel, cv2.COLOR_BGR2HSV)
    return HSV_Pixel.reshape((3,))


class Color:
    """
    Class to store color name and (R,G,B)
    """

    def __init__(self, name, RGB_Value) -> None:
        self.NAME = name
        self.RGB_VALUE = RGB_Value


def drawBB(img, cnts, color: Color):
    for c in cnts:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        cv2.rectangle(img, (x, y), (x + w, y + h), color.RGB_VALUE, 3)
        cv2.putText(img, color.NAME, (x + 10, y - 10), 0, 0.7, color.RGB_VALUE)


RED = Color("red", (0, 0, 255))
GREEN = Color("green", (0, 255, 0))
BLACK = Color("black", (0, 0, 0))
BLUE = Color("turntable", (255, 0, 0))

#############################
#       Output Video        #
#############################

frame_width = 1024
frame_height = 768

# # Used for YOLO training
# rawVid = cv2.VideoWriter(
#     str("raw.mp4"),
#     cv2.VideoWriter.fourcc(*"mp4v"),
#     20,
#     (frame_width, frame_height),
# )

outVid = cv2.VideoWriter(
    str("detector_threshold.mp4"),
    cv2.VideoWriter.fourcc(*"mp4v"),
    20,
    (frame_width, frame_height),
)

###################################
#       Basic CV Functions        #
##################################


def apply_CLAHE(img, gridsize=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def filterByExtent(cnts, extentRange):
    """
    Filters cnts by ratio of major axis to minor axis obtained by fitEllipse.
    """
    low, high = extentRange
    output = []
    for cnt in cnts:
        if len(cnt) < 5:
            continue
        _, (ma, MA), _ = cv2.fitEllipse(cnt)
        if low <= MA / ma <= high:
            output.append(cnt)
    return output


def detectRegion(img, hsv, includeRanges, excludeRanges, kernelSize, color: Color):
    # First find the union of all pixels within each range in includeRanges
    # Then, among the pixels found, exclude the union of all pixels in excludeRanges
    mask = cv2.inRange(hsv, *includeRanges[0])
    for range in includeRanges[1:]:
        newMask = cv2.inRange(hsv, *range)
        mask = cv2.bitwise_or(mask, newMask)
    for range in excludeRanges:
        newMask = cv2.inRange(hsv, *range)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(newMask))

    # Morphological operations fill in holes and remove grains
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Removes misdetections of white axles of turntable, which are long and thin
    cnts = filterByExtent(cnts, (1, 3))
    # Get the 2 largest contours
    cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)[:2]

    cv2.drawContours(img, cnts, -1, color.RGB_VALUE, 3)
    drawBB(img, cnts, color)

    if DEBUG_DISPLAY_FLAG:
        cv2.drawContours(hsv, cnts, -1, color.RGB_VALUE, 3)
        drawBB(hsv, cnts, color)
        cv2.imshow("mask", mask)

    return cnts


####################################
#       ROS2 Detection Node        #
####################################


class Detector(Node):
    def __init__(self):
        super().__init__("detector")
        self.pub_debug_img = self.create_publisher(Image, "/detected/debug_img", 10)
        self.sub_image_feed = self.create_subscription(
            CompressedImage,
            "/auv/bot_cam/image_color/compressed",
            self.image_feed_callback,
            10,
        )
        self.bridge = CvBridge()

    def image_feed_callback(self, msg):
        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg)
        # cv_img = apply_CLAHE(cv_img, 32)
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

        # print(cv_img.shape)

        # # Used for YOLO training
        # rawVid.write(cv_img)

        ###################################
        #       Detect Red Regions        #
        ###################################

        # detectRegion(
        #     img=cv_img,
        #     hsv=hsv,
        #     range1=((160, 50, 50), (180, 255, 255)),
        #     color=(0, 0, 255),
        #     range2=((140, 10, 180), (170, 70, 215)),
        # )
        redCnts = detectRegion(
            img=cv_img,
            hsv=hsv,
            includeRanges=[((160, 50, 50), (180, 255, 255))],
            excludeRanges=[],
            kernelSize=7,
            color=RED,
        )

        #####################################
        #       Detect Green Regions        #
        #####################################

        greenCnts = detectRegion(
            img=cv_img,
            hsv=hsv,
            includeRanges=[((100, 60, 50), (115, 145, 255))],
            excludeRanges=[
                ((110, 40, 45), (135, 120, 225)),  # Exclude black region
                ((90, 105, 130), (165, 120, 150)),  # Exclude white part of turntable
            ],
            kernelSize=7,
            color=GREEN,
        )

        #####################################
        #       Detect Black Regions        #
        #####################################

        blackCnts = detectRegion(
            img=cv_img,
            hsv=hsv,
            includeRanges=[((110, 40, 45), (135, 120, 225))],
            excludeRanges=[
                ((90, 105, 130), (165, 120, 150))  # Exclude white part of turntable
            ],
            kernelSize=7,
            color=BLACK,
        )

        #################################
        #       Detect Turntable        #
        #################################

        # Draw a bounding box that contains all detected contours
        height, width, _ = cv_img.shape
        min_x, min_y = width, height
        max_x = max_y = 0
        for cnt in redCnts + greenCnts + blackCnts:
            (x, y, w, h) = cv2.boundingRect(cnt)
            min_x, max_x = min(x, min_x), max(x + w, max_x)
            min_y, max_y = min(y, min_y), max(y + h, max_y)

        cv2.rectangle(cv_img, (min_x, min_y), (max_x, max_y), BLUE.RGB_VALUE, 3)
        cv2.putText(cv_img, BLUE.NAME, (x + 10, y - 10), 0, 0.7, BLUE.RGB_VALUE)

        cv2.rectangle(hsv, (min_x, min_y), (max_x, max_y), BLUE.RGB_VALUE, 3)
        cv2.putText(hsv, BLUE.NAME, (x + 10, y - 10), 0, 0.7, BLUE.RGB_VALUE)

        ########################
        #       Display        #
        ########################

        if DEBUG_DISPLAY_FLAG:
            cv2.imshow("HSV", hsv)
        if OUTPUT_DISPLAY_FLAG:
            cv2.imshow("Output", cv_img)
        if DEBUG_DISPLAY_FLAG or OUTPUT_DISPLAY_FLAG:
            key = cv2.waitKey(1)
            if key == ord("p") or key == ord("P"):
                while True:
                    key = cv2.waitKey(0)
                    if key == ord("p") or key == ord("P"):
                        break

        if WRITE_FLAG:
            outVid.write(cv_img)

        img_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")

        self.pub_debug_img.publish(img_msg)
        self.get_logger().info("Published image.")


def main(args=None):
    rclpy.init(args=args)
    detector = Detector()

    if not WRITE_FLAG:
        rclpy.spin(detector)
    else:
        try:
            rclpy.spin(detector)
        except KeyboardInterrupt:
            outVid.release()
            # # Used for YOLO training
            # rawVid.release()
            cv2.destroyAllWindows()

    # Below lines are not strictly necessary
    detector.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
