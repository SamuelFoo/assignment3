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

# REQUIRES ULTRALYTICS IF TRAINING

# WANTS CUDA ENABLED FOR FAST INFERENCING

######################
#       Flags        #
######################

OUTPUT_DISPLAY_FLAG = True  # For published image
WRITE_FLAG = False  # To write video to see accuracy

##############################################
#       Generic Functions and Classes        #
##############################################


class Color:
    """
    Class to store color name and (R,G,B)
    """

    def __init__(self, name, RGB_Value) -> None:
        self.NAME = name
        self.RGB_VALUE = RGB_Value


RED = Color("red", (0, 0, 255))
GREEN = Color("green", (0, 255, 0))
BLACK = Color("black", (0, 0, 0))
BLUE = Color("turntable", (255, 0, 0))

COLORS = [RED, GREEN, BLACK, BLUE]


#############################
#       Output Video        #
#############################

frame_width = 1024
frame_height = 768

outVid = cv2.VideoWriter(
    str("detector_yolo.mp4"),
    cv2.VideoWriter.fourcc(*"mp4v"),
    20,
    (frame_width, frame_height),
)


#####################################
#       Process ONNX Network        #
#####################################
# Taken from
# https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-OpenCV-ONNX-Python/main.py


model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(
    "/home/aorus/ros2_ws/src/assignment3/YOLO/weights/yolov8n_010923_1.onnx"
)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{COLORS[class_id].NAME} ({confidence:.2f})"
    color = COLORS[class_id].RGB_VALUE
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(
        img, COLORS[class_id].NAME, (x + 10, y - 10), 0, 0.7, COLORS[class_id].RGB_VALUE
    )


def get_ONNX_Detections(original_image):
    [height, width, _] = original_image.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / 640

    blob = cv2.dnn.blobFromImage(
        image, scalefactor=1 / 255, size=(640, 640), swapRB=True
    )
    model.setInput(blob)
    outputs = model.forward()

    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(
            classes_scores
        )
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            "class_id": class_ids[index],
            "class_name": COLORS[class_ids[index]].NAME,
            "confidence": scores[index],
            "box": box,
            "scale": scale,
        }
        detections.append(detection)
        draw_bounding_box(
            original_image,
            class_ids[index],
            scores[index],
            round(box[0] * scale),
            round(box[1] * scale),
            round((box[0] + box[2]) * scale),
            round((box[1] + box[3]) * scale),
        )

    return detections


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

        get_ONNX_Detections(cv_img)

        ########################
        #       Display        #
        ########################

        if OUTPUT_DISPLAY_FLAG:
            cv2.imshow("Output", cv_img)
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
            cv2.destroyAllWindows()

    # Below lines are not strictly necessary
    detector.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
