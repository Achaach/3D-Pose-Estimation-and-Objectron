import os
import numpy as np
import cv2
import mediapipe as mp
from utils import *

def detect_3d_box(img_path):

    '''
        Args:
        -    img_path: the path of the RGB chair image

        Returns:
        -

        boxes: numpy array of 2D points, which represents the 8 vertices of 3D bounding boxes
        annotated_image: the original image with the overlapped bounding boxes

    '''

    model_path = 'object_detection_3d_chair.tflite'
    if os.path.exists('../object_detection_3d_chair.tflite'):
        model_path = "../object_detection_3d_chair.tflite"
    elif os.path.exists('../../object_detection_3d_chair.tflite'):
        model_path = "../../object_detection_3d_chair.tflite"

    boxes = None
    hm = None
    displacements = None

    inshapes = [[1, 3, 640, 480]]
    outshapes = [[1, 16, 40, 30], [1, 1, 40, 30]]
    print(inshapes, outshapes)

    if img_path == 'cam':
        cap = cv2.VideoCapture(0)

    while True:
        if img_path == 'cam':
            _, img_orig = cap.read()
        else:
            img_file = img_path
            img_orig = cv2.imread(img_file)

        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (inshapes[0][3], inshapes[0][2]))
        img = img.transpose((2,0,1))
        image = np.array(img, np.float32)/255.0
        # Step-1: Call inference and get the result.
        hm, displacements = inference(image, model_path)
        # Step-2: Decode bounding boxes from inference result .
        boxes = decode(hm, displacements)
        # Draw the bounding box.
        for obj in boxes:
            draw_box(img_orig, obj)
        return boxes[0], cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)


def hand_pose_img(test_img):
    """
        Args:
        -    img: path to rgb image

        Returns:
        -    landmark: numpy array of size (n, 2) the landmark detected by mediapipe,
        where n is the length of landmark, 2 represents x and y coordinates
        (Note, not in the range 0-1, you need to get the real 2D coordinates in images)

        the order of these landmark should be consistent with the original order returned by mediapipe
        -    annotated_image: the original image overlapped with the detected landmark

        Useful functions/class: mediapipe.solutions.pose, mediapipe.solutions.drawing_utils
    """

    landmark = None
    annotated_image = None

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Init pose model and read the image.
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    image = cv2.imread(test_img)

    # Convert the BGR image to RGB before processing.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rows = image.shape[0]
    cols = image.shape[1]

    # Step-1: Pass the image to the pose model.
    results = pose.process(image)
    # Step-2: Get landmark from the result.
    landmark = results.pose_landmarks

    # Display the pose.
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
      annotated_image, landmark, mp_pose.POSE_CONNECTIONS)
    pose.close()
    landmark1=np.zeros((len(results.pose_landmarks.landmark),2))
    for i in range(len(results.pose_landmarks.landmark)):
        landmark1[i,:] = [results.pose_landmarks.landmark[i].x*cols, results.pose_landmarks.landmark[i].y*rows]
    return landmark1, annotated_image


def check_hand_inside_bounding_box(hand, pts):
    """
    This function checks whether the hand is inside the bounding box of the
    chair or not.
    Args:
        hand: 3D coordinate of the hand (numpy.array, size 1*3)
        pts: 3D coordinates of the 8 vertices of the bounding box (numpy.array, size 8*3)
    Returns:
        inside: boolean value, True if hand is inside the bounding box, and
                False otherwise.

    """

    inside = None

    width = pts[:, 0].max()
    depth = pts[:, 2].max()
    height = pts[:, 1].max()

    if hand[0] >= 0 and hand[0] <= width and hand[1] >= 0 and hand[1] <= height and hand[2] >= 0 and hand[2] <= depth:
        inside = True
    else:
        inside = False

    return inside


def draw_box_intersection(image, hand, pts, pts_2d):
    """
    Draw the bounding box (in blue) around the chair. If the hand is within the
    bounding box, then we draw it with another color (red)
    Args:
        image: the image in which we'll draw the bounding box, the channel follows RGB order
        hand: 3D coordinate of the hand (numpy.array, 1*3)
        pts: 3D coordinates of the 8 vertices of the bounding box (numpy.array, 8*3)
        pts_2d: 2D coordinates of the 8 vertices of the bounding box (numpy.array, 8*2)

    Returns:
        image: annotated image
    """
    if np.shape(pts)[1] == 3:
        pts = np.concatenate([pts, np.ones((8,1))], axis=1)

    color = (0, 0, 0)
    if check_hand_inside_bounding_box(hand, pts):
        color = (0, 255, 255)
        print("Check succeed!")

    thickness = 5

    scaleX = image.shape[1]
    scaleY = image.shape[0]

    lines = [(0, 1), (1, 3), (0, 2), (3, 2), (1, 5), (0, 4), (2, 6), (3, 7), (5, 7), (6, 7), (6, 4), (4, 5)]
    for line in lines:
        pt0 = pts_2d[line[0]]
        pt1 = pts_2d[line[1]]
        pt0 = (int(pt0[0] * scaleX), int(pt0[1] * scaleY))
        pt1 = (int(pt1[0] * scaleX), int(pt1[1] * scaleY))
        cv2.line(image, pt0, pt1, color, thickness)


    for i in range(8):
        pt = pts_2d[i]
        pt = (int(pt[0] * scaleX), int(pt[1] * scaleY))
        cv2.circle(image, pt, 8, (0, 255, 0), -1)
        cv2.putText(image, str(i), pt, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    print(image.shape)
    return image
