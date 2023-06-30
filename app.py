# Import library
import cv2
import glob
import numpy as np
from PIL import Image
import streamlit as st

from src.detection_keypoint import DetectKeypoint
from src.classification_keypoint import KeypointClassification

detection_keypoint = DetectKeypoint()
classification_keypoint = KeypointClassification(
    './models/pose_classification.pth'
)

def pose_classification(img, col=None):
    image = Image.open(img)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    # show image col 1
    col1.write("Original Image :")
    col1.image(image_rgb)

    # detection keypoint
    results = detection_keypoint(image_cv)
    results_keypoint = detection_keypoint.get_xy_keypoint(results)

    # classification keypoint
    input_classification = results_keypoint[10:]
    results_classification = classification_keypoint(input_classification)

    # visualize result
    image_draw = results.plot(boxes=False)
    x_min, y_min, x_max, y_max = results.boxes.xyxy[0].numpy()
    image_draw = cv2.rectangle(
                    image_draw, 
                    (int(x_min), int(y_min)),(int(x_max), int(y_max)), 
                    (0,0,255), 2
                )
    (w, h), _ = cv2.getTextSize(
                    results_classification.upper(), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
    image_draw = cv2.rectangle(
                    image_draw, 
                    (int(x_min), int(y_min)-20),(int(x_min)+w, int(y_min)), 
                    (0,0,255), -1
                )
    image_draw = cv2.putText(image_draw,
                    f'{results_classification.upper()}',
                    (int(x_min), int(y_min)-4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255),
                    thickness=2
                )
    image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)
    col2.write("Keypoint Result :wrench:")
    col2.image(image_draw)
    col2.text(f'Pose Classification : {results_classification}')
    return image_draw, results_classification

st.set_page_config(
    layout="wide", 
    page_title="YoloV8 Keypoint Classification"
)
st.write(
    "## YoloV8 Keypoint Yoga Pose Classification"
)
st.write(
    ":dog: Try uploading an image to Classification Yoga Basic Pose like a Downdog, Goddess, Plank, Tree, Warrior2 :grin:"
)
st.sidebar.write(
    "## Upload Image :gear:"
)

col1, col2 = st.columns(2)
img_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if img_upload is not None:
    pose_classification(img=img_upload)

# show sample image
st.write('## Sample Image')
images = glob.glob('./images/*.jpeg')
row_size = len(images)
grid = st.columns(row_size)
col = 0
for image in images:
    with grid[col]:
        st.image(f'{image}')
        st.button(label='RUN', key=f'run_{image}',
                  on_click=pose_classification, args=(image, 'run'))
    col = (col + 1) % row_size

