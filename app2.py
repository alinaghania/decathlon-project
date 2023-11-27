#Import all the library
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

#Variables to use to calculates angles 
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Function to display image
def display_image(image_path):
    image = Image.open(image_path)
    st.image(image, use_column_width=True)
    
#Calculate Angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

DEMO_VIDEO = 'demo.mp4'
DEMO_IMAGE = 'demo.jpg'

my_list = []

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

#ANGLES IN IMAGE 
def calculate_angles_in_image(image_path):
    cap = cv2.VideoCapture(image_path)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            try:
                landmarks = results.pose_landmarks.landmark

                # Obtain landmark coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                # Calculate angles
                angle_acromion_coude_poignet = calculate_angle(shoulder, elbow, wrist)
                angle_hanche_genou_cheville = calculate_angle(hip, knee, ankle)
                angle_bassin_acromion_poignet = calculate_angle(hip, shoulder, wrist)
                angle_acromion_bassin_cheville = calculate_angle(shoulder, hip, ankle)
                angle_dos = calculate_angle(hip, shoulder, [hip[0], (hip[1] + shoulder[1]) / 2])

                # Display angles
                st.text(f"Angle bras: {round(angle_acromion_coude_poignet, 2)}")
                st.text(f"Angle jambe: {round(angle_hanche_genou_cheville, 2)}")
                st.text(f"Angle Bras/buste: {round(angle_bassin_acromion_poignet, 2)}")
                st.text(f"Angle tronc: {round(angle_acromion_bassin_cheville, 2)}")
                st.text(f"Angle dos: {round(angle_dos, 2)}")
                
                # Draw pose skeleton
                mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
            except:
                pass

            st.image(image, channels="RGB", use_column_width=True)

    cap.release()
    cv2.destroyAllWindows()    
    
        
def runonvideo():
    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Upload a video")
    
    if record:
        st.checkbox("Video", value=True)

        st.sidebar.markdown('---')
        sameer=""
        st.markdown(' ## Output')
        st.markdown(sameer)

        stframe = st.empty()
        video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
        tfflie = tempfile.NamedTemporaryFile(delete=False)

        if not video_file_buffer:
            if use_webcam:
                vid = cv2.VideoCapture(0)
            else:
                vid = cv2.VideoCapture(DEMO_VIDEO)
                tfflie.name = DEMO_VIDEO

        else:
            tfflie.write(video_file_buffer.read())
            vid = cv2.VideoCapture(tfflie.name)

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(vid.get(cv2.CAP_PROP_FPS))

        codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
        out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

        st.markdown("<hr/>", unsafe_allow_html=True)

        st.sidebar.markdown('---')
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 400px;
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 400px;
                margin-left: -400px;
            
            </style>
            """,
            unsafe_allow_html=True,
        )
        finger_tips = [8, 12, 16, 20]
        thumb_tip = 4
        
        st.markdown('### Angles détectés')  # Titre initial pour les angles
        angle_texts = [st.empty() for _ in range(5)]  # Création de 5 éléments vides pour afficher les angles

        while use_webcam:
            ret, img = vid.read()
            img = cv2.flip(img, 1)
            h, w, c = img.shape
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img)

            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Draw pose skeleton
                mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                mid_hip = [(hip[0] + hip[0]) / 2, (hip[1] + hip[1]) / 2]

                # Calculer les angles
                angle_acromion_coude_poignet = calculate_angle(shoulder, elbow, wrist)
                angle_hanche_genou_cheville = calculate_angle(hip, knee, ankle)
                angle_bassin_acromion_poignet = calculate_angle(hip, shoulder, wrist)
                angle_acromion_bassin_cheville = calculate_angle(shoulder, hip, ankle)
                angle_dos = calculate_angle(hip, mid_hip, shoulder)

                ## Mettre à jour les valeurs des angles
                angle_texts[0].text(f"Angle bras: {int(angle_acromion_coude_poignet)}")
                angle_texts[1].text(f"Angle jambe: {int(angle_hanche_genou_cheville)}")
                angle_texts[2].text(f"Angle Bras/buste: {int(angle_bassin_acromion_poignet)}")
                angle_texts[3].text(f"Angle tronc: {int(angle_acromion_bassin_cheville)}")
                angle_texts[4].text(f"Angle dos: {int(angle_dos)}")

              # Update Streamlit UI with the processed frame
                stframe.image(img, channels="BGR")

            # Check if the "Use Webcam" button has been dis
            if not use_webcam:
                break

        vid.release()
        out.release()
            

# Fonction principale de l'application
def main():
    st.title('Decathlon dynamic bike pose detection')
    st.sidebar.title('Decathlon dynamic bike pose detection')

    app_mode = st.sidebar.selectbox('Choose the App mode', ['About App', 'Run on Image', 'Run on Video'])

    if app_mode =='About App':
        st.sidebar.image("/Users/alina/Desktop/M2-S10/DECAT_PROJECT/Pictures/lll.png", use_column_width=True)
        st.sidebar.markdown('<div style="text-align:center;"><h2>About App</h2></div>', unsafe_allow_html=True)
        
        st.image("/Users/alina/Desktop/M2-S10/DECAT_PROJECT/Pictures/Decathlon-logo.png", use_column_width=True)
        st.markdown('<div style="text-align:center;"><h2>Here are a few angles to calculate to find out if your position is correct</h2></div>', unsafe_allow_html=True)
        st.image("/Users/alina/Desktop/M2-S10/DECAT_PROJECT/Pictures/Angles.png", use_column_width=True)
        
    elif app_mode == 'Run on Image':
        st.sidebar.image("/Users/alina/Desktop/M2-S10/DECAT_PROJECT/Pictures/lll.png", use_column_width=True)
        st.title('Pose Angles Calculator')
        uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            temp_image_path = tempfile.NamedTemporaryFile(delete=False)
            temp_image_path.write(uploaded_file.read())
            st.success("Image uploaded successfully!")

            if st.button("Calculate Angles"):
                calculate_angles_in_image(temp_image_path.name)
                st.success("Angles calculated! Check 'calculate_angles.csv' file.")
                
    elif app_mode == 'Run on Video':
        runonvideo()

if __name__ == "__main__":
    main()
        
        
        