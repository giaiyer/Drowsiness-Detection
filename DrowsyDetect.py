from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np

# Eye Aspect Ratio (EAR) calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Mouth Aspect Ratio (MAR) calculation
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[13], mouth[19])  # 14-20 in 1-based indexing
    B = distance.euclidean(mouth[14], mouth[18])  # 15-19
    C = distance.euclidean(mouth[15], mouth[17])  # 16-18
    D = distance.euclidean(mouth[12], mouth[16])  # 13-17 (horizontal)
    mar = (A + B + C) / (3.0 * D)
    return mar

# Thresholds and constants
EAR_THRESH = 0.25
EAR_CONSEC_FRAMES = 10
MAR_THRESH = 0.7

# Counters
blink_counter = 0
blink_total = 0
yawn_total = 0
yawn_flag = False  

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/gia/Downloads/CV Project/shape_predictor_68_face_landmarks.dat")

# Get indexes of facial landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# Start video stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the camera
    frame = imutils.resize(frame, width=900)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Mouth
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)

        # Draw contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 255), 1)

        # Blink detection
        if ear < EAR_THRESH:
            blink_counter += 1
            if blink_counter >= EAR_CONSEC_FRAMES:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        else:
            if blink_counter >= 3:
                blink_total += 1
            blink_counter = 0

        # Yawn detection
        if mar > MAR_THRESH:
            if not yawn_flag:  
                yawn_total += 1
                yawn_flag = True  
            cv2.putText(frame, "YAWNING!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else:
            yawn_flag = False  

        # Display stats on the right side
        frame_width = frame.shape[1]  
        cv2.putText(frame, f"Blinks: {blink_total}", (frame_width - 200, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Yawns: {yawn_total}", (frame_width - 200, 490),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # Show frame
    cv2.imshow("Drowsiness Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()