import cv2
import dlib
from scipy.spatial import distance

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

drowsiness_duration = 0  # Variable to track drowsiness duration
drowsy_start_time = None  # Variable to store the time when drowsiness started

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    drowsy_detected = False  # Flag to indicate if drowsiness is detected in this frame

    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        for n in range(42,48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)
        if EAR < 0.26:
            if not drowsy_detected:  # Check if drowsiness is newly detected
                drowsy_start_time = cv2.getTickCount() / cv2.getTickFrequency()  # Record the start time of drowsiness
                drowsy_detected = True
            drowsiness_duration = (cv2.getTickCount() / cv2.getTickFrequency()) - drowsy_start_time
            if drowsiness_duration >= 3.0:  # If drowsiness duration exceeds 3 seconds
                cv2.putText(frame,"DROWSY",(20,100),
                    cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),4)
                cv2.putText(frame,"Are you Sleepy?",(20,400),
                    cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
                print("Drowsy")
        else:
            drowsy_detected = False  # Reset the flag if EAR is above the threshold

        print(EAR)

    cv2.imshow("Are you Sleepy", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
