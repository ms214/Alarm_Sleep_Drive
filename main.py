import cv2
import mediapipe as mp
import copy
import numpy as np
from queue import Queue
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
queue = []
cnt = 0
isSleep = False
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5, max_num_faces=2) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    debug_image = copy.deepcopy(image)
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        brect = calc_bounding_rect(debug_image, face_landmarks)
        #print(brect)
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []
        landmark_point2 = []

        for index, landmark in enumerate(face_landmarks.landmark):
            if landmark.visibility < 0 or landmark.presence < 0:
                continue

            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point.append((landmark_x, landmark_y))
            landmark_point2.append(landmark.z*100)

        # middle point of lips
        #print(landmark_point2[13])

        if(len(queue) < 100):
            queue.append(landmark_point2[13])

        else:
            ptr = sum(queue) // 100 + 1
            if(landmark_point2[13] >= ptr):
                cnt += 1
                #print(cnt)
                isSleep = True
            else:
                isSleep = False
                cnt = 0

            if(cnt > 50 and isSleep):
                print(isSleep)


        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break


cap.release()