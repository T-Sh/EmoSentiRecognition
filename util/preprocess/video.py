import dlib
import imutils
from cv2 import COLOR_BGR2GRAY, VideoCapture, cvtColor

from util.preprocess.face_aligner import FaceAligner


class VideoProcessor:
    def __init__(self, desired_face_width=64):
        predictor = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat"
        )
        self.detector = dlib.get_frontal_face_detector()
        self.aligner = FaceAligner(
            predictor, desiredFaceWidth=desired_face_width
        )

    def __get_aligned_face(self, frame):
        frame = imutils.resize(frame, width=256)
        gray = cvtColor(frame, COLOR_BGR2GRAY)
        faces = self.detector(gray, 2)
        face_aligned = []

        if len(faces):
            face = faces[0]
            face_aligned = self.aligner.align(frame, gray, face)

        return face_aligned

    def get_features(self, path, each_frame=5):
        capture = VideoCapture(path)

        frame_numbers = 0
        faces = []

        while True:
            success, frame = capture.read()
            if not success:
                if frame_numbers > 0:
                    break
                else:
                    frame_numbers += 1
                    continue

            frame_numbers += 1
            if frame_numbers % each_frame == 0:
                face = self.__get_aligned_face(frame)
                faces.append(face)

        return faces
