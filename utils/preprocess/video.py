import dlib
import imutils
from cv2 import cvtColor
from cv2 import COLOR_BGR2GRAY
from cv2 import VideoCapture

from utils.preprocess.face_aligner import FaceAligner


# TODO: move to initialization
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
aligner = FaceAligner(predictor, desiredFaceWidth=64)


def _get_aligned_face(frame, detector, aligner: FaceAligner):
    frame = imutils.resize(frame, width=256)
    gray = cvtColor(frame, COLOR_BGR2GRAY)
    faces = detector(gray, 2)
    faceAligned = []

    if len(faces):
        face = faces[0]
        faceAligned = aligner.align(frame, gray, face)

    return faceAligned


def get_faces(path, detector, aligner: FaceAligner, each_frame=5):
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
            face = _get_aligned_face(frame, detector, aligner)
            faces.append(face)

    return faces
