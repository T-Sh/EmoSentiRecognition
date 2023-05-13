import dlib
import imutils
import torch
from cv2 import COLOR_BGR2GRAY, VideoCapture, cvtColor
from torch.nn.utils.rnn import pad_sequence

from util.preprocess.face_aligner import FaceAligner

import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


class VideoProcessor:
    def __init__(self, desired_face_width=64):
        predictor = dlib.shape_predictor(
            "/app/shape_predictor_68_face_landmarks.dat"
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

        float_f = []
        max_l = 25
        group_size = 10
        overlap = 5

        ind = 0
        while ind < max_l:
            group = []
            end_pos = ind + overlap

            for i in range(ind, end_pos):
                if i >= len(faces):
                    break
                f = faces[i]
                if f != []:
                    f = rgb2gray(f)
                else:
                    f = [[0 for col in range(64)] for row in range(64)]
                f = torch.FloatTensor(f)

                group.append(f)

            for i in range(len(group), group_size):
                group.append(torch.zeros(64, 64))

            group = torch.transpose(pad_sequence(group), 0, 1)
            float_f.append(group)
            ind += overlap

        try:
            X = torch.transpose(pad_sequence(float_f), 0, 1)
        except Exception:
            print(len(float_f), float_f[0].shape, float_f[1].shape)
            raise Exception

        video_features_tensor = torch.transpose(pad_sequence([X]), 0, 1)

        return video_features_tensor
