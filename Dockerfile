FROM python:3.9

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

RUN wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 &&  \
    bunzip2 --force shape_predictor_68_face_landmarks.dat.bz2

COPY ./requirements.txt requirements.txt

RUN python3 -m pip install cmake
RUN python3 -m pip install -r requirements.txt

ADD main.py main.py
ADD setup.py setup.py
ADD models/ models/
ADD util/ util/

EXPOSE 5000

CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "main:app" ]
