import flask
from flask import request
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
import cv2
import numpy as np
from io import BytesIO
from deepface import DeepFace
from os import makedirs
from time import time
import csv
import datetime

id_to_name = {}

retry_unknown_face_detection = True

plot_limited_objects = True

def get_person_name(face):
    
    try:
        # backends = ['opencv', 'ssd', 'dlib', 'mtcnn']

        # model_name = "Facenet", "VGG-Face", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"
        # distance_metric = "cosine", "euclidean", "euclidean_l2"
        
        result = DeepFace.find(img_path=face, db_path="face_db_bkp/", model_name = "DeepID", silent=True, enforce_detection=False, threshold=0.4, distance_metric="cosine", detector_backend='opencv')
        person_name = str(result[0]).split("/")[-2]

        print(f"Person name found: {person_name}")
        return person_name
    except Exception as e:
        print(e)
        return "unknown"

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


# conn = sql.connect("person_tracking.db")
# c = conn.cursor()
# c.execute('''CREATE TABLE IF NOT EXISTS person_tracking
#              (person_name TEXT, from_time REAL, to_time REAL, camera_name TEXT)''')

# conn.commit()

# file = open("tracking.csv", "rb")



app = flask.Flask(__name__)

model = YOLO("yolov8n.pt").to("cuda")
names = model.model.names

track_history = {}
last_seen = {}
id_to_object = {}

@app.route("/track", methods=["POST"])
def track():
    file = flask.request.files["file"]
    cam_name = flask.request.form.get("name")
    if not cam_name:
        cam_name = "Webcam"
    current_time = 0
    frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
    boxes = results[0].boxes.xyxy.cpu()

    if results[0].boxes.id is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, cls, track_id in zip(boxes, clss, track_ids):
            current_time = time()

            if track_id not in track_history:
                track_history[track_id] = []
                last_seen[track_id] = current_time

            track = track_history[track_id]
            track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
            if len(track) > 30:
                track.pop(0)

            object_name = names[int(cls)]
            id_to_object[track_id] = object_name

            if names[int(cls)] == "person":
                face_region = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                track_id = get_person_name(face_region)
                id_to_object[track_id] = ""


            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
            cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
            cv2.putText(frame, f"{object_name} {track_id}", (track[-1][0] + 10, track[-1][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors(int(cls), True), 2)
            cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

            last_seen[track_id] = current_time

    for track_id, last_time in list(last_seen.items()):
        if current_time - last_time > 10:
            appearance_time = last_time - 10
            leaving_time = last_time
            object_name = f"{id_to_object.get(track_id, f'unknown_')}{track_id}"
            # with open("tracking.csv", "a") as file:
            #     file.write(f"{object_name},{datetime(appearance_time)},{datetime(leaving_time)},{cam_name}\n")
            del last_seen[track_id]
            del track_history[track_id]
            del id_to_object[track_id]
            # print(id_to_object)

    _, buffer = cv2.imencode(".jpg", frame)
    return flask.send_file(BytesIO(buffer), mimetype="image/jpeg")

'''
@app.route("/track", methods=["POST"])
def track():
    file = flask.request.files["file"]
    cam_name = flask.request.form.get("name")
    # print(flask.request.form.keys())
    if not cam_name:
        cam_name = "Webcam"
    frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    # print("Frame received", frame.shape)
    results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
    boxes = results[0].boxes.xyxy.cpu()

    if results[0].boxes.id is not None:
        # Extract prediction results
        clss = results[0].boxes.cls.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, cls, track_id in zip(boxes, clss, track_ids):
            if track_id not in track_history:
                track_history[track_id] = []
            track = track_history[track_id]
            track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
            if len(track) > 30:
                track.pop(0)


            if names[int(cls)] == "person":
                # if track_id not in id_to_name:
                #     face_region = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                #     id_to_name[track_id] = get_person_name(face_region)

                #     track_id = id_to_name[track_id]
                # else:
                #     if retry_unknown_face_detection:
                #         if id_to_name[track_id] == "unknown":
                #             face_region = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                #             id_to_name[track_id] = get_person_name(face_region)
                #     track_id = id_to_name[track_id] + str(track_id)
                
                face_region = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                track_id = get_person_name(face_region)

                # with open("tracking.csv", "a") as file:
                #     file.write(f"{track_id},{time()},{cam_name}\n")
                
                
                # current_time = time()
                # c.execute("INSERT INTO person_tracking VALUES (?, ?, ?, ?)", (track_id, current_time, current_time, "camera1"))
                # c.execute("SELECT * FROM person_tracking WHERE person_name = ? AND camera_name = ?", (track_id, "camera1"))
                # existing_records = c.fetchall()
                # if len(existing_records) > 1:
                #     c.execute("UPDATE person_tracking SET to_time = ? WHERE person_name = ? AND camera_name = ? AND to_time = ?", (current_time, track_id, "camera1", existing_records[-1][2]))

                # else:
                #     c.execute("INSERT INTO person_tracking VALUES (?, ?, ?, ?)",
                #               (track_id, current_time, current_time, "camera1"))
                # conn.commit()
                
            # Plot tracks with updated names
            # if plot_limited_objects and names[int(cls)] in classNames:
            #     points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
            #     cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
            #     cv2.putText(frame, f"{names[int(cls)]} {track_id}", (track[-1][0] + 10, track[-1][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors(int(cls), True), 2)
            #     cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
            cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
            cv2.putText(frame, f"{names[int(cls)]} {track_id}", (track[-1][0] + 10, track[-1][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors(int(cls), True), 2)
            cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)
            if names[int(cls)] == "person":
                with open("tracking.csv", "a") as file:
                    file.write(f"{track_id},{time()},{cam_name}\n")
            else:
                with open("tracking.csv", "a") as file:
                    file.write(f"{names[int(cls)]}{track_id},{time()},{cam_name}\n")

    _, buffer = cv2.imencode(".jpg", frame)
    return flask.send_file(BytesIO(buffer), mimetype="image/jpeg")
'''

# @app.route("/search", methods=["POST"])
# def search():
#     # Get the object_name query parameter
#     object_name = request.args.get('object_name', '')

#     # Initialize an empty list to store the search results
#     results = []

#     # Open the tracking.csv file
#     with open('tracking.csv', 'r') as file:
#         reader = csv.reader(file)
#         # Iterate over each row in the CSV file
#         for row in reader:
#             # Check if the row matches the object_name
#             if row[0] == object_name:
#                 appearance_time = datetime.fromtimestamp(float(row[1])).strftime('%Y-%m-%d %H:%M:%S')
#                 leaving_time = datetime.fromtimestamp(float(row[2])).strftime('%Y-%m-%d %H:%M:%S')
#                 results.append([row[0], appearance_time, leaving_time, row[3]])

#     return flask.jsonify(results), 200




@app.route("/add_face", methods=["POST"])
def add_face():
    file = flask.request.files["file"]
    name = flask.request.form["name"]
    makedirs(f"face_db/{name}", exist_ok=True)
    frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = model(frame)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if box.cls == 0:
                cord = box.xyxy[0].cpu()
                face_region = frame[int(cord[1]):int(cord[3]), int(cord[0]):int(cord[2])]
                # print(type(face_region))
                cv2.imwrite(f"face_db/{name}/{name}_{time()}.jpg", face_region)
                return "Face added to database", 200
    return "Please add a face to the frame", 400

# /ping route for health check
@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200


print("Server is running...")
app.run(host="0.0.0.0", port=5000)
