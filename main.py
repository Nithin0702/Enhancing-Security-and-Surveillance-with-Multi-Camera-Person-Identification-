import streamsync as ss
import requests
import json
from os import makedirs
from pandas import read_csv, to_datetime
from random import randint

with open('settings.json') as f:
    settings = json.load(f)
    # camera_source = settings["camera_source"]
    objects = settings["objects"]
    model = settings["yolo_model"]
    api_endpoint = settings["api_endpoint"]

def update_tracking_df(state):
    df = read_csv("tracking.csv", header=None, names=['object_name', 'appearance_time', 'leaving_time', 'camera_name'])
    # convert "appearance_time" and "leaving_time" to datetime
    df['appearance_time'] = to_datetime(df['appearance_time'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
    df['leaving_time'] = to_datetime(df['leaving_time'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
    state["tracking_df"] = df



def _process_frame(frame,cam_name="webcam"):
    return requests.post("http://localhost:5000/track", files={"file": frame}, data={"name": cam_name}).content

def process_webcam_frame(payload,state):
    state["webcamfeed"] = _process_frame(payload)
    # print(type(state["webcamfeed"]))
    

def face_capture(payload,state):
    # print(type(payload))
    state["temp_face_capture"] = payload

def add_to_db(state):
    if state["new_registration_name"] != None:
        makedirs(f"face_db/{state['new_registration_name']}", exist_ok=True)
        # try:
        #     cv2.imwrite(f"face_db/{state['new_registration_name']}/{state['new_registration_name']}_{time()}.jpg", cv2.imdecode(np.frombuffer(state["temp_face_capture"], np.uint8), cv2.IMREAD_COLOR))
        #     state.add_notification("success", "A Success", "New image added to database")
        # except Exception as e:
        #     state.add_notification("error", "An Error", str(e))
        response = requests.post("http://localhost:5000/add_face", files={"file": state["temp_face_capture"]}, data={"name": state["new_registration_name"]})
        if response.status_code == 200:
            state.add_notification("success", "A Success", "New image added to database " + state["new_registration_name"] )
        elif response.status_code == 400:
            state.add_notification("error", "An Error", "Please add a face to the frame")
    else:
        state.add_notification("error", "An Error", "Please enter a name")


def is_api_endpoint_up(state, payload):
    if payload == None:
        with ss.init_ui() as ui:
                message_box = ui.find("crueypuog025im50")
                # print(message_box.properties, "yes")

                message_box.content["message"] = "Using default API endpoint"
                state["api_endpoint"] = "http://localhost:5000"

        # state["msg"] = "Selected default API endpoint"
        return None
    try:
        print("Checking API endpoint")
        response = requests.get(f"{payload}/ping")
        if response.status_code == 200:
            with ss.init_ui() as ui:
                    message_box = ui.find("crueypuog025im50")

                    message_box.content["message"] = "+API endpoint is reachable"
                    state["api_endpoint"] = payload
                    return True
        else:
            with ss.init_ui() as ui:
                    message_box = ui.find("crueypuog025im50")
                    # print(message_box.properties,"ye    s")

                    message_box.content["message"] = "-API endpoint is not reachable"
                    state["api_endpoint"] = "http://localhost:5000"
                    return False
                    
    except:
        with ss.init_ui() as ui:
                message_box = ui.find("crueypuog025im50")
                # print(message_box,"ye    s")
                message_box.content["message"] = "-API endpoint is not reachable"
                state["api_endpoint"] = "http://localhost:5000"
                return False

def select_yolo_model(state, payload):
    state["yolo_model"] = payload
    # print(payload)
    return True

def get_objects_list(state, payload):
    state["objects"] = payload
    # print(payload)
    return True

def sync_settings(state):
    # print(state)

    if state["api_endpoint"] == None:
        api_endpoint = "http://localhost:5000"
    else:
        api_endpoint = state["api_endpoint"]
    if state["yolo_model"] == None:
        yolo_model = "yolov8s"
    else:
        yolo_model = state["yolo_model"]
    if state["objects"] == None:
        objects = ["person"]
    else:
        objects = state["objects"]
 
    
    settings = {
        "api_endpoint": api_endpoint,
        "yolo_model": yolo_model,
        "objects": objects
    }
    with open('settings.json', 'w') as f:
        json.dump(settings, f)
    state.add_notification("success", "A Success", "Settings saved successfully")
    return True
    


def cam1_toggle(state,payload): state["cam1"] = payload
def cam2_toggle(state,payload): state["cam2"] = payload
def cam3_toggle(state,payload): state["cam3"] = payload
def cam4_toggle(state,payload): state["cam4"] = payload
def cam5_toggle(state,payload): state["cam5"] = payload
def cam6_toggle(state,payload): state["cam6"] = payload
def surveillance_system_toggle(state,payload): state["surveillance_system"] = payload


camera_sources = {
    "cam1": "http://localhost:6000/cam1",
    "cam2": "http://localhost:6000/cam2",
    "cam3": "http://localhost:6000/cam3",
    "cam4": "http://localhost:6000/cam4",
    "cam5": "http://localhost:6000/cam5",
    "cam6": "http://localhost:6000/cam1"
}


def process_ipcam1_frame(state):
    if state["cam1"]:
        # print("Processing cam1")

        try:
            payload = requests.get(camera_sources["cam1"]).content
            state["ipcamfeed1"] = _process_frame(payload,"Out side")
        except Exception as e:
            print(e)
            # state["ipcamfeed1"] = None


def process_ipcam2_frame(state):
    if state["cam2"]:
        
        payload = requests.get(camera_sources["cam2"]).content
        state["ipcamfeed2"] = _process_frame(payload, "Office Front Side")

def process_ipcam3_frame(state):
    if state["cam3"]:

        payload = requests.get(camera_sources["cam3"]).content
        state["ipcamfeed3"] = _process_frame(payload, "Cabin Celing View")

def process_ipcam4_frame(state):
    if state["cam4"]:

        payload = requests.get(camera_sources["cam4"]).content
        state["ipcamfeed4"] = _process_frame(payload, "Camera 4")

def process_ipcam5_frame(state):
    if state["cam5"]:

        payload = requests.get(camera_sources["cam5"]).content
        state["ipcamfeed5"] = _process_frame(payload, "Camera 5")

def process_ipcam6_frame(state):
    if state["cam6"]:

        payload = requests.get(camera_sources["cam6"]).content
        state["ipcamfeed6"] = _process_frame(payload, "Camera 6")


# def surveillance_system(state):

#     process_ipcam1_frame(state)
#     process_ipcam2_frame(state)
#     process_i     pcam3_frame(state)
#     process_ipcam4_frame(state)
#     process_ipcam5_frame(state)
#     process_ipcam6_frame(state)

def upload_video_file(state, payload):
    uploaded_files = payload
    for i, uploaded_file in enumerate(uploaded_files):
        name = uploaded_file.get("name")
        file_data = uploaded_file.get("data")
        with open(f"{name}", "wb") as file_handle:
            file_handle.write(file_data)
        import cv2
        cap = cv2.VideoCapture(name)
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                _, buffer = cv2.imencode('.jpg', frame)
                state["video_playback"] = _process_frame(buffer,"Uploaded Video")

    state.add_notification("success", "A Success", "File Processed successfully")


        
    


        
def go_to_dashboard(state):
    state.set_page("dashboard")

def go_to_settings(state):
    state.set_page("settings")




initial_state = ss.init_state({
    "my_app": {
        "title": "Dashboard"
    },

})

initial_state.import_stylesheet("theme", f"/static/custom.css?{randint(1, 1000)}")

