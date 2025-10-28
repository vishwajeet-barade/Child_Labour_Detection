# python detect_actions.py --video input.mp4 --models models --out labeled_output_with_action.mp4

import os
import cv2
import argparse
import time
import numpy as np
import torch
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
from collections import deque
from dotenv import load_dotenv

load_dotenv() 

# --- Configuration for Models ---

# Default model filenames (match your Drive)
DEFAULT_MODEL_DIR = "models"
AGE_PROTO_NAME = "age_deploy.prototxt"
AGE_MODEL_NAME = "age_net.caffemodel"
GENDER_PROTO_NAME = "gender_deploy.prototxt"
GENDER_MODEL_NAME = "gender_net.caffemodel"
FACE_PB_NAME = "opencv_face_detector_uint8.pb"
FACE_PBTXT_NAME = "opencv_face_detector.pbtxt"

# Age buckets used by the Caffe age model
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# --- NEW: Configuration for Child and Action Detection ---

# Define which age buckets we consider as "child"
CHILD_AGE_BUCKETS = {'(0-2)', '(4-6)', '(8-12)', '(15-20)'}

# Define a set of actions that could be considered "labour-like".
# These labels come from the Kinetics-400 dataset the model was trained on.
# You can add or remove actions based on your specific needs.
LABOUR_ACTIONS = {
    'digging', 'hammering', 'sweeping floor', 'carrying weight', 'chopping wood',
    'cleaning floor', 'excavating', 'farming', 'mining', 'pushing cart', 'washing dishes', 'welding','laying bricks'
}

hf_token = os.getenv("HF_TOKEN")

# Hugging Face Action Recognition Model
ACTION_MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
# How many frames to collect for one action prediction
CLIP_LENGTH = 16
# Run action inference every N frames to save computation
ACTION_INFERENCE_INTERVAL = 16 

# --- Model Loading and Utilities ---

# ... (All your existing utility functions: expand_box, find_face_model, etc. remain here unchanged) ...
def expand_box(box, w, h, margin=0.35):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    x1m = max(0, int(x1 - margin * bw))
    y1m = max(0, int(y1 - margin * bh))
    x2m = min(w, int(x2 + margin * bw))
    y2m = min(h, int(y2 + margin * bh))
    return x1m, y1m, x2m, y2m

def find_face_model(model_dir):
    pb = os.path.join(model_dir, FACE_PB_NAME)
    pbtxt = os.path.join(model_dir, FACE_PBTXT_NAME)
    if os.path.exists(pb) and os.path.exists(pbtxt):
        return "tf", pb, pbtxt
    return None, None, None # Simplified for brevity

def load_face_net(model_dir):
    kind, p1, p2 = find_face_model(model_dir)
    if kind == "tf":
        print(f"[INFO] Using TensorFlow face detector.")
        net = cv2.dnn.readNetFromTensorflow(p1, p2)
        return net
    else:
        raise FileNotFoundError("No TF face detector found in model dir.")

def load_caffe_net(proto_path, model_path, name="net"):
    if not os.path.exists(proto_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model files not found for {name}")
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    print(f"[INFO] Loaded {name}")
    return net

def load_action_model():
    """Loads the action recognition model from Hugging Face."""
    print(f"[INFO] Loading action recognition model: {ACTION_MODEL_ID}")
    print("[INFO] This may take a moment on the first run as the model is downloaded...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_extractor = VideoMAEFeatureExtractor.from_pretrained(ACTION_MODEL_ID,token=hf_token)
        model = VideoMAEForVideoClassification.from_pretrained(ACTION_MODEL_ID,token=hf_token).to(device)
        model.eval() # Set model to evaluation mode
        print(f"[INFO] Action model loaded successfully on device: {device}")
        return model, feature_extractor, device
    except Exception as e:
        print(f"[ERROR] Could not load action model. Please check your internet connection and transformers installation.")
        print(e)
        return None, None, None

# --- Main processing ---
def main(video_source=0, model_dir=DEFAULT_MODEL_DIR, out_path=None, conf_threshold=0.6):
    # Load all models
    face_net = load_face_net(model_dir)
    age_net = load_caffe_net(os.path.join(model_dir, AGE_PROTO_NAME), os.path.join(model_dir, AGE_MODEL_NAME), "age_net")
    gender_net = load_caffe_net(os.path.join(model_dir, GENDER_PROTO_NAME), os.path.join(model_dir, GENDER_MODEL_NAME), "gender_net")
    action_model, feature_extractor, device = load_action_model()

    if action_model is None:
        print("[ERROR] Exiting due to action model failure.")
        return

    # Video capture setup
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {video_source}")
    
    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        print(f"[INFO] Writing output to {out_path}")

    # --- NEW: Action detection variables ---
    frame_buffer = deque(maxlen=CLIP_LENGTH)
    last_detected_action = "analyzing..."
    
    frame_idx = 0
    t_start = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            
            # Add current frame to buffer for action detection
            # Model expects RGB, OpenCV provides BGR, so we convert
            frame_buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # --- Perform Action Recognition periodically ---
            if len(frame_buffer) == CLIP_LENGTH and frame_idx % ACTION_INFERENCE_INTERVAL == 0:
                # Prepare the video clip for the model
                video_clip = list(frame_buffer)
                inputs = feature_extractor(video_clip, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = action_model(**inputs)
                    logits = outputs.logits
                    predicted_class_idx = logits.argmax(-1).item()
                    last_detected_action = action_model.config.id2label[predicted_class_idx]

            (h, w) = frame.shape[:2]

            # --- Perform Face, Age, Gender Detection (your original logic) ---
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 177, 123], swapRB=False, crop=False)
            face_net.setInput(blob)
            detections = face_net.forward()

            for i in range(0, detections.shape[2]):
                confidence = float(detections[0, 0, i, 2])
                if confidence < conf_threshold:
                    continue

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                x1e, y1e, x2e, y2e = expand_box((x1, y1, x2, y2), w, h)

                face = frame[y1e:y2e, x1e:x2e]
                if face.size == 0:
                    continue

                # Gender prediction
                blob_gender = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.42, 87.76, 114.89), swapRB=False)
                gender_net.setInput(blob_gender)
                gender = GENDER_LIST[gender_net.forward()[0].argmax()]
                
                # Age prediction
                blob_age = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.42, 87.76, 114.89), swapRB=False)
                age_net.setInput(blob_age)
                age_bucket = AGE_BUCKETS[age_net.forward()[0].argmax()]

                # --- NEW: Check for Child Labour Condition ---
                is_child = age_bucket in CHILD_AGE_BUCKETS
                is_labour_action = last_detected_action in LABOUR_ACTIONS

                label = f"{gender}, {age_bucket}"
                color = (0, 255, 0) # Default: Green

                if is_child and is_labour_action:
                    # If a child is detected and a labour action is happening, flag it
                    label = f"!! POTENTIAL CHILD LABOUR !! Age: {age_bucket}"
                    color = (0, 0, 255) # Red for alert
                    cv2.rectangle(frame, (x1e, y1e), (x2e, y2e), color, 4) # Thicker box
                else:
                    cv2.rectangle(frame, (x1e, y1e), (x2e, y2e), color, 2)

                y_text = y1e - 10 if y1e - 10 > 10 else y1e + 20
                cv2.putText(frame, label, (x1e, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            # Display the overall detected action on the top-left corner
            action_text = f"Action: {last_detected_action}"
            cv2.putText(frame, action_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Age, Gender & Action Detection", frame)
            if writer:
                writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Finished.")

# -------------------
# CLI
# -------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Detect age, gender, and actions, highlighting potential child labour.")
    p.add_argument("--video", type=str, default=None, help="Path to input video file. If omitted, webcam used.")
    p.add_argument("--models", type=str, default=DEFAULT_MODEL_DIR, help="Folder containing model files.")
    p.add_argument("--out", type=str, default=None, help="Optional output video file path.")
    p.add_argument("--conf", type=float, default=0.6, help="Face detection confidence threshold.")
    args = p.parse_args()

    video_src = args.video if args.video else 0
    main(video_source=video_src, model_dir=args.models, out_path=args.out, conf_threshold=args.conf)