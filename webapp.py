from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import torch
import glob
import argparse
import concurrent.futures

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DETECTIONS_FOLDER'] = 'static/detections'

basepath = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(basepath, 'static/uploads')
DETECTIONS_FOLDER = os.path.join(basepath, 'static/detections')

MODEL_WEIGHTS_PATH = 'best.pt'  # Your fine-tuned model path
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_WEIGHTS_PATH)


def clear_detection_folder():
    """Remove all files in the detections folder."""
    files = glob.glob(os.path.join(DETECTIONS_FOLDER, '*'))
    for f in files:
        os.remove(f)


def process_frame(model, frame):
    """Process a single frame with the YOLO model."""
    results = model(frame)
    return results.render()[0]


@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            filepath = os.path.join(basepath, UPLOAD_FOLDER, f.filename)
            print(f"File path {filepath}")
            f.save(filepath)
            file_extension = f.filename.rsplit('.', 1)[1].lower()

            try:
                if file_extension in ['jpg', 'jpeg', 'png']:
                    img = cv2.imread(filepath)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = model(img)
                    results.save(save_dir=os.path.join(basepath, DETECTIONS_FOLDER))
                    detected_image_path = os.path.join(app.config['DETECTIONS_FOLDER'], f.filename)
                    return render_template("index.html", image_path=f.filename,
                                           detected_image_path=f.filename + '/image0.jpg')

                elif file_extension in ['mp4']:
                    cap = cv2.VideoCapture(filepath)
                    if not cap.isOpened():
                        raise ValueError("Could not open the video file.")

                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    f_out_name = f.filename.split('.')[0]
                    f_name = f_out_name + '_output.mp4'
                    output_path = os.path.join(DETECTIONS_FOLDER, f'{f_out_name}_output.mp4')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                    frames = []
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(frame)

                    cap.release()

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        processed_frames = list(executor.map(lambda f: process_frame(model, f), frames))

                    for frame in processed_frames:
                        out.write(frame)

                    out.release()
                    return render_template("index.html", video_path=f.filename, detected_video_path=f_name)

            except Exception as e:
                print(f"Error processing file: {e}")
                return render_template("index.html", error=str(e))

    return render_template("index.html")


@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/static/detections/<filename>')
def detected_file(filename):
    return send_from_directory(app.config['DETECTIONS_FOLDER'], filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000, help='Port to run the app on')
    opt = parser.parse_args()
    app.run(host='0.0.0.0', port=opt.port)
