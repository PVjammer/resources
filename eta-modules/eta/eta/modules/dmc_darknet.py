
import os
import time
import sys

DARKPY_PATH = "../../../madhawav/YOLO3-4-Py"
# DARKPY_PATH = "."

sys.path.append(DARKPY_PATH)
from pydarknet import Detector, Image
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CFG = os.path.join(DARKPY_PATH, "cfg", "yolov3.cfg")
WEIGHTS = os.path.join(DARKPY_PATH, "weights", "yolov3.weights")
DATA = os.path.join(DARKPY_PATH, "cfg", "coco.data")

def run_darknet(config=None):
    dknet_config ={
        "cfg_path" : CFG,
        "weights_path" : WEIGHTS,
        "data_path" : DATA
    }
    
    if config is not None:
        dknet_config.update(config.data[0].__dict__)
        dknet_config.update(config.parameters.__dict__)
    logger.info(dknet_config)

    net = Detector(bytes(dknet_config["cfg_path"], encoding="utf-8"), bytes(dknet_config["weights_path"], encoding="utf-8"), 0, bytes(dknet_config["data_path"], encoding="utf-8"))

    logger.info("Attempting to process: {!s}".format(dknet_config["input_path"]))
    dknet_config["input_path"].split("/")[-1].split(".")[0]
    if  dknet_config["input_path"].split("/")[-1].split(".")[0] == "webcam":
        logger.debug("Running on webcam")
        cap = cv2.VideoCapture(0)
        webcam = True
    else:
        cap = cv2.VideoCapture(dknet_config["input_path"])
        webcam = False
    logger.info("Got capture")

    currentFrame = 0

    # Get current width of frame
#    width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)   # float
    # Get current height of frame
 #   height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) # float


    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(dknet_config["output_path"],fourcc, 20.0, (640,480))

    logger.debug("Capture open?: {!s}".format(cap.isOpened()))
    while(cap.isOpened()):
        r, frame = cap.read()
        logger.debug("Capturing video: {!s}".format(r))
        if not r:
            logger.warn("No frame captured")
            break
        if r:

            start_time = time.time()

            # Only measure the time taken by YOLO and API Call overhead

            dark_frame = Image(frame)
            results = net.detect(dark_frame)
            del dark_frame

            end_time = time.time()
            logger.info("Elapsed Time: {!s}".format(end_time-start_time))

            for cat, score, bounds in results:
                x, y, w, h = bounds
                cv2.rectangle(frame, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0))
                cv2.putText(frame, str(cat.decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
             # Handles the mirroring of the current frame
#             frame = cv2.flip(frame,1)

            # Saves for video
            out.write(frame)
            cv2.imshow("preview", frame)

        k = cv2.waitKey(1)
        if webcam:
            if currentFrame > 900:
                break
    
        if k == 0xFF & ord("q"):
            break
        currentFrame += 1

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def camera_test():
    net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3.weights", encoding="utf-8"), 0,
                   bytes("cfg/coco.data", encoding="utf-8"))

    print("Attempting to capture webcam")
    cap = cv2.VideoCapture(0)
    print("Got capture")

    while True:
        r, frame = cap.read()
        if r:
            start_time = time.time()

            # Only measure the time taken by YOLO and API Call overhead

            dark_frame = Image(frame)
            results = net.detect(dark_frame)
            del dark_frame

            end_time = time.time()
            print("Elapsed Time:",end_time-start_time)

            for cat, score, bounds in results:
                x, y, w, h = bounds
                cv2.rectangle(frame, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0))
                cv2.putText(frame, str(cat.decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))

            cv2.imshow("preview", frame)

        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            break


if __name__ == "__main__":
    # Optional statement to configure preferred GPU. Available only in GPU version.
    # pydarknet.set_cuda_device(0)

    class Config():
        def __init__(self):
            self.cfg_path = CFG
            self.weights_path = WEIGHTS
            self.data_path = DATA
            self.input_path = 0
            self.output_path = "test.avi"

    config = Config()
    run_darknet(config)
