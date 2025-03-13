import cv2
import pyzed.sl as sl
from ultralytics import YOLO

# Load YOLO model
model = YOLO('best.pt')

def main():
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080 # Use HD720 opr HD1200 video mode, depending on camera type.
    init.camera_fps = 30  # Set fps at 30
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()
    
    runtime = sl.RuntimeParameters()
    mat = sl.Mat() 
    win_name = "YOLO Tracking"
    cv2.namedWindow(win_name)
    key = ''

    while key != 113:  # 'q' key to exit
        err = cam.grab(runtime) 
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat, sl.VIEW.LEFT)  
            cvImage = mat.get_data()  # Convert sl.Mat to numpy array (cv2 image)

            # Fix: Convert RGBA (4-channel) to RGB (3-channel) before passing to YOLO
            cvImage = cv2.cvtColor(cvImage, cv2.COLOR_RGBA2RGB)

            # Run YOLO detection
            results = model.predict(source=cvImage, conf=0.25, save=False)
            cv2.imshow("YOLO tracking",results[0].plot())
            # Extract results
            for result in results:
                boxes = result.boxes.xyxy  # Bounding box coordinates (x1, y1, x2, y2)
                confs = result.boxes.conf  # Confidence scores
                classes = result.boxes.cls  # Class labels

                # Draw bounding boxes
                for box, conf, cls in zip(boxes, confs, classes):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{int(cls)}: {conf:.2f}"

                    # Draw bounding box
                    cv2.rectangle(cvImage, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label
                    cv2.putText(cvImage, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show the image with OpenCV
            # cv2.imshow(win_name, cvImage)

        else:
            print("Error during capture:", err)
            break

        key = cv2.waitKey(5)

    # Cleanup
    cv2.destroyAllWindows()
    cam.close()

if __name__ == "__main__":
    main()
