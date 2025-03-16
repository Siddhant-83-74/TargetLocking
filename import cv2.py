import cv2
import pyzed.sl as sl
from ultralytics import YOLO

CONFIDENCE_THRESHOLD=0.25
model = YOLO('best.pt')

def main():
    init = sl.InitParameters()
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
    while key != 113:  # for 'q' key
        err = cam.grab(runtime) 
        if err == sl.ERROR_CODE.SUCCESS: # Check that a new image is successfully acquired
            cam.retrieve_image(mat, sl.VIEW.LEFT) # Retrieve left image
            cvImage = mat.get_data() # Convert sl.Mat to cv2.Mat
            # if (not selection_rect.is_empty() and selection_rect.is_contained(sl.Rect(0,0,cvImage.shape[1],cvImage.shape[0]))): #Check if selection rectangle is valid and draw it on the image
            #     cv2.rectangle(cvImage,(selection_rect.x,selection_rect.y),(selection_rect.width+selection_rect.x,selection_rect.height+selection_rect.y),(220, 180, 20), 2)
            results = model.predict(source=cvImage, conf=CONFIDENCE_THRESHOLD, save=False)#56 show=True)
            cv2.imshow("win_name ",results[0].plot())
            # cv2.imshow(win_name, cvImage) #Display image

        else:
            print("Error during capture : ", err)
            break
        
        key = cv2.waitKey(5)
        # Change camera settings with keyboard
        # update_camera_settings(key, cam, runtime, mat)
    cv2.destroyAllWindows()

    cam.close()


if __name__ == "__main__":
    main()