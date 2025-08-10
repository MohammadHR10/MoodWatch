import cv2, time, threading
from datetime import datetime
# pip install deepface
from deepface import DeepFace

CAM_INDEX = 0
SHOW_FPS = True
ANALYZE_FPS = 8          # cap emotion model to this rate
FRAME_SIZE = (1280, 720) # try (640,480) if your CPU is modest

state = {
    "frame": None,
    "frame_id": 0,
    "last_draw": None,
    "running": True
}
lock = threading.Lock()

def detect_emotion(cam=0):
    print(f"Opening camera {cam}...")
    capture_video = cv2.VideoCapture(cam)
    
    if not capture_video.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Camera opened successfully!")
    print("Waiting for camera to fully initialize...")
    time.sleep(3)  # Give camera time to initialize
    
    # Try to set camera properties
    capture_video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Try multiple frame reads to get a good frame
    for attempt in range(10):
        ret, frame = capture_video.read()
        if ret and frame.sum() > 0:  # Check if frame has content
            print(f"Got working frame on attempt {attempt + 1}")
            break
        time.sleep(0.5)  # Wait between attempts
    else:
        print("Error: Could not get a working frame after 10 attempts.")
        capture_video.release()
        return
    
    if ret:
        # Display the frame
        cv2.imshow("First Frame", frame)
        
        # Bring window to front and give it focus
        cv2.setWindowProperty("First Frame", cv2.WND_PROP_TOPMOST, 1)
        
        print("Camera working! Press 'q' or 'ESC' to close the window")
        print("Make sure to click on the camera window first!")
        
        while True:
            # Refresh the display
            cv2.imshow("First Frame", frame)
            
            # Check for key press
            key = cv2.waitKey(30) & 0xFF  # Increased wait time
            if key == ord('q') or key == 27:  # 'q' or ESC key
                print("Closing camera...")
                break
            elif key != 255:  # Any other key was pressed
                print(f"Key pressed: {key}. Press 'q' or ESC to quit.")
        
        # Close the window
        cv2.destroyAllWindows()
    else:
        print("Error: Could not read the frame.")
    
    # Release the video capture object
    capture_video.release()

if __name__ == "__main__":
    detect_emotion()