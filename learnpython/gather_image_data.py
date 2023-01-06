import matplotlib.pyplot as plt # import the pyplot module of the matplotlib library
import cv2 # import the opencv library
import os # import the os library
import numpy as np # import the numpy library

# create a VideoCapture object 'vid' to capture the video from the default camera
vid = cv2.VideoCapture(0)

# create a while loop to continuously capture and display frames from the video
for i in range(10):
    # read the current frame of the video and store it in the 'ret' and 'frame' variables
    ret, frame = vid.read()

    # convert the frame from BGR (blue-green-red) color space to RGB (red-green-blue) color space
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # display the frame using the pyplot imshow function
    plt.imshow(frame)
    # pause the pyplot window for a small amount of time to allow the frame to be displayed
    plt.pause(1e-10)
    
    # create a directory named "data" if it doesn't already exist
    if not os.path.exists("data"):
        os.makedirs("data")
        
    # save the frame as a numpy array in the "data" directory
    np.save(f"data/frame-{i}.npy", frame)

# release the VideoCapture object and close all windows created by opencv
vid.release()
cv2.destroyAllWindows()
