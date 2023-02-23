import matplotlib.pyplot as plt # import the pyplot module of the matplotlib library
import glob # import the glob library
import cv2 # import the opencv library
import numpy as np # import the numpy library

# use glob to search for all numpy files in the "data" directory
filenames = glob.glob("data/*.npy")

# sort the filenames in ascending order
filenames.sort()

# load the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# iterate through the filenames and load each frame
for filename in ['kim.jpg']: #filenames:
    # try to load the frame from the file
    try:
        if '.npy' in filename:
            frame = np.load(filename)
        else:
            frame = cv2.imread(filename)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    
    except Exception as e:
        print(f"Error loading image {filename}: {e}")
        continue
    
    # convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect faces in the frame
    faces = face_cascade.detectMultiScale(image=gray_frame)

    # loop through the faces and draw a rectangle around them
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

    # display the frame with the detected faces using the pyplot imshow function
    plt.imshow(frame)
    # # pause the pyplot window for a small amount of time to allow the frame to be displayed
    # plt.pause(1e-10)
    plt.show()

    break

# close the pyplot window
plt.close()
