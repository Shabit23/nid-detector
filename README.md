# nid-detector

A python program to detect the NAME and NID NO from scanned nid images and record them in the RECORD.csv file.
After recording the NAME and NID NO, the program renames the scanned images in the format `$ NAME_NID`
If a file with the name format `$ NAME_NID` already exists, it skips the text extraction process altogether.
Then the program extracts 68 facial features utilizing the Face Landmark Estimation algorithm after detecting the face using Histogram of Oriented Gradient algorithm and stored in encode_list array.
After that the webcam is opened up and the same face recognition technique is applied.
The Euclidean Distace between the facial features of the scanned images and the web cam is stored in an array. The one with the least distance is detected to be the face on the web cam
After that a record of the detected face is kept on the Detected Records.csv file with the time of detection

Steps to run the program:
1) Install the requirements of the project using the command:
    pip install -r requirements.txt
2) Run the main.py file

N.B. The RegionSelector.py file was used to select the region of interest manually from an NID. It is not a part of the main program.
