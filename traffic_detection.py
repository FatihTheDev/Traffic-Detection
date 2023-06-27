import cv2

trainingData = cv2.CascadeClassifier("haarcascade_fullbody.xml")
carTrainingData = cv2.CascadeClassifier("cars.xml")

camera = cv2.VideoCapture(input("Enter file name:"))

while 1:
    successful_frame, frame = camera.read()

    grayscaled_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    pedestrian_coordinates = trainingData.detectMultiScale(grayscaled_vid)
    car_coordinates = carTrainingData.detectMultiScale(grayscaled_vid)

    for (x,y,w,h) in pedestrian_coordinates:
        rect = cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 255), 4)
    for (x,y,w,h) in car_coordinates:
        rect = cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 4)

    cv2.imshow('Car and pedestrian recognition', frame)
    key = cv2.waitKey(1)

    if(key == 81 or key == 113):
        break
