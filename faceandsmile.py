import cv2 as cv

haarcascade_face = cv.CascadeClassifier('haar_face.xml')
haarcascade_smile = cv.CascadeClassifier('haar_smile.xml')
main_video = cv.VideoCapture(0)

while True:
    is_true, frame = main_video.read()
    grayscale_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detection_list = haarcascade_face.detectMultiScale(grayscale_frame)
    detection_list_smile = haarcascade_smile.detectMultiScale(grayscale_frame,1.7,22)
    
    for [x,y,w,h] in detection_list:
        cv.rectangle(frame, [x,y], [x+w,y+h], [0,255,0], 2)

    for [x,y,w,h] in detection_list_smile:
        cv.rectangle(frame, [x,y], [x+w,y+h], [0,255,0], 2)

    cv.imshow("Ahmad check", frame)
    key = cv.waitKey(1)
    if key == 81 or key == 113:
        break

main_video.release()
print("Completed")