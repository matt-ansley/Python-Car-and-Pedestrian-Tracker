import cv2

# input image
# img_file = 'car_image.jpg'
video = cv2.VideoCapture('dashcam.mp4')
# pre-trained car classifier
classifier_file = 'car_detector.xml'

car_tracker = cv2.CascadeClassifier(classifier_file)
pedestrian_tracker = cv2.CascadeClassifier('haarcascade_fullbody.xml')



while True:
    # create opencv image, and read until car crashes or something
    read_successful, frame = video.read()

    if read_successful:
        # convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    cars = car_tracker.detectMultiScale(grayscaled_frame, minSize=(75, 75))
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    for(x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 256, 0), 2)

    print(cars)

    for(x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 256, 0), 2)

    print(pedestrians)

    cv2.imshow('Car and Pedestrian Tracker', frame)
    key = cv2.waitKey(3)

    if key == 81 or key == 113:
        break


"""
# convert to grayscale
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file, )

car_coordinates = car_tracker.detectMultiScale(grayImage)

print(car_coordinates[0])

# display the image

for (x, y, w, h) in car_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 256, 0), 10)

cv2.imshow('Car and Pedestrian Tracker', img)
cv2.waitKey()
"""