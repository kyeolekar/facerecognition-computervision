import numpy as np
import os
import cv2
import time

def arrayOfImageName(dir_name):
    return os.listdir(dir_name)


def crop_image_to_face(folder_name, cascade, output_folder, width, height):
    print('The program will exit once the images are proccessed.')
    images = arrayOfImageName(folder_name)
    for img in images:
        faceCascade = cv2.CascadeClassifier(cascade)
        imgRead = cv2.imread(folder_name+'/'+img, 1)
        if(imgRead is None):
            exit(1)
        # imgRead = cv2.resize(imgRead, (width, height))
        grayImage = cv2.cvtColor(imgRead, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(grayImage, 1.3, 5)
        # crop_img = []
        for (x, y, wd, ht) in faces:
            cv2.rectangle(grayImage, (x, y), (x+wd, y+ht), (0, 255, 0), 2)
            cropImage = imgRead[y:y+ht, x:x+wd]
            cropImage = cv2.resize(cropImage, (width, height))
            cv2.imwrite(output_folder+"/"+img, cropImage)


def capture_from_webcam(name):  
    cap = cv2.VideoCapture(0)
    count = 0

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',grayImage)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            cv2.imwrite("raw_images/"+name+"%d.real.jpg" % count, grayImage)
            count += 1
        if key & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def main():
    name = input("Enter file name:")
    capture_from_webcam(name)
    folder_name = 'raw_images'
    cascade_name = 'haarcascade_frontalcatface.xml'
    output_folder = 'train'
    crop_image_to_face(folder_name, cascade_name, output_folder, 32, 32)
    print("Training data saved at train/ Folder")


if __name__ == '__main__':
    main()
