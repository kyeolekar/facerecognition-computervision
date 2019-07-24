import numpy as np
import os
import sys
import cv2

def arrayOfImageName(dir_name):
    return os.listdir(dir_name)

def crop_image_to_face(folder_name, cascade, output_folder, width, height):
    print('The program will exit once the images are proccessed.')
    images = arrayOfImageName(folder_name)
    for img in images:
        faceCascade = cv2.CascadeClassifier(cascade)
        img_read = cv2.imread(folder_name+'/'+img, 1)
        if(img_read is None):
            exit(1)
        # img_read = cv2.resize(img_read, (width, height))
        gray = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        # crop_img = []
        for (x, y, w, h) in faces:
            cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cropimg = gray[y:y+h, x:x+w]
            cropimg = cv2.resize(cropimg, (width, height))
            cv2.imwrite(output_folder+"/"+img, cropimg)

def helper():
    print('Arguments\n1.Folder Name\n2.Cascade File\n3.Output Folder')

def main():
    
    width = 32
    height = 32

    folder_name = 'raw_images'
    cascade_name = 'haarcascade_frontalcatface.xml'
    output_folder = 'train'
    crop_image_to_face(folder_name, cascade_name, output_folder, width, height)

if __name__ == '__main__':
    main()
