import numpy as np
import os
import cv2

def arrayOfImageName(dir_name):
    return os.listdir(dir_name)

def crop_image_to_face(img, cascade, width, height):
    faceCascade = cv2.CascadeClassifier(cascade)
    imgRead = cv2.imread(img, 0)
    # gray = cv2.cvtColor(imgRead, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        imgRead,
        scaleFactor= 1.1,
        minNeighbors= 5,
        minSize= (30, 30)
    )
    cropImage = 0
    
    for (x, y, wd, ht) in faces:
        cv2.rectangle(imgRead, (x, y), (x+wd, y+ht), (0, 255, 0), 2)
        cropImage = imgRead[y:y+ht, x:x+wd]
        cropImage = cv2.resize(cropImage, (width, height))
        cv2.imwrite("test/"+img.split('/')[1], cropImage)
    if (len(faces) <= 0):
        print('Couldnt detect face.')
        exit(1)


def get_vector_from_img(training_img_dir, training_images_names, width, height):
    # TODO : Add last char '/' check
    trainingArray = np.ndarray(shape=(len(training_images_names), height*width), dtype=np.float64)
    for i in range(len(training_images_names)):
        img = cv2.imread(training_img_dir + training_images_names[i], 0)
        img = cv2.equalizeHist(img)
        img = cv2.resize(img, (width, height))
        trainingArray[i, :] = np.array(img, dtype='float64').flatten()
    return trainingArray

def helper():
    # TODO : Add additional parameters
    print('The following parameters can be manipulated: \n 1. Height of Image \n 2. Width of Image')

def get_mean_face(trainingImageVector, training_images_names, width, height):
    mean_face_vec = np.zeros((1,height*width))
    for i in trainingImageVector:
        mean_face_vec = np.add(mean_face_vec,i)
    mean_face_vec = np.divide(mean_face_vec,float(len(training_images_names))).flatten()
    return mean_face_vec

def get_normalised_face(trainingImageVector, training_images_names, width, height, mean_face):
    normalised_image_vector = np.ndarray(shape=(len(training_images_names), height*width))
    for i in range(len(training_images_names)):
        normalised_image_vector[i] = np.subtract(trainingImageVector[i],mean_face)
    return normalised_image_vector

def get_covariance_matrix(normalised, mval = 8.0):
    covariance_matrix = np.cov(normalised)
    covariance_matrix = np.divide(covariance_matrix, mval)
    return covariance_matrix

def get_eigen(covariance_matrix):
    eigenValues, eigenVectors, = np.linalg.eig(covariance_matrix)
    eigenpairs = [(eigenValues[index], eigenVectors[:,index]) for index in range(len(eigenValues))]
    eigenpairs.sort(reverse=True)
    eigenValuesSort  = [eigenpairs[index][0] for index in range(len(eigenValues))]
    eigenVectorsSort = [eigenpairs[index][1] for index in range(len(eigenValues))]
    return eigenValuesSort, eigenVectorsSort


def faceRecognition(img, train_image_names,imgProjected,imgWeight, width, height, mean_face, NormalVector, threshold=7500000):
    global count,numOfTotalImages,correctFacePredictions
    count        = 0
    numOfTotalImages   = 0
    correctFacePredictions = 0


    unknownFace = cv2.imread('test/' + img, 0)
    unknownFace = cv2.resize(unknownFace, (width, height))
    unknownFace = cv2.equalizeHist(unknownFace)
 
    numOfTotalImages          += 1
    unknownFaceVector = np.array(unknownFace, dtype='float64').flatten()
    normalisedFaceVector = np.subtract(unknownFaceVector,mean_face)
    
    count+=1
    
    unknownImageWeight = np.dot(imgProjected, normalisedFaceVector)
    changeOfWeights  = imgWeight - unknownImageWeight
    norms = np.linalg.norm(changeOfWeights, axis=1)
    minNormIndex = np.argmin(norms)

    print('\n Testing : {0} \n Norms : {1}'.format(img, norms[minNormIndex]))
    NormalVector.append(norms[minNormIndex])

    print(train_image_names[minNormIndex])
    if(norms[minNormIndex] < threshold):
        if img.split('.')[0] == train_image_names[minNormIndex].split('.')[0]:
            correctFacePredictions += 1
        else:
            None
            # print('Unknown Face')
    else:
        print('Unkown Face')

    count+=1
    return correctFacePredictions, numOfTotalImages

def main():
    # helper()
    width  = 32
    height = 32
    inputNumEigVec = input('Enter number of eigen vectors:') or 7
    threshold = input('Enter a threshold value(default = 7500000):') or 7500000
    threshold = int(threshold)
    inputNumEigVec = int(inputNumEigVec)
    trainingImageVector = get_vector_from_img('train/', arrayOfImageName('train'), width, height)
    mean_face = get_mean_face(trainingImageVector, arrayOfImageName('train'), width, height)
    normalisedImage = get_normalised_face(trainingImageVector, arrayOfImageName('train'), width, height, mean_face)
    covariance_matrix = get_covariance_matrix(normalisedImage)
    eigenValues, eigenVectors = get_eigen(covariance_matrix)

#    Varsum = np.cumsum(eigenValues)/sum(eigenValues)
    eigVecTranspose = np.array(eigenVectors[:inputNumEigVec]).transpose()
    imgProjected = np.dot(trainingImageVector.transpose(),eigVecTranspose)
    imgProjected = imgProjected.transpose()
    imgWeight = np.array([np.dot(imgProjected,i) for i in normalisedImage])

    NormalVector = []
    test_image_names = arrayOfImageName('test')
    train_image_names = arrayOfImageName('train')
    correctFacePrediction = 0
    numImages = 0

    userinput=int(input('Enter \n 1. Real time capture \n 2. Predefined image :'))
    if(userinput==1):
        try: 
            os.remove("preprocess/frame1.jpg")
        except:
            pass
        
        vidcap = cv2.VideoCapture(0)
        success,image = vidcap.read()
        count = 0
        if(count==0):
            success = True
            count+=1   
        while success:
            success,image = vidcap.read()
            cv2.imwrite("preprocess/frame%d.jpg" % count, image)
            vidcap.release()
            if(count>0):
                success=False 
        # img=cv2.imread('preprocess/frame1.jpg', 0)
        crop_image_to_face('preprocess/frame1.jpg', 'haarcascade_frontalcatface.xml', 32, 32)

        correctFacePrediction, nuimg = faceRecognition('frame1.jpg', train_image_names,imgProjected,imgWeight, width, height, mean_face, NormalVector, threshold)
                
    if(userinput==2):
        for i in range(len(test_image_names)):
            correctFacePrediction2, numImages2 = faceRecognition(test_image_names[i], train_image_names,imgProjected,imgWeight, width, height, mean_face, NormalVector, threshold)
            correctFacePrediction += correctFacePrediction2
            numImages += numImages2
        print('Correct predictions: {}/{} = {}%'.format(correctFacePrediction, numImages, correctFacePrediction/numImages*100.00))


if __name__ == '__main__':
    main()
