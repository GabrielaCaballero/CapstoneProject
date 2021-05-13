import os
import cv2
import random
import numpy as np
import tensorflow as tf
import pytesseract
from core.utils import read_class_names
from core.config import cfg
from scipy.spatial import distance as dist
from collections import OrderedDict

nextObjectID = 0
objects = OrderedDict()
disappeared = OrderedDict()
maxDisappeared = 50
final_output = dict()
total_output = dict()

# function to count objects, can return total classes or count per class
def count_objects(data, by_class = False, allowed_classes = list(read_class_names(cfg.YOLO.CLASSES).values())):
    boxes, scores, classes, num_objects = data

    #create dictionary to hold count of objects
    counts = dict()

    # if by_class = True then count objects per class
    if by_class:
        class_names = read_class_names(cfg.YOLO.CLASSES)
        # loop through total number of objects found
        for i in range(num_objects):
            # grab class index and convert into corresponding class name
            if scores[i] >= 0.9:
                class_index = int(classes[i])
                class_name = class_names[class_index]
                if class_name in allowed_classes:
                    if (class_name in final_output) == False:
                      final_output.update({class_name : 1})
                      print(final_output)                
                    objects = OrderedDict()
                    objects = update(boxes,class_name)
                    #final_output.update({ class_name: len(objects)})
                    #print(class_name)
                    print(objects)
                    #print(final_output)
                    #counts[class_name] = counts.get(class_name, 0) + 1
                    total_output.update(final_output)
                else:
                    continue
            else:
                continue
        print(final_output)

    # else count total objects found
    else:
        counts['total object'] = num_objects
    
    return counts

# function for cropping each detection and saving as new image
def crop_objects(img, data, path, allowed_classes):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    #create dictionary to hold count of objects for image name
    counts = dict()
    for i in range(num_objects):
        # get count of class for part of image name
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name in allowed_classes:
            counts[class_name] = counts.get(class_name, 0) + 1
            # get box coords
            xmin, ymin, xmax, ymax = boxes[i]
            # crop detection from image (take an additional 5 pixels around all edges)
            cropped_img = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
            # construct image name and join it to path for saving crop properly
            img_name = class_name + '_' + str(counts[class_name]) + '.png'
            img_path = os.path.join(path, img_name )
            # save image
            cv2.imwrite(img_path, cropped_img)
        else:
            continue
        
# function to run general Tesseract OCR on any detections 
def ocr(img, data):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    for i in range(num_objects):
        # get class name for detection
        class_index = int(classes[i])
        class_name = class_names[class_index]
        # separate coordinates from box
        xmin, ymin, xmax, ymax = boxes[i]
        # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
        box = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
        # grayscale region within bounding box
        gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
        # threshold the image using Otsus method to preprocess for tesseract
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # perform a median blur to smooth image slightly
        blur = cv2.medianBlur(thresh, 3)
        # resize image to double the original size as tesseract does better with certain text size
        blur = cv2.resize(blur, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
        # run tesseract and convert image text to string
        try:
            text = pytesseract.image_to_string(blur, config='--psm 11 --oem 3')
            print("Class: {}, Text Extracted: {}".format(class_name, text))
        except: 
            text = None
            
def deregister(objectID,class_name):
    # to deregister an object ID we delete the object ID from
    # both of our respective dictionaries
    global objects 
    global disappeared
    del objects[objectID]
    del disappeared[objectID]
    final_output[class_name] = final_output.get(class_name) -1
            
def register(centroid,class_name):
        # when registering an object we use the next available object
        # ID to store the centroid
        global objects 
        global disappeared
        global nextObjectID
        
        #centroides = {'centroid': centroid}
        #objects[nextObjectID] = list(centroid)
        centroides = []
        centroides = centroid
        objects[nextObjectID] = list()
        objects[nextObjectID].append(centroides)
        disappeared[nextObjectID] = 0
        #object_type = {'class_name' : class_name}
        #objects[nextObjectID].update(object_type)
        print(nextObjectID)
        objects[nextObjectID].append(class_name)
        print(objects[nextObjectID])
        nextObjectID += 1
        final_output[class_name] = final_output.get(class_name) +1
    
def update(rects,class_name):
    # check to see if the list of input bounding box rectangles
    # is empty
    global objects 
    global disappeared
    global nextObjectID
    global maxDisappeared
    count = 0
    if len(rects) == 0:
        # loop over any existing tracked objects and mark them
        # as disappeared
        for objectID in list(disappeared.keys()):
            disappeared[objectID] += 1
            # if we have reached a maximum number of consecutive
            # frames where a given object has been marked as
            # missing, deregister it
            if disappeared[objectID] > maxDisappeared:
                deregister(objectID,class_name)
        # return early as there are no centroids or tracking info
        # to update
        return objects
    # initialize an array of input centroids for the current frame
    inputCentroids = np.zeros((len(rects), 2), dtype="int")
    # loop over the bounding box rectangles
    for (i, (startX, startY, endX, endY)) in enumerate(rects):
        # use the bounding box coordinates to derive the centroid
        cX = int((startX + endX) / 2.0)
        cY = int((startY + endY) / 2.0)
        inputCentroids[i] = (cX, cY)
    # if we are currently not tracking any objects take the input
    # centroids and register each of them
    if len(objects) == 0:
        for i in range(0, len(inputCentroids)):
            register(inputCentroids[i],class_name)
    # otherwise, are are currently tracking objects so we need to
    # try to match the input centroids to existing object
    # centroids
    else:
        # grab the set of object IDs and corresponding centroids
        objectIDs = list(objects.keys())
        objectCentroidsList = list(objects.values())
        objectCentroids = list()
        #objectCentroids = np.zeros((len(objectCentroidsList), 2), dtype="int")
        #objectCentroids = numpy.empty_like(inputCentroids)
        for d in objectCentroidsList:
          print(d[0])
          objectCentroids.append(d[0])
        #objectCentroids.append([d[0] for d in objectCentroids])
        objectCentroids = np.array(objectCentroids)
        print("object centroids are: ")
        print(objectCentroids)
        print("input centroids are:")
        print(inputCentroids)
        # compute the distance between each pair of object
        # centroids and input centroids, respectively -- our
        # goal will be to match an input centroid to an existing
        # object centroid
        D = dist.cdist(np.array(objectCentroids), inputCentroids)
        # in order to perform this matching we must (1) find the
        # smallest value in each row and then (2) sort the row
        # indexes based on their minimum values so that the row
        # with the smallest value is at the *front* of the index
        # list
        rows = D.min(axis=1).argsort()
        # next, we perform a similar process on the columns by
        # finding the smallest value in each column and then
        # sorting using the previously computed row index list
        cols = D.argmin(axis=1)[rows]
        # in order to determine if we need to update, register,
        # or deregister an object we need to keep track of which
        # of the rows and column indexes we have already examined
        usedRows = set()
        usedCols = set()
        # loop over the combination of the (row, column) index
        # tuples
        for (row, col) in zip(rows, cols):
            # if we have already examined either the row or
            # column value before, ignore it
            # val
            if row in usedRows or col in usedCols:
                continue
            # otherwise, grab the object ID for the current row,
            # set its new centroid, and reset the disappeared
            # counter
            objectID = str(objectIDs[row])
            objects[objectID] = inputCentroids[col]
            disappeared[objectID] = 0
      
            # indicate that we have examined each of the row and
            # column indexes, respectively
            usedRows.add(row)
            usedCols.add(col)
        # compute both the row and column index we have NOT yet
        # examined
        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        unusedCols = set(range(0, D.shape[1])).difference(usedCols)
        # in the event that the number of object centroids is
        # equal or greater than the number of input centroids
        # we need to check and see if some of these objects have
        # potentially disappeared
        if D.shape[0] >= D.shape[1]:
            # loop over the unused row indexes
            for row in unusedRows:
                # grab the object ID for the corresponding row
                # index and increment the disappeared counter
                objectID = objectIDs[row]
                disappeared[objectID] += 1
                # check to see if the number of consecutive
                # frames the object has been marked "disappeared"
                # for warrants deregistering the object
                if disappeared[objectID] > maxDisappeared:
                    deregister(objectID,class_name)
        # otherwise, if the number of input centroids is greater
        # than the number of existing object centroids we need to
        # register each new input centroid as a trackable object
        else:
            for col in unusedCols:
                register(inputCentroids[col],class_name)
                
    # return the set of trackable objects
    return objects

def count_objects_passing_line(data, by_class = False, allowed_classes = list(read_class_names(cfg.YOLO.CLASSES).values())):
    boxes, scores, classes, num_objects = data

   
    #create dictionary to hold count of objects
    counts = dict()

    # if by_class = True then count objects per class
    if by_class:
        class_names = read_class_names(cfg.YOLO.CLASSES)

        # loop through total number of objects found
        for i in range(num_objects):
            # grab class index and convert into corresponding class name
            if scores[i] >= 0.9:
                class_index = int(classes[i])
                class_name = class_names[class_index]
                if class_name in allowed_classes:
                    counts[class_name] = counts.get(class_name, 0) + 1
                else:
                    continue

    # else count total objects found
    else:
        counts['total object'] = num_objects
    
    return counts            