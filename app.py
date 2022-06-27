import os
from flask import Flask, render_template, flash, request, redirect, url_for
from flask_mysqldb import MySQL

import collections
from numpy import dot
from numpy.linalg import norm
import numpy as np
import glob
import random
import imageio
import PIL
import cv2
import pandas as pd
# import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image, erosion
from skimage.morphology import square
import matplotlib.image as mpimg
import skimage
import math
from scipy.ndimage import convolve
from PIL import Image, ImageFilter
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '1234'
app.config['MYSQL_DB'] = 'csdldptn4_final'

mysql = MySQL(app)


UPLOAD_FOLDER = './static/data/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_all_row():
    cursor = mysql.connection.cursor()
    cursor.execute("select file_tbl.ID, file_tbl.filename, file_tbl.filepath, file_tbl.label, featurbif_tbl.vector as bfeature, featurterm_tbl.vector as tfeature from file_tbl, featurbif_tbl, featurterm_tbl where file_tbl.ID = featurbif_tbl.IdFile and file_tbl.ID=featurterm_tbl.IdFile")
    mysql.connection.commit()
    items = cursor.fetchall()
    cursor.close()
    return items

def row_to_list():
    items = get_all_row()
    all_sample_min = []
    count = 0
    for item in items:
        print(item[1])
        get_split_bif = item[4].split(' ')
        get_split_term = item[5].split(' ')
        sampleMin = []
        for s in get_split_bif:
            temp = s.split(',')
            locX = float(temp[0])
            locY = float(temp[1])
            Orientation = float(temp[2])
            sampleMin.append(MinutiaeFeature(locX, locY, Orientation, 'B', 0))
            # print(locX, locY, Orientation, computeDistance(locY, locX))

        for s in get_split_term:
            temp = s.split(',')
            locX = float(temp[0])
            locY = float(temp[1])
            Orientation = float(temp[2])
            sampleMin.append(MinutiaeFeature(locX, locY, Orientation,'T', 0))
            # print(locX, locY, Orientation, computeDistance(locY, locX))
        sumX=0
        sumY=0
        for element in sampleMin:
            sumX += element.locX
            sumY += element.locY
        print(len(sampleMin))
        sumX = sumX/(len(sampleMin))
        sumY = sumY/(len(sampleMin))
        print(sumX, sumY)
        for element in sampleMin:
            element.offset = computeDistance(element.locY, element.locX, sumY, sumX)
        # q = ""
        sampleMin.sort(key= lambda x:(x.offset, x.locX))
        # for element in sampleMin:
        #     q+= (str(element.locX)+','+ str(element.locY)+','+str(element.Orientation) + "," + str(element.offset)+" ")
        # print(q)

        all_sample_min.append(sampleMin)

    return all_sample_min

def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5

def min_max_scaler(x, min, max, new_min, new_max):
    return (((x-min)/(max-min))*(new_max-new_min))+(new_min)

class GetResult(object):
    def __init__(self,index, sum):
        self.index = index
        self.sum = sum

def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num

def translation(input_vector, x, y, z):
    for element in input_vector:
        element.locX += (x*5)
        element.locY += (y*5)
        
        element.locX = element.locX*(math.cos(z*(math.pi/12))) - element.locY*(math.sin(z*(math.pi/12)))
        element.locY = element.locX*(math.sin(z*(math.pi/12))) + element.locY*(math.cos(z*(math.pi/12)))
    return input_vector

@app.route('/uploadimage', methods=['GET', 'POST'])
def upload_image_page():
    items = get_all_row()
    list_vector_in_db = row_to_list()
    print(len(list_vector_in_db))
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # resultTerm, resultBif = feature(path)
            input_vector = feature(path)
            if(len(input_vector)==0):
                result = "Ảnh không phải ảnh vân tay"
                id_most_relevant = "--"
                path_most_relevant = "--"
                euclidean_distance = "--"
                return render_template('uploadimage.html', result=result, path = path, path_most_relevant = path_most_relevant, id_most_relevant=id_most_relevant, euclidean_distance=euclidean_distance)
            knn = []
            for k in range(-2,3):
                for j in range(-2, 3):
                    for l in range(-2,3):
                        shifted_vector = translation(input_vector, k, j, l)
                        index = 0
                        
                        for vector in list_vector_in_db:
                            sum=0
                            for i in range(30):
                                # a = [min_max_scaler(vector[i].locX, 0, 103), min_max_scaler(vector[i].locY, 0, 96), min_max_scaler(vector[i].Orientation, -180, 180)]
                                # b = [min_max_scaler(input_vector[i].locX, 0, 103), min_max_scaler(input_vector[i].locY, 0, 96), min_max_scaler(input_vector[i].Orientation, -180, 180)]
                                if vector[i].Type == 'B':
                                    a = [min_max_scaler(vector[i].locX, 0, 103,-1,1), min_max_scaler(vector[i].locY, 0, 96,-1,1), min_max_scaler(vector[i].Orientation, -180, 180,-1,1), -0.5]
                                if vector[i].Type == 'T':
                                    a = [min_max_scaler(vector[i].locX, 0, 103,-1,1), min_max_scaler(vector[i].locY, 0, 96,-1,1), min_max_scaler(vector[i].Orientation, -180, 180,-1,1), 0.5]
                                if input_vector[i].Type == 'B':
                                    b = [min_max_scaler(shifted_vector[i].locX, 0, 103,-1,1), min_max_scaler(shifted_vector[i].locY, 0, 96,-1,1), min_max_scaler(shifted_vector[i].Orientation, -180, 180,-1,1), -0.5]
                                if input_vector[i].Type == 'T':
                                    b = [min_max_scaler(shifted_vector[i].locX, 0, 103,-1,1), min_max_scaler(shifted_vector[i].locY, 0, 96,-1,1), min_max_scaler(shifted_vector[i].Orientation, -180, 180,-1,1), 0.5]
                                c_similarity = dot(a, b)/(norm(a)*norm(b))
                                sum+=c_similarity
                            sum/=30
                            knn.append(GetResult(index, sum))
                            index += 1
            knn.sort(key= lambda x:x.sum, reverse=True)
            result_label = []
            for i in range(1):
                result_label.append(items[knn[i].index][3])
                print(items[knn[i].index][3])
            # if knn[0].sum >=-0.5 and  knn[0].sum< -0.2:
            #     result = "Ảnh không phải ảnh vân tay"
            #     id_most_relevant = "--"
            #     path_most_relevant = "--"
            #     euclidean_distance = "--"
            #     return render_template('uploadimage.html', result=result, path = path, path_most_relevant = path_most_relevant, id_most_relevant=id_most_relevant, euclidean_distance=euclidean_distance)
            # if knn[0].sum <= 0.5 and knn[0].sum >= -0.2:
            #     result = "Ảnh vân tay chưa tồn tại trong cơ sở dữ liệu"
            #     id_most_relevant = "--"
            #     path_most_relevant = "--"
            #     euclidean_distance = "--"
            #     return render_template('uploadimage.html', result=result, path = path, path_most_relevant = path_most_relevant, id_most_relevant=id_most_relevant, euclidean_distance=euclidean_distance)
            # elif knn[0].sum <=0.9:
            #     result = ("Ảnh vân tay có khả năng nằm trong cơ sở dữ liệu: " + most_frequent(result_label))
            #     id_most_relevant = items[knn[0].index][0]
            #     path_most_relevant = items[knn[0].index][2]
            #     euclidean_distance = knn[0].sum
            #     return render_template('uploadimage.html', result=result, path = path, path_most_relevant = path_most_relevant, id_most_relevant=id_most_relevant, euclidean_distance=euclidean_distance)
            # else:
            id_most_relevant = items[knn[0].index][0]
            path_most_relevant = items[knn[0].index][2]
            similarity = knn[0].sum
            print(path_most_relevant)
            result = most_frequent(result_label)
            print(result)
                # return render_template('uploadimage.html', resultTerm=resultTerm, resultBif=resultBif, path = path, filename = filename)
            return render_template('uploadimage.html', result=result, path = path, path_most_relevant = path_most_relevant, id_most_relevant=id_most_relevant, similarity=similarity) 
    return render_template('uploadimage.html')


@app.route("/")
def index_page():
    items = get_all_row()
    return render_template('./index.html', items=items)

# Ridge_detection


def detect_ridges(gray, sigma=0.5):
    H_elems = hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges

# Minutiae_detection


def getTerminationBifurcation(img, mask):
    img = img == 255
    (rows, cols) = img.shape
    minutiaeTerm = np.zeros(img.shape)
    minutiaeBif = np.zeros(img.shape)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if(img[i][j] == 1):
                if(i > 10 and i < 195):
                    if(j > 10 and j < 180):
                        block1 = img[i-1][j-1].astype(np.float32)
                        block2 = img[i][j-1].astype(np.float32)
                        block3 = img[i+1][j-1].astype(np.float32)
                        block4 = img[i+1][j].astype(np.float32)
                        block5 = img[i+1][j+1].astype(np.float32)
                        block6 = img[i][j+1].astype(np.float32)
                        block7 = img[i-1][j+1].astype(np.float32)
                        block8 = img[i-1][j].astype(np.float32)

                        block9 = np.abs(np.subtract(block1, block2))
                        block10 = np.abs(np.subtract(block2, block3))
                        block11 = np.abs(np.subtract(block3, block4))
                        block12 = np.abs(np.subtract(block4, block5))
                        block13 = np.abs(np.subtract(block5, block6))
                        block14 = np.abs(np.subtract(block6, block7))
                        block15 = np.abs(np.subtract(block7, block8))
                        block16 = np.abs(np.subtract(block8, block1))
                        blk_val = (block9+block10+block11 + block12 +
                                   block13 + block14 + block15 + block16)/2
                        if(blk_val == 1):
                            minutiaeTerm[i, j] = 1

                        elif(blk_val == 3):
                            minutiaeBif[i, j] = 1

    mask = convex_hull_image(mask > 0)
    mask = erosion(mask, square(2))
    minutiaeTerm = np.uint8(mask)*minutiaeTerm
    return(minutiaeTerm, minutiaeBif)

# Main_process


class MinutiaeFeature(object):
    def __init__(self, locX, locY, Orientation, Type, offset):
        self.locX = locX
        self.locY = locY
        self.Orientation = Orientation
        self.Type = Type
        self.offset = offset

# class SampleMinutiaeFeture(object):
#     def __init__(self, locX, locY, Orientation):
#         self.locX = locX
#         self.locY = locY
#         self.Orientation = Orientation

def computeDistance(col, row, sumY, sumX):
     return np.abs(col-sumY)+np.abs(row-sumX)

def computeAngle(block, minutiaeType):
    angle = 0
    (blkRows, blkCols) = np.shape(block)
    CenterX, CenterY = (blkRows-1)/2, (blkCols-1)/2
    if(minutiaeType.lower() == 'termination'):
        sumVal = 0
        for i in range(blkRows):
            for j in range(blkCols):
                if((i == 0 or i == blkRows-1 or j == 0 or j == blkCols-1) and block[i][j] != 0):
                    angle = -math.degrees(math.atan2(i-CenterY, j-CenterX))
                    break
        return(angle)
    elif(minutiaeType.lower() == 'bifurcation'):
        (blkRows, blkCols) = np.shape(block)
        CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
        angle = 0
        sumVal = 0
        for i in range(blkRows):
            for j in range(blkCols):
                if ((i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0):
                    angle= -math.degrees(math.atan2(i -CenterY, j - CenterX))
                    break
        # if(sumVal != 3):
        #     angle = 0
        return(angle)


def extractMinutiaeFeatures(skel, minutiaeTerm, minutiaeBif):
    FeaturesTerm = []

    minutiaeTerm = skimage.measure.label(minutiaeTerm, connectivity=2)
    RP = skimage.measure.regionprops(minutiaeTerm)

    WindowSize = 2
    FeaturesTerm = []
    for i in RP:
        (row, col) = np.int16(np.round(i['Centroid']))
        block = skel[row-WindowSize:row+WindowSize +
                     1, col-WindowSize:col+WindowSize+1]
        angle = computeAngle(block, 'Termination')
        FeaturesTerm.append(MinutiaeFeature(round(row/2,1), round(col/2,1), round(angle,1), 'T', 0))
    FeaturesBif = []    
    minutiaeBif = skimage.measure.label(minutiaeBif, connectivity=2)
    RP = skimage.measure.regionprops(minutiaeBif)
    WindowSize = 1
    for i in RP:
        (row, col) = np.int16(np.round(i['Centroid']))
        block = skel[row-WindowSize:row+WindowSize +
                     1, col-WindowSize:col+WindowSize+1]
        angle = computeAngle(block, 'Bifurcation')
        FeaturesBif.append(MinutiaeFeature(round(row/2,1), round(col/2,1), round(angle,1), 'B', 0))
    
    featureVector = FeaturesBif+FeaturesTerm
    print(len(featureVector))
    if(len(featureVector)<30):
        featureVector=[]
        return featureVector

    sumX=0
    sumY=0
    for element in featureVector:
        sumX += element.locX
        sumY += element.locY
    sumX = sumX/(len(featureVector))
    sumY = sumY/(len(featureVector))
    for element in featureVector:
        element.offset = computeDistance(element.locY, element.locX, sumY, sumX)

    featureVector.sort(key= lambda x:x.offset)
    cutVector = featureVector[0:30]
    print(len(cutVector))
    sumX=0
    sumY=0
    for element in cutVector:
        sumX += element.locX
        sumY += element.locY
    sumX = sumX/(len(cutVector))
    sumY = sumY/(len(cutVector))
    print(sumX, sumY)
    for element in cutVector:
        element.offset = computeDistance(element.locY, element.locX, sumY, sumX)

    cutVector.sort(key= lambda x:(x.offset, x.locX))
    # cutVector.sort(key= lambda x:x.Type)
    resultBif = ""
    resultTerm = ""
    result=""
    for v in cutVector:
        if v.Type == 'T':
            resultTerm += (str(v.locX)+','+ str(v.locY)+','+str(v.Orientation) + " ")
        if v.Type == 'B':
            resultBif += (str(v.locX)+','+ str(v.locY)+','+str(v.Orientation)+ ' ')
        result += (str(v.locX)+','+ str(v.locY)+','+str(v.Orientation) +','+str(v.offset) +' ')
    
    # print(resultBif)
    # print('\n================================')
    # print(resultTerm)
    # print('\n================================')
    # print(result)
    # return resultTerm, resultBif
    return cutVector

def feature(dir):
    img_name = dir
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    k = 0.5
    width = int((img.shape[1])/k)
    height = int((img.shape[0]/k))
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    # gaussian_filter
    g = cv2.GaussianBlur(resized_img, (5, 5), 0)
    # median_filter
    m = cv2.medianBlur(g, 1)
    # detect_ridge
    a, b = detect_ridges(m, sigma=0.2)
    # use threshold
    th3 = cv2.adaptiveThreshold(m, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    a = np.array(a >= th3).astype(int)
    # skeletionize
    skel = skimage.morphology.skeletonize(a)
    skel = np.uint8(skel)*255
    # apply mask
    mask = a*255
    # extract_minutiae
    (minutiaeTerm, minutiaeBif) = getTerminationBifurcation(skel, mask)
    input_vector = extractMinutiaeFeatures(
        skel, minutiaeTerm, minutiaeBif)
    print(len(input_vector))    
    return input_vector
