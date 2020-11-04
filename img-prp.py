import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
from scipy.ndimage import variance
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import laplace
from skimage.transform import resize
from sklearn import preprocessing, svm
import numpy as np
import matplotlib.pyplot as plt
import cv2





    
def ckb(img): 
    img = rgb2gray(img)
    edge_laplace = laplace(img, ksize=3)
    y_pred=logistic_regression.predict([[variance(edge_laplace), np.amax(edge_laplace)]])
    print(str(y_pred[0]))
 
def ed(lc):
    source = cv2.imread(lc)

    scaleX = 0.6
    scaleY = 0.6
    scaleDown = cv2.resize(source, None, fx= scaleX, fy= scaleY, interpolation= cv2.INTER_LINEAR)
    scaleUp = cv2.resize(source, None, fx= scaleX*3, fy= scaleY*3, interpolation= cv2.INTER_LINEAR)
    crop = source[50:150,20:200]

    cv2.imshow("Original", source)
    cv2.imshow("Scaled Down", scaleDown)
    cv2.imshow("Scaled Up", scaleUp)
    cv2.imshow("Cropped Image",crop) 
    
    im_rotate = cv2.rotate(source, cv2.cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow("Rotated image",im_rotate) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def prp(image):
    h, w = image.shape[:2] 
    print("Height = {},  Width = {}".format(h, w)) 
    (B, G, R) = image[100, 100] 
    print("R = {}, G = {}, B = {}".format(R, G, B)) 
  
    
    
def main():
    lc=input("enter photo location ")
    img = cv2.imread(lc)
    cv2.imshow("photo",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    op = {1: "check for blur", 2: 'properties', 3: 'editing'}
    op1=int(input(str(op)+" "))
    if op1==1:
        ckb(img)
    if op1==2:
        prp(img)
    if op1==3:
        ed(lc)
        
        
if __name__=="__main__":
    df=pd.read_csv("pic.csv") 
    X=df[['variance','maximum']]
    Y=df['det']
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state=0)
    logistic_regression= LogisticRegression()
    logistic_regression.fit(X,Y)
    Y_pred=logistic_regression.predict(X_test)
    print('Accuracy: ',metrics.accuracy_score(Y_test, Y_pred))
    main()
    