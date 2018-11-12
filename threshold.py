import numpy as np
import cv2, warnings
from matplotlib import pyplot as plt
warnings.filterwarnings("error")

########################################### mean method
def mean_threshold(gray):
    T=127
    mean1=0
    mean2=0
    h,w = gray.shape
    while 1:
        c1=c2=count1=count2=0
        for i in range(h):
            for j in range(w):
                if gray[i,j] > T:
                    c1+=gray[i,j]; count1+=1
                else:
                    c2+=gray[i,j]; count2+=1
        newmean1 = c1/float(count1)
        newmean2 = c2/float(count2)
        Tnew = int((newmean1 + newmean2)/2)
        if newmean1 == mean1 and newmean2 == mean2:
            break
        T=Tnew
        mean1 = newmean1
        mean2 = newmean2
    return T
#######################################################

################### otsu threshold ####################
def otsu_threshold(gray):
    h,w = gray.shape
    count = {i:0 for i in range(256)}
    for i in range(h):
        for j in range(w):
            count[gray[i,j]] += 1
    prob = np.array([count[i]/float(h*w) for i in sorted(count)])
    means = np.array([prob[i]*(i+1) for i in count])
    mean = np.sum(means)
    minvar = -np.inf
    minT = 0
    for t in range(256):
        w1 = np.sum([i for i in prob[:t+1]])
        w2 = 1.0-w1
        if not w1 or not w2: continue
        m1 = np.sum([i for i in means[:t+1]])
        mean1 = m1/w1
        mean2 = (mean - m1)/w2
        bcvar = w1*w2*(mean2-mean1)**2
        if bcvar > minvar:
            minvar = bcvar
            minT = t
    return minT
#######################################################

if __name__ == "__main__":
    img = cv2.imread('test.jpg',cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    img = cv2.resize(img,(int(.1*w), int(.1*h)), interpolation = cv2.INTER_CUBIC)

    h, w = img.shape[:2]
    gray = np.array([[img[i,j,2] for j in range(w)]for i in range(h)])
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray',gray)

    # ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    T = otsu_threshold(gray=gray)
    thresh = np.array([[0 if gray[i,j]<T else 255 for j in range(w)]for i in range(h)], dtype=np.uint8)
    cv2.imshow('thresh',thresh)
    cv2.waitKey(0)
