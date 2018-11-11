## this file contain code for binary classification i.e. good or not good grain

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from util import display_mask, get_boundry_img_matrix
from PCA import pca
import numpy as np, keras, cv2
from segment_formation_v6 import segment_image4

if __name__ == "__main__":
    color = {i: np.random.randint(20, 255, 3) for i in range(5, 5000)}
    color[1] = [255, 255, 255]
    color[2] = [0, 0, 255]
    imgFile =  raw_input("Enter the file(wheat image) location to dectect : ")
    # imgFile = 'test.jpg'
    count = 1

    model = keras.models.load_model('weights_results_2out/weights_01234567.pkl')

    # for imgFile in imgFile:
    print "Segmentation in process..."
    segments, segLocation, _, mask= segment_image4(imgFile)
    print "Segmentation in Complete."


    features = {}
    print "Feature extraction in process..."
    for gi in segments:
        gcolor = segments[gi]
        h, w, _ = gcolor.shape
        ggray = gcolor[:,:,2]
        thresh = np.array([[255 if pixel > 20 else 0 for pixel in row] for row in ggray])
        b = np.array(get_boundry_img_matrix(thresh, bval=1), dtype=np.float32)
        boundry = np.sum(b) / (h * w)
        area = np.sum(np.sum([[1.0 for j in range(w) if ggray[i, j]] for i in range(h)]))
        mean_area = area / (h * w)
        r, b, g = np.sum([gcolor[i, j] for j in range(w) for i in range(h)], axis=0) / (area * 256)
        _, _, eigen_value = pca(ggray)
        eccentricity = eigen_value[0] / eigen_value[1]
        l = [mean_area, boundry, r, b, g, eigen_value[0], eigen_value[1], eccentricity]
        features[gi] = np.array(l)
    print "Feature extraction in complete."

    out = {}
    for i in features:
        out[i] = model.predict(np.array([features[i]]))

    rect = cv2.imread(imgFile, cv2.IMREAD_COLOR)
    good = not_good = 0
    for i in out:
        try:
            s = segLocation[i]
        except KeyError:
            print "Key Error"
            continue
        if np.argmax(out[i][0]) == 0:
            good += 1
            rect = cv2.rectangle(rect, (s[2], s[0]), (s[3], s[1]), (0, 0, 0), 1)
        else:
            not_good+=1
            rect = cv2.rectangle(rect, (s[2], s[0]), (s[3], s[1]), (0, 0, 255), 3)
    print "Number of good grain :", good
    print "Number Not good grain or imputity:", not_good

    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w, _ = rect.shape
    cv2.putText(rect, text='Number of good grain: %d  Number Not good grain or imputity: %d'%(good,not_good), org=(10, h - 50), fontScale=1, fontFace=font, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    maskFile = 'mask_'+imgFile.split('/')[-1]
    outFile = 'result_'+imgFile.split('/')[-1]
    cv2.imwrite(outFile, rect)
    display_mask('mask',mask,sname=maskFile)
    cv2.waitKey(0)
    count+=1