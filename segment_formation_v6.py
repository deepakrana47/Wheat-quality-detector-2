# import warnings
# warnings.filterwarnings("ignore", message="numpy.dtype size changed")
# warnings.simplefilter("ignore", DeprecationWarning)

from tensorflow import keras

from threshold import otsu_threshold
from Area import areaThreshold_by_havg
from _8connected import get_8connected_v2
from util_ import *
from L2_Segmentation_v5 import L2_segmentation_2
import time

mFile = 'segmentation_data/weights_30_30_.h5'
model = keras.models.load_model(mFile)
# model.summary()
rm_detail = open('log.txt', 'a')


def isMoregrain(iimg, T):
    iimg = generate_newcolorimg_by_padding(iimg, 30, 30)[:, :, 2]
    gray = np.array([[1 if pixel >= T else 0 for pixel in row] for row in iimg], dtype=np.uint8)
    boundry = np.array([get_boundry_img_matrix(gray, 1).reshape(30, 30, 1)], dtype=np.float32)
    return 1 if np.argmax(model.predict(boundry)) == 0 else 0

def get_img_value_inRange(img, mask, sindex, s):
    return np.array([[img[i, j] if mask[i, j] == sindex else [0, 0, 0] for j in range(s[2], s[3])] for i in range(s[0], s[1])], dtype=np.uint8)

def remove_mask(mask, val, mrange):
    mask[mrange[0]:mrange[1], mrange[2]:mrange[3]] = [[0 if pixel == val else pixel for pixel in row] for row in mask[mrange[0]:mrange[1], mrange[2]:mrange[3]]]
    return mask

def segment_image4(img_file, dlog=0):
    t0 = time.time()
    org = cv2.imread(img_file, cv2.IMREAD_COLOR)
    h, w = org.shape[:2]

    img=org.copy()
    # print("Reading ",time.time()-t0)
    t0 = time.time()
    
    # removing noise by using Non-local Means Denoising algorithm
    img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    # cv2.imshow('cleaned',img)
    # print("noise removing ",time.time()-t0)
    t0 = time.time()

    # Taking the red component out of RBG image as it is less effected by shadow of grain or impurity
    gray = np.array([[pixel[2] for pixel in row]for row in img])
    # cv2.imshow('gray',gray)

    # calculating threshold value by using otsu thresholding
    T = otsu_threshold(gray=gray)
    # print("threshold calc ",time.time()-t0)
    t0 = time.time()

    # incresing contrast about the threshold
    # gray = np.array([[max(pixel - 25, 0) if pixel < T else min(pixel + 25, 255) for pixel in row] for row in gray], dtype=np.uint8)
    # cv2.imshow('contrast',gray)
    # print("Increasing contrast ",time.time()-t0)
    # t0 = time.time()

    # generating a threshold image
    thresh = np.array([[0 if pixel<T else 255 for pixel in row]for row in gray], dtype=np.uint8)
    # cv2.imshow('Threshold',thresh)
    # print("Generating Threshold ",time.time()-t0)
    t0 = time.time()

    ########################## 1st level of segmentation ########################################
    # print(" Level 1 segmentation")

    # generating a mask using 8-connected component method on threshold image
    mask = get_8connected_v2(thresh, mcount=5)
    # display_mask("Initial mask",mask)
    # print("Mask Generation ",time.time()-t0)
    t0 = time.time()

    # Calcutaing the grain segment using mask image
    s = cal_segment_area(mask)
    # print("Calculating segment ends",time.time()-t0)
    t0 = time.time()

    # cv2.waitKey()
    # removing the backgraound of grain
    # timg = np.array([[[0,0,0] if mask[i,j] == 0 else org[i,j] for j in range(w)] for i in range(h)], dtype=np.uint8)

    # removing very small particals (smaller the 2^3 the average size)
    low_Tarea, up_Tarea = areaThreshold_by_havg(s, 3)
    slist = list(s)
    s1count = total = 0
    total += len(slist)
    for i in slist:
        area = (s[i][0] - s[i][1]) * (s[i][2] - s[i][3])
        if area < low_Tarea:# or area > up_Tarea:
            rm = s.pop(i)
            s1count += 1
            # if dlog == 1: rm_detail.write(str(rm)+'\n')
            # cv2.imwrite('/media/zero/41FF48D81730BD9B/kisannetwork/removed/'+img_file.split('/')[-1].split(['.'])[0]+'_l1_'+str(s1count), get_img_value_inRange(org, mask, i, s[i]))
    print("Level 1 segmentation Finished:")
    print("\tRejected segment: %d" % (s1count))

    if dlog == 1: rm_detail.write("\n\t%d Number of segment rejected out of %d in L1 segmentation\n"%(s1count, total))
    # print(" Level 1 segmentation Finished",time.time()-t0)
    t0 = time.time()
    ####################### 1st level of segmentation Finished ##################################

    ####################### 2nd level of segmentation ###########################################
    # print("\t Level 2 segmentation")
    # print(s)
    new_s = {}

    s_range = [i for i in s]

    max_index = max(s_range)

    segments = {}
    s2count = extra = 0
    # print("Level 2 seg. start", time.time() - t0)
    t0 = time.time()
    for sindex in s_range:
        s1 = {}
        org1 = get_img_value_inRange(org, mask, sindex, s[sindex])
        iimg = get_img_value_inRange(img, mask, sindex, s[sindex])
        # cv2.imshow('image',iimg)
        if len(iimg) == 0:
            continue

        if isMoregrain(iimg, T):
            a = L2_segmentation_2(iimg, T=T, index=max_index + 5 + len(new_s))
        else:
            segments[sindex] = org1
            continue
        if not a:
            segments[sindex] = org1
            extra += 1
            continue
        masks, trm = a
        s2count += trm
        total += len(masks) + trm -1
        for msk in masks:
            a = cal_segment_area(msk)
            s1.update(a)
            for ii in a:
                # display_mask("mask %d"%(ii), msk)
                segments[ii] = get_img_value_inRange(org1, msk, ii, s1[ii])
        # cv2.waitKey()
        ######################################## segmenting adding ########################
        m = s.pop(sindex)
        mask =remove_mask(mask, sindex, m)
        mask1 = np.sum(masks, axis=0)
        # mask[m[0]:m[1], m[2]:m[3]] = [[0 if pixel == sindex else pixel for pixel in row] for row in mask[m[0]:m[1], m[2]:m[3]]]
        mask[m[0]:m[1], m[2]:m[3]] += mask1

        for k in s1:
            area = (s1[k][0] - s1[k][1]) * (s1[k][2] - s1[k][3])
            if area > low_Tarea and area < up_Tarea:
                new_s[k] = [m[0] + s1[k][0], m[0] + s1[k][1], m[2] + s1[k][2], m[2] + s1[k][3]]
        max_index = max([max_index]+list(new_s))
        ###################################################################################

    print("\nLevel 2 segmentation Finished:")
    print("\tRejected segment: %d" % (s2count))
    # t0 = time.time()
    #####################2nd level of segmentation Finished ###################################
    print("\n\nTotal number of segments %d"%(total))
    print("Number of rejected segments %d\n\n"%(s1count+s2count))
    # print
    if dlog == 1: rm_detail.write("\tIn level 2 segmentation %d rejected\n\tTotal number of segments %d\n\tNumber of rejected segments %d\n"%(s2count,total,s1count+s2count))


    s.update(new_s)

    # marking the segments
    torg = org.copy()
    for i in s:
        imgRectangled = cv2.rectangle(torg, (s[i][2], s[i][0]), (s[i][3], s[i][1]), (0, 0, 255), 1)
        # segments[i] = get_mask_value_inRange(org, mask, i, s[i])
        # segments[i] = np.array([[org[x,y] if mask[x,y] == i else [0,0,0] for y in range(s[i][2],s[i][3])] for x in range(s[i][0],s[i][1])], dtype=np.uint8)
        # cv2.imshow("segment %d" % (count), segments[count])

    # cv2.imshow('Marked image',imgRectangled)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return segments, s, imgRectangled, mask

# if __name__ == "__main__":
#     img_files = [
#         '/media/zero/41FF48D81730BD9B/kisannetwork/IMG_20180211_131308_2.jpg',
#     ]
#     count = 0
#     for img in img_files:
#         print(img)
#         seg, s, imgRectangled, mask1 = segment_image4(img)
#         print(s)
#         print(len(seg))
#         if not seg: continue
#         iimg = cv2.imread(img, cv2.IMREAD_COLOR)
#         # for i in s:
#         #     cv2.imshow('segment_%d'%(i),seg[i])
#
#         display_mask("Final mask",mask1)
#         cv2.imshow("Final detect", imgRectangled)
#         # cv2.imwrite('../test_area/mark_new1.jpg', mask_section)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
