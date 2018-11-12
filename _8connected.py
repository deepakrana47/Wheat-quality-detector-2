import numpy as np, cv2

from util import *

def get_equiv(equivlence, labels, mcount):
    for i in reversed(sorted(list(equivlence))):
        for j in equivlence[i]:
            if j in equivlence:
                equivlence[i] = list(set(equivlence[i] + equivlence[j]))
    a = sorted(list(equivlence))
    for i in reversed(range(len(a))):
        for j in equivlence[a[i]]:
            if j in equivlence:
                equivlence.pop(j)
            if j in labels:
                labels.pop(labels.index(j))

    out_labels = {}
    for i in labels:
        out_labels[i] = mcount
        mcount+=1

    for i in equivlence:
        for j in equivlence[i]:
            out_labels[j] = out_labels[i]
    return out_labels

##################### Image segementing by 8 connected ############
def get_8connected_v2(thresh, mcount=5):
    h,w=thresh.shape
    image_label = np.zeros((h,w), dtype=np.int)
    label = 1
    equivlence = {}
    kernal = np.array([
                        [1,1,1],
                        [1,1,1],
                        [1,1,1]
                      ], dtype=np.uint8)

    image_label = padding2D_zero(image_label,1)
    thresh = padding2D_zero(thresh,1)
    out_labels = []
    for i in range(1,h+1):
        for j in range(1,w+1):
            if thresh[i,j] == 255 and image_label[i,j] == 0:
                labels = set(image_label[i-1:i+2,j-1:j+2].reshape(3*3).tolist())
                labels = sorted(labels)
                labels.pop(0) if labels[0] == 0 else None
                if labels:
                    val = labels[0]
                else:
                    label += 1
                    val = label
                    out_labels.append(label)
                image_label[i-1:i+2,j-1:j+2] = (thresh[i-1:i+2,j-1:j+2]/255)*val
                if len(labels) > 1:
                    if labels[0] in equivlence:
                        equivlence[labels[0]] = list(set(equivlence[labels[0]] +labels[1:]))
                    else:
                        equivlence[labels[0]] = labels[1:]
    image_label = remove_padding2D_zero(image_label,1)
    # print equivlence
    # print out_labels
    # display_mask('image label',image_label)
    # raw_input()
    seg = get_equiv(equivlence, out_labels, mcount)

    for i in range(h):
        for j in range(w):
            if image_label[i, j]:
                image_label[i, j] = seg[image_label[i, j]]
    # print seg
    return image_label
########################################################################################

if __name__ == "__main__":
    # thresh = np.array(
    #     [
    #         [0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [255, 255, 255, 255, 0, 0, 255, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 255, 0, 255],
    #         [255, 255, 255, 255, 255, 0, 255, 255, 255, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 255, 0, 0],
    #         [255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 0],
    #         [0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 255, 255],
    #         [0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 255, 0],
    #         [0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 255, 255],
    #         [0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [255, 255, 255, 255, 0, 0, 255, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0],
    #         [255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0],
    #         [255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0],
    #         [0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0],
    #         [0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 255, 255],
    #         [0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 255, 255],
    #         [0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255],
    #     ]
    #     , dtype=np.uint8
    # )
    thresh = cv2.imread('test.jpg', cv2.IMREAD_COLOR)[:,:,2]
    h,w=thresh.shape
    thresh = np.array([[0 if thresh[i, j] < 75 else 255 for j in range(w)] for i in range(h)], dtype=np.uint8)
    i=get_8connected_v2(thresh, mcount=5)

    display_mask('test',i, sname='mark.jpg')
    cv2.waitKey(0)