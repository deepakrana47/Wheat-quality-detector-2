import cv2
from util import *
import warnings, traceback
warnings.filterwarnings("error")

def get_chain_code(boundry):
    chainCode = [0]
    p1 = boundry[0]
    for p2 in boundry[1:]:
        if p2[0] - p1[0] == 0  and p2[1] - p1[1] == 1:
            chainCode.append(0)
        elif p2[0] - p1[0] == 1  and p2[1] - p1[1] == 1:
            chainCode.append(7)
        elif p2[0] - p1[0] == 1  and p2[1] - p1[1] == 0:
            chainCode.append(6)
        elif p2[0] - p1[0] == 1  and p2[1] - p1[1] == -1:
            chainCode.append(5)
        elif p2[0] - p1[0] == 0  and p2[1] - p1[1] == -1:
            chainCode.append(4)
        elif p2[0] - p1[0] == -1  and p2[1] - p1[1] == -1:
            chainCode.append(3)
        elif p2[0] - p1[0] == -1  and p2[1] - p1[1] == 0:
            chainCode.append(2)
        elif p2[0] - p1[0] == -1  and p2[1] - p1[1] == 1:
            chainCode.append(1)
        p1=p2
    return chainCode

# imgFile = '/media/zero/41FF48D81730BD9B/kisannetwork/dataset/grain/IMG_20161016_134107263_2984.jpg'
# img = cv2.imread(imgFile, cv2.IMREAD_COLOR)
# gray = padding2D_zero(img[:,:,2],2)
# data = np.where(gray < 50, 0, 255)
# h,w=gray.shape
# boundry = get_boundry_as_points(data)
# chainCode = get_chain_code(boundry)

def elliptic_fourier_descriptors(contour, order=10):
    dxy = np.diff(contour, axis=0)
    dt = np.sqrt((dxy ** 2).sum(axis=1))
    t = np.concatenate([([0., ]), np.cumsum(dt)])
    T = t[-1]
    try:
        phi = (2 * np.pi * t) / T
    except RuntimeWarning:
        traceback.print_exc()
        return None

    coeffs = np.zeros((order, 4))
    for n in range(1, order + 1):
        const = T / (2 * n * n * np.pi * np.pi)
        phi_n = phi * n
        d_cos_phi_n = np.cos(phi_n[1:]) - np.cos(phi_n[:-1])
        d_sin_phi_n = np.sin(phi_n[1:]) - np.sin(phi_n[:-1])
        a_n = const * np.sum((dxy[:, 0] / dt) * d_cos_phi_n)
        b_n = const * np.sum((dxy[:, 0] / dt) * d_sin_phi_n)
        c_n = const * np.sum((dxy[:, 1] / dt) * d_cos_phi_n)
        d_n = const * np.sum((dxy[:, 1] / dt) * d_sin_phi_n)
        coeffs[n - 1, :] = a_n, b_n, c_n, d_n
    return coeffs

def efd(coeffs,contour_1,  locus=(0., 0.)):

    N = coeffs.shape[0]
    N_half = int(np.ceil(N / 2))
    n_rows = 2
    n = len(contour_1)

    t = np.linspace(0, 1.0, n)
    xt = np.ones((n,)) * locus[0]
    yt = np.ones((n,)) * locus[1]

    for n in range(coeffs.shape[0]):
        xt += (coeffs[n, 0] * np.cos(2 * (n + 1) * np.pi * t)) + \
              (coeffs[n, 1] * np.sin(2 * (n + 1) * np.pi * t))
        yt += (coeffs[n, 2] * np.cos(2 * (n + 1) * np.pi * t)) + \
              (coeffs[n, 3] * np.sin(2 * (n + 1) * np.pi * t))
    return xt,yt

# def curvature(coeffs, contour_1):
#     N = coeffs.shape[0]
#     N_half = int(np.ceil(N / 2))
#     n_rows = 2
#     n = len(contour_1)
#
#     t = np.linspace(0, 1.0, n)
#     dxt = np.zeros((n,))
#     dyt = np.zeros((n,))
#     ddxt = np.zeros((n,))
#     ddyt = np.zeros((n,))
#
#     for n in range(coeffs.shape[0]):
#         c1 = 2 * (n + 1) * np.pi
#         c2 = (2 * (n + 1) * np.pi)**2
#         dxt += (-(coeffs[n, 0] * np.sin(c1* t)) + (coeffs[n, 1] * np.cos(c1 * t)))*c1
#         dyt += (-(coeffs[n, 2] * np.sin(c1 * t)) + (coeffs[n, 3] * np.cos(c1 * t)))*c1
#         ddxt += (-(coeffs[n, 0] * np.cos(c1 * t)) - (coeffs[n, 1] * np.sin(c1 * t)))*c2
#         ddyt += (-(coeffs[n, 2] * np.cos(c1 * t)) - (coeffs[n, 3] * np.sin(c1 * t)))*c2
#     k = (dxt*ddyt - dyt*ddxt)/(dxt**2 + dyt**2)**(3.0/2.0)
#     return k


# coff = elliptic_fourier_descriptors(boundry, 50)
# x,y = np.int0(efd(coff, contour_1=boundry, locus=np.mean(boundry, axis=0)+2))
# k = curvature(coff, boundry)
# ctop = []
# s = 6
# llen = len(boundry)
#
# for i in range(llen):
#     t1 = np.max(k[i-s:i]) if i-s >= 0 else np.max(k[:i].tolist()+k[-(s-i):].tolist())
#     t2 = np.max(k[i:i+s]) if i+s <= llen else np.max(k[i:].tolist()+k[:llen-i].tolist())
#     # ctop.append((t1*t2)**2 * (np.sum(k[i:i+s]) if i+s < llen else np.sum(k[i:llen])+np.sum(k[:llen-i+s])))
#
# for i in range(len(boundry)):
#     print (x[i], y[i]), boundry[i], k[i]
#
# img1 = np.zeros((h,w,3), dtype=np.uint8)
# img2 = np.zeros((h,w,3), dtype=np.uint8)
#
# print h,w
# dline = []
# for i in range(len(boundry)):
#     if abs(k[i]) < 0.2:
#         img1[boundry[i][0],boundry[i][1]] = [255,255,255]
#         img2[x[i], y[i]] = [255,255,255]
#     else:
#         # print (x[i], y[i]), boundry[i], k[i], ctop[i]
#         img1[boundry[i][0],boundry[i][1]] = [0,0,255]
#         img2[x[i], y[i]] = [0, 0, 255]
#         dline.append([x[i], y[i]])
# dline = np.array(dline)
# ddline=[]
# temp = [dline[-1]]
# for i in reversed(range(len(dline)-1)):
#     if abs(dline[i][0] -dline[i+1][0]) <= 1 and abs(dline[i][1] -dline[i+1][1]) <= 1:
#         temp.append(dline[i])
#     else:
#         if temp:
#             ddline.append(temp[int(np.ceil(float(len(temp))/2))])
#             temp=[]
#         temp.append(dline[i])
# if temp:
#     ddline.append(temp[int(np.ceil(float(len(temp))/2))])
#
# ddline=np.array(ddline,dtype=np.float)
# ddist = np.dot(ddline, ddline.T)
# ls = []
# np.fill_diagonal(ddist, np.inf)
#
# # print ddist
# for i in range(len(ddline)/2):
#     dmin = np.unravel_index(np.argmin(ddist), ddist.shape)
#     ls.append(dmin)
#     ddist[dmin[0],:] = ddist[:,dmin[1]] = ddist[:,dmin[0]] = ddist[dmin[1],:]= np.inf
#
# ddline = np.int0(ddline)
# for i in ls:
#     img2 = cv2.line(img2, (ddline[i[0]][1], ddline[i[0]][0]), (ddline[i[1]][1], ddline[i[1]][0]), [255,255,255], 1)
#     # img2[int(x[i])-1,int(y[i])-1] = 255
#
# cv2.imshow("boundry", img1)
# cv2.imshow("approx boundry", img2)
# # cv2.imshow("+approx boundry", img3)
# cv2.waitKey()
#
# import matplotlib.pyplot as plt
# plt.plot(k.tolist())
# plt.ylabel('some numbers')
# plt.show()