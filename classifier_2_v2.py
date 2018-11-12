import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


import cv2, numpy as np, random, os, pickle, keras, itertools
from mlp import train
from PCA import pca
from util import get_boundry_img_matrix, get_files

def make_sets(inputs, out, percent):
    if len(inputs) != len(out): print "Error input size not equal to output size !!!"
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    rang = range(len(inputs))
    random.shuffle(rang)
    for i in rang:
        if random.random() < percent:
            x_test.append(inputs[i])
            y_test.append(out[i])
        else:
            x_train.append(inputs[i])
            y_train.append(out[i])
    return x_train, y_train, x_test, y_test

############### constant ################
MEAN_AREA =0
PERIMETER = 1
R = 2
B = 3
G = 4
EIGEN_VALUE_1 = 5
EIGEN_VALUE_2 = 6
ECCENTRICITY = 7
features = {0:'MEAN_AREA', 1:'PERIMETER', 2:'R', 3:'B', 4:'G', 5:'EIGEN_VALUE_1', 6:'EIGEN_VALUE_2', 7:'ECCENTRICITY', 8:'NUMBER_GRAIN',}
#########################################

data_dir = './dataset5_dep_on_4/'
result_dir = './weights_results_2out/'

if __name__ == "__main__":
    ftrain = []
    ftest = []
    grain_class = {
        'grain': 0,
        'not_grain':1,
    }
    # extracting features of grains
    feat_data = result_dir + 'grain_feature.pkl'
    if os.path.isfile(feat_data):
        ftrain, y_train, ftest, y_test = pickle.load(open(feat_data, 'rb'))
    else:
        grain_particles = {
                            'damaged_grain' : data_dir + 'damaged_grain',
                            'foreign' : data_dir + 'foreign_particles',
                            'grain' : data_dir + 'grain',
                            'broken_grain' : data_dir + 'grain_broken',
                            'grain_cover' : data_dir + 'grain_covered'
                          }
        grain_partical_list = {'grain' :get_files(grain_particles['grain'])}
        grain_partical_list['not_grain'] = get_files(grain_particles['damaged_grain']) + get_files(grain_particles['foreign']) + get_files(grain_particles['broken_grain']) + get_files(grain_particles['grain_cover'])
        # impurity_list = get_files(impure)
        # grain_list = get_files(grain)
        all_partical = []
        for i in grain_partical_list: all_partical += grain_partical_list[i]
        partical_classes = []
        for i in grain_partical_list:
            a = np.zeros(len(grain_class))
            a[grain_class[i]] = 1
            partical_classes += [a for j in range(len(grain_partical_list[i]))]
        # out = [[1, 0] for i in range(len(grain_list))] + [[0, 1] for i in range(len(impurity_list))]
        x_train, y_train, x_test, y_test = make_sets(all_partical, partical_classes, 0.3)

        print(' Number of grain: ', len(grain_partical_list['grain']))
        print(' Number of not grain: ', len(grain_partical_list['not_grain']))
        print("Total of sample: ", len(all_partical))

        xgtrain = []
        xctrain = []
        for g in x_train:
            img = cv2.imread(g, cv2.IMREAD_COLOR)
            xctrain.append(img)
            xgtrain.append(img[:, :, 2])

        xgtest = []
        xctest = []
        for i in x_test:
            img = cv2.imread(i, cv2.IMREAD_COLOR)
            xctest.append(img)
            xgtest.append(img[:, :, 2])

        for gi in range(len(xctrain)):
            gcolor = xctrain[gi]
            ggray = xgtrain[gi]
            h, w = ggray.shape
            thresh = np.array([[255 if pixel > 0 else 0 for pixel in row] for row in ggray])
            b = np.array(get_boundry_img_matrix(thresh, bval=1), dtype=np.float32)
            perameter = np.sum(b)/(h*w)
            area = np.sum(np.sum([[1.0 for j in range(w) if ggray[i,j]] for i in range(h)]))
            mean_area = area/(h*w)
            r, b, g = np.sum([gcolor[i,j] for j in range(gcolor.shape[1]) for i in range(gcolor.shape[0])], axis=0)/(area*256)
            _,_,eigen_value = pca(ggray)
            eccentricity = eigen_value[0]/eigen_value[1]
            l = [mean_area, perameter, r,b,g,eigen_value[0],eigen_value[1], eccentricity]
            ftrain.append(np.array(l))

        for gi in range(len(xctest)):
            gcolor = xctest[gi]
            ggray = xgtest[gi]
            h, w = ggray.shape
            thresh = np.array([[255 if pixel > 0 else 0 for pixel in row] for row in ggray])
            b = np.array(get_boundry_img_matrix(thresh, bval=1), dtype=np.float32)
            perameter = np.sum(b)/(h*w)
            area = np.sum(np.sum([[1.0 for j in range(w) if ggray[i,j]] for i in range(h)]))
            mean_area = area / (h * w)
            r, b, g = np.sum([gcolor[i, j] for j in range(gcolor.shape[1]) for i in range(gcolor.shape[0])], axis=0) / (area*256)
            _, _, eigen_value = pca(ggray)
            eccentricity = eigen_value[0] / eigen_value[1]
            l = [mean_area, perameter, r, b, g, eigen_value[0], eigen_value[1], eccentricity]
            ftest.append(l)
        pickle.dump([ftrain, y_train, ftest, y_test], open(result_dir + 'grain_feature.pkl', 'wb'))

    print "Total of sample for training :", len(ftrain)
    print "Total of sample for testing :", len(ftest)

    # MLP
    fd = open(result_dir + 'Test_results.txt','a')
    # m = [MEAN_AREA, PERIMETER, R, B, G]
    # n = [[EIGEN_VALUE_1, EIGEN_VALUE_2], [ECCENTRICITY], [NUMBER_GRAIN]]
    print "Trainning linear MLP..."
    # allComb = [list(j) for i in range(1,len(n)+1) for j in itertools.combinations(n, i)]
    allComb = [[MEAN_AREA, PERIMETER, R, B, G, EIGEN_VALUE_1, EIGEN_VALUE_2, ECCENTRICITY]]
    for feat in allComb:
        # print n, np.array(ftrain)[:, n].shape
        # feat = [i for i in m]
        # for i in n: feat += i
        print 'Paremeters :', [features[i] for i in feat]," ##### Number of classes :", [i for i in grain_class]
        modleFile = result_dir + 'weights_'+''.join([str(i) for i in feat])+'.h5'
        if os.path.isfile(modleFile):
            model = keras.models.load_model(modleFile)
        else:
            model = train(np.array(ftrain)[:,feat], np.array(y_train), modelf=modleFile)
            model.save(modleFile)
        score = model.evaluate(np.array(ftest)[:,feat], np.array(y_test))
        print('MLP Test loss:', score[0])
        print('MLP Test accuracy:', score[1])

        fd.write("Featrues: "+str([features[i] for i in feat])+'\n')
        fd.write('MLP Test loss: %f\n'%(score[0]))
        fd.write('MLP Test accuracy: %f\n\n'%(score[1]))
