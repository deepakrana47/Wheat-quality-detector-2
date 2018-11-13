import keras, os
from keras.models import Sequential, save_model
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten

def make_model(input_shape, out_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(out_shape, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(lr=0.001), metrics=['accuracy'])
    return model

def classify(model, inp):
    return model.pridect(inp)

def evaluate(model, x, y):
    model.evaluate(x=x, y=y, batch_size=50, verbose=1)
    return

def train(x, y, modelf=None):
    if len(x.shape) == 3:
        x=x.reshape(x.shape[0],x.shape[1],x.shape[2],1)
    # if os.path.isfile(modelf):
    #     model = keras.models.load_model(modelf)
    else:
        batch_size = 100
        epochs = 20
        model = make_model(x[0].shape, y[0].shape[0])
        model.summary()
        if raw_input("y to continue: ") == 'y': pass
        else: exit()
        model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=1)
        # model.save(modelf if modelf else 'weights.pkl')
    return model


import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


import cv2, numpy as np, random
# from segment_formation import get_files
# from codes.mlp import train

def get_files(indir):
    indir = indir.rstrip('/')
    flist =os.listdir(indir)
    files = []
    for f in flist:
        f = indir+'/'+f
        if os.path.isdir(f):
            tfiles = get_files(f)
            files += [tf for tf in tfiles]
        else:
            files.append(f)
    return files

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

import pickle
if __name__ == "__main__":
    # a = cv2.imread('/media/zero/41FF48D81730BD9B/kisannetwork/dataset/_number_detect/boundry/1/IMG_20161016_124056_1013.jpg', cv2.IMREAD_GRAYSCALE)
    oneGrain = 'segmentation_data/boundry_1_30.pkl'
    moreGrain = 'segmentation_data/boundry_2_30.pkl'
    oneGrain_list = pickle.load(open(oneGrain,'rb'))
    moreGrain_list = pickle.load(open(moreGrain,'rb'))
    Grain = oneGrain_list + moreGrain_list
    out = [[0,1] for i in range(len(oneGrain_list))] + [[1,0] for i in range(len(moreGrain_list))]
    x_train, y_train, x_test, y_test = make_sets(Grain, out, 0.3)


    print "Number of one grain sample:", len(oneGrain_list)
    print "Number of more grain sample:",len(moreGrain_list)
    print "Total of sample:",len(Grain)

    from util import get_boundry_img_matrix
    # extracting features of grains
    # ftrain = []
    # c=0
    # for tx in x_train:
    #     h,w = tx.shape
    #     area = np.sum(np.sum([[1.0 for j in range(w) if tx[i, j]] for i in range(h)]))/(h*w)
    #     boundry = get_boundry_img_matrix(tx, bval=255)
    #     perameter = np.sum(np.sum([[1.0 for j in range(w) if boundry[i, j]] for i in range(h)]))/(2*(h+w))
    #     l = [perameter, area]
    #     ftrain.append(np.array(l))
    #     if c < 20: print [area, perameter],
    #     c+=1
    # print
    # ftest = []
    # c=0
    # for gi in range(len(x_test)):
    #     h,w = tx.shape
    #     area = np.sum(np.sum([[1.0 for j in range(w) if tx[i, j]] for i in range(h)]))/(h*w)
    #     boundry = get_boundry_img_matrix(tx, bval=1)
    #     perameter = np.sum(np.sum([[1.0 for j in range(w) if boundry[i, j]] for i in range(h)]))/(2*(h+w))
    #     l = [perameter, area]
    #     ftest.append(l)
    #     if c < 20: print [area, perameter],
    #     c+=1
    # print

    # MLP
    print "Trainning linear mlp..."
    x_train = np.array(x_train).reshape((len(x_train), x_train[0].shape[0], x_train[0].shape[1], 1))
    x_test = np.array(x_test).reshape((len(x_test), x_test[0].shape[0], x_test[0].shape[1], 1))
    model = train(np.array(x_train),  np.array(y_train))
    score = model.evaluate(np.array(x_test), np.array(y_test))
    print('cnn Test loss:', score[0])
    print('cnn Test accuracy:', score[1])
    model_file = 'segmentation_data/weights_'+str(x_train[0].shape[0])+"_"+str(x_train[0].shape[1])+'_.h5'
    save_model(model, model_file)
    print "weight file is saved."
