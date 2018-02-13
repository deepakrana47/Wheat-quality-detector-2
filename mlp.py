import keras, os
from keras.models import Sequential
from keras.layers import Dense,Dropout

def make_model(input_shape, out_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_shape))
    # model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(out_shape, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(lr=0.001), metrics=['accuracy'])
    return model

def classify(model, inp):
    return model.pridect(inp)

def evaluate(model, x, y):
    model.evaluate(x=x, y=y, batch_size=50, verbose=1)
    return

def train(x, y, modelf=None):
    if os.path.isfile(modelf):
        model = keras.models.load_model(modelf)
    else:
        batch_size = 100
        epochs = 2000
        model = make_model(x[0].shape[0], y[0].shape[0])
        # model.summary()
        # if raw_input("y to continue: ") == 'y': pass
        # else: exit()
        model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=1)
    return model

