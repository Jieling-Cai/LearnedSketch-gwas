import numpy as np
import time
import os
from tensorflow.keras import models, callbacks,Input,initializers
from tensorflow.keras.layers import Embedding,Concatenate,Flatten,SimpleRNN,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_data_label_list(data_arr, label_dict_path):
    label_dict = np.load(label_dict_path,allow_pickle=True).item()
    label_list = []
    for ID in data_arr[:,3]:
        key = 'rs' + str(int(ID))
        label_list.append(abs(label_dict[key]))
    print('label_list:', 'max:', np.max(label_list), 'min:', np.min(label_list), 'mean:',
          np.mean(label_list), 'median:', np.median(label_list))
    return np.asarray(data_arr),np.asarray(label_list)

def preprocess_pos(pos_arr):
    pos_all = []
    for pos in pos_arr:
        pos = str(pos)
        encode_pos = []
        for i in pos:
            if i == '0':
                encode_pos.append(10)
            else:
                encode_pos.append(int(i))
        pos_all.append(encode_pos)
    pos_all = pad_sequences(pos_all, maxlen=9, dtype='float', padding='post', value=0.0)
    return np.asarray(pos_all)

def shuffle(data, labels):
    index = np.arange(len(labels))
    np.random.shuffle(index)
    print('index:', index)
    return data[index,:], labels[index]

def run_NN2(train_data, train_labels, val_data, val_labels, checkpoint_path, loss_err_save_path, num_epoch, batch_size_number):
    input1 = Input(shape=(9,),name='input1')
    input2 = Input(shape=(1,),name='input2')
    input3 = Input(shape=(1,),name='input3')
    input4 = Input(shape=(9,),name='input4')
    # x = Dense(32,activation='relu')(input1)
    # x = Dense(32,activation='relu')(x)
    x = Embedding(11, 1, input_length=9, mask_zero=True)(input1)
    x = SimpleRNN(9)(x)
    y =  Embedding(5, 1, input_length=1)(input2)
    y = SimpleRNN(9)(y)
    z =  Embedding(5, 1, input_length=1)(input3)
    z = SimpleRNN(9)(z)
    k = Embedding(11, 1, input_length=9, mask_zero=True)(input4)
    k = SimpleRNN(9)(k)

    # y = Dense(32,activation='relu')(input2)
    # y = Dense(32,activation='relu')(y)
    # z = Dense(32,activation='relu')(input3)
    # z = Dense(32,activation='relu')(z)
    # k =  Dense(32,activation='relu')(input4)
    # k = Dense(32,activation='relu')(k)
    # y =  Dense(64,kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=2),activation='relu')(input2)
    # y = Dense(32,kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=2),activation='relu')(y)
    # z =  Dense(64,kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=2),activation='relu')(input3)
    # z = Dense(32,kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=2),activation='relu')(z)

    a = Concatenate()([x, y, z, k])
    # a = Flatten()(a)
    a = Dense(64,activation='relu')(a)
    a = Dense(32, activation='relu')(a)
    a = Dense(4,activation='relu')(a)
    output = Dense(1, name='output')(a)

    model = Model(inputs=[input1,input2,input3,input4], outputs=output)
    model.summary()
    model.compile(optimizer=Adam(lr=0.05), loss='mean_squared_error', metrics=['mae'])
    save_checkpoint_and_plot_error(model, checkpoint_path, train_data, train_labels, val_data, val_labels, num_epoch,
                                   batch_size_number, loss_err_save_path)

def run_NN(train_data, train_labels, val_data, val_labels, checkpoint_path, loss_err_save_path, num_epoch, batch_size_number):
    input1 = Input(shape=(4,),name='input1')
    x = Dense(128, activation='relu')(input1)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1, name='output')(x)

    model = Model(inputs=input1, outputs=output)
    model.summary()
    model.compile(optimizer=Adam(lr=0.05), loss='mean_squared_error', metrics=['mae'])
    save_checkpoint_and_plot_error(model, checkpoint_path, train_data, train_labels, val_data, val_labels, num_epoch,
                                   batch_size_number, loss_err_save_path)

def normalization(train_data, val_data, test_data):
    train_data = train_data.astype(float)
    val_data = val_data.astype(float)
    test_data = test_data.astype(float)
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data -= mean
    train_data /= std
    val_data -= mean
    val_data /= std
    test_data -= mean
    test_data /= std
    print('mean',mean)
    print('std',std)
    return train_data, val_data, test_data

def save_checkpoint_and_plot_error(model, checkpoint_path, train_data, train_labels, validation_data, validation_labels, num_epoch,
                                   batch_size_number, loss_err_save_path):
    checkpoint = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss',verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    history = model.fit(x=train_data, y=train_labels, validation_data=(validation_data, validation_labels), epochs=num_epoch,
                        batch_size=batch_size_number, shuffle=True, callbacks=[checkpoint])
    plt.plot(history.epoch, history.history['mae'],label='training mean absolute error')
    plt.plot(history.epoch, history.history['val_mae'],label='validation mean absolute error')
    plt.plot(history.epoch, history.history['loss'],label='training mean squared error')
    plt.plot(history.epoch, history.history['val_loss'],label='validation mean squared error')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.savefig(loss_err_save_path)
    plt.clf()

def reload_model_and_continue_training(checkpoint_path, train_data, train_labels, validation_data, validation_labels, num_epoch,
                                       batch_size_number,loss_err_save_path, initial_epoch):
    model = models.load_model(checkpoint_path)
    # train_pos_arr = preprocess_pos(train_data[:,0])
    # val_pos_arr = preprocess_pos(validation_data[:,0])
    # train_ID_arr = preprocess_pos(train_data[:,0])
    # val_ID_arr = preprocess_pos(validation_data[:,0])
    checkpoint = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss',verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    history = model.fit(x=train_data, y=train_labels,
                        validation_data=(validation_data,validation_labels),
                        epochs=num_epoch, batch_size=batch_size_number, shuffle=True, callbacks=[checkpoint],initial_epoch=initial_epoch)
    # history = model.fit(x=[train_pos_arr,train_data[:,1],train_data[:,2],train_ID_arr], y=train_labels,
    #                     validation_data=([val_pos_arr,validation_data[:,1],validation_data[:,2],val_ID_arr],validation_labels),
    #                     epochs=num_epoch, batch_size=batch_size_number, shuffle=True, callbacks=[checkpoint],initial_epoch=initial_epoch)
    plt.plot(history.epoch, history.history['mae'],label='training mean absolute error')
    plt.plot(history.epoch, history.history['val_mae'],label='validation mean absolute error')
    plt.plot(history.epoch, history.history['loss'],label='training mean squared error')
    plt.plot(history.epoch, history.history['val_loss'],label='validation mean squared error')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.savefig(loss_err_save_path)
    plt.clf()

def evaluate(checkpoint_path, data, labels, batch_size_number):
    model = load_model(checkpoint_path)
    # evaluate
    # mqe, mae = model.evaluate(data, labels, batch_size=batch_size_number,verbose=0)
    mqe, mae = model.evaluate(data, labels, batch_size=batch_size_number,verbose=0)
    print('mean squared error:', mqe, 'mean absolute error:', mae)

def predict(checkpoint_path, test_data, predict_results_save_path, test_data_aft_norm):
    model = load_model(checkpoint_path)
    # test_pos_arr = preprocess_pos(test_data_aft_norm[:,0])
    # test_ID_arr = preprocess_pos(test_data_aft_norm[:,3])
    predict_results = model.predict(test_data_aft_norm)
    # predict_results = model.predict(test_data_aft_norm)
    print('Predictions:', 'max:', np.max(predict_results), 'min:', np.min(predict_results), 'mean:', np.mean(predict_results), 'median:',
          np.median(predict_results))
    print('Start saving results:')
    # save results as dict
    IDs = []
    for i in test_data[:,3]:
        IDs.append('rs' + str(int(i)))
    test_ID_counts = dict(zip(np.asarray(IDs), predict_results))
    np.save(predict_results_save_path, test_ID_counts)
    print('Finish saving results.')

if __name__ == '__main__':
    num_epoch = 2000
    batch_size_number = 50000
    learn_from_scratch = 0
    initial_epoch = 999

    # data path
    train_data_path = r'\output\train_data.npy'
    train_labels_path = r'\output\train_Fi.npy'
    val_data_path = r'\output\val_data.npy'
    val_labels_path = r'\output\val_Fi.npy'
    test_data_path = r'\output\test_data.npy'
    test_labels_path = r'\output\test_Fi.npy'

    # save path
    val_save_name = 'val_predict_results.npy'
    test_save_name = 'test_predict_results.npy'
    checkpoint_path = r'\NN\model.h5'
    loss_err_save_path = r'\NN\loss_error.png'
    predict_results_save_path = '/NN/'

    # load data
    train_data, train_labels = get_data_label_list(np.load(train_data_path), train_labels_path)
    val_data, val_labels = get_data_label_list(np.load(val_data_path), val_labels_path)
    test_data, test_labels = get_data_label_list(np.load(test_data_path), test_labels_path)

    # shuffle the training and validation set
    train_data, train_labels = shuffle(train_data, train_labels)
    val_data, val_labels = shuffle(val_data, val_labels)
    print('training data shape:', train_data.shape, 'training labels shape:', train_labels.shape)
    print('validation data shape:', val_data.shape, 'validation labels shape:', val_labels.shape)
    print('test data shape:', test_data.shape, 'test labels shape:', test_labels.shape)

    train_data_aft_norm,val_data_aft_norm,test_data_aft_norm = normalization(train_data, val_data, test_data)

    # Train model and plot metrics
    start_time = time.time()
    print('Start training:')
    if learn_from_scratch == 1:
        print('Train from scratch:')
        run_NN(train_data_aft_norm, train_labels, val_data_aft_norm, val_labels, checkpoint_path, loss_err_save_path,
         num_epoch, batch_size_number)
    else:
        reload_model_and_continue_training(checkpoint_path, train_data_aft_norm, train_labels, val_data_aft_norm, val_labels, num_epoch,
                                           batch_size_number, loss_err_save_path, initial_epoch)
    t = time.time() - start_time
    print('Finish training - Takes %f seconds' % t)

    # evaluate on validation set and test set
    # print('validation  data:')
    # evaluate(checkpoint_path, val_data, val_labels, batch_size_number)
    # print('test data:')
    # evaluate(checkpoint_path, test_data, test_labels, batch_size_number)

    # predict
    predict(checkpoint_path, val_data, predict_results_save_path+val_save_name, val_data_aft_norm)
    predict(checkpoint_path, test_data, predict_results_save_path+test_save_name, test_data_aft_norm)

