import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import tree
import pickle

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

def shuffle(data, labels):
    index = np.arange(len(labels))
    np.random.shuffle(index)
    return data[index], labels[index]

def train_Regressor(train_data_after_norm, train_labels, model_save_path):
    model = tree.DecisionTreeRegressor()
    # Fit on training data
    model.fit(train_data_after_norm, train_labels)
    with open(model_save_path,'wb') as file:
         pickle.dump(model,file)

def predict(data_aft_norm, word_list, model_save_path, predict_results_save_path, save_name):
    with open(model_save_path,'rb') as file:
         model = pickle.load(file)
    predictions = model.predict(data_aft_norm)
    print('predictions:',predictions)
    IDs = []
    for i in word_list:
        IDs.append('rs'+str(i))
    # save results as dict
    ID_Fi = dict(zip(np.array(IDs), predictions))
    np.save(os.path.join(predict_results_save_path, save_name), ID_Fi)

if __name__ == '__main__':
    val_save_name = 'val_predict_results.npy'
    test_save_name = 'test_predict_results.npy'

    # data path
    train_data_path = r'\output\train_data.npy'
    train_labels_path = r'\output\train_Fi.npy'
    val_data_path = r'\output\val_data.npy'
    val_labels_path = r'\output\val_Fi.npy'
    test_data_path = r'\output\test_data.npy'
    test_labels_path = r'\output\test_Fi.npy'

    # save path
    model_save_path = r'\tree\model.pkl'
    predict_results_save_path = r'\tree'

    # load data
    train_data, train_labels = get_data_label_list(np.load(train_data_path), train_labels_path)
    val_data, val_labels = get_data_label_list(np.load(val_data_path), val_labels_path)
    test_data, test_labels = get_data_label_list(np.load(test_data_path), test_labels_path)

    # shuffle
    train_data, train_labels = shuffle(train_data, train_labels)
    print('training data shape:', train_data.shape, 'training labels shape:', train_labels.shape)

    # train Regressor
    train_Regressor(train_data, train_labels, model_save_path)

    # Inference and save results
    predict(val_data, val_data[:,3], model_save_path, predict_results_save_path, val_save_name)
    predict(test_data, test_data[:,3], model_save_path, predict_results_save_path, test_save_name)


