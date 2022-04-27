import numpy as np
import os
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

def get_extracted_feats(model_path, feats):
    from tensorflow.keras.models import load_model
    from tensorflow.keras.models import Model
    model = load_model(model_path)
    layer_model = Model(inputs=model.input, outputs=model.get_layer('dense_5').output)
    # feat_pos_arr = preprocess_pos(feats[:,0])
    # feat_ID_arr = preprocess_pos(feats[:,3])
    extracted_feats = np.asarray(layer_model.predict(feats))
    print('extracted features\' shape:',extracted_feats.shape)
    return extracted_feats

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
    for id in word_list:
        IDs.append('rs'+str(id))
    # save results as dict
    ID_Fi = dict(zip(np.asarray(IDs), np.asarray(predictions)))
    np.save(os.path.join(predict_results_save_path, save_name), ID_Fi)

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

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--val_save_name", type=str, default='val_predict_results.npy')
    argparser.add_argument("--test_save_name", type=str, default='test_predict_results.npy')
    argparser.add_argument("--train_data_path", type=str, default=r'\output\train_data.npy')
    argparser.add_argument("--train_labels_path", type=str,  default=r'\output\train_Fi.npy')
    argparser.add_argument("--val_data_path", type=str,  default=r'\output\val_data.npy')
    argparser.add_argument("--val_labels_path", type=str,  default=r'\output\val_Fi.npy')
    argparser.add_argument("--test_data_path", type=str,  default=r'\output\test_data.npy')
    argparser.add_argument("--test_labels_path", type=str,  default=r'\output\test_Fi.npy')
    argparser.add_argument("--NN_model_path", type=str,  default=r'\NN\model.h5')
    argparser.add_argument("--model_save_path", type=str,  default=r'\NN\model.pkl')
    argparser.add_argument("--predict_results_save_path", type=str,  default=r'\NN')    
    args = argparser.parse_args()

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

    # get extracted feats
    train_extracted = get_extracted_feats(NN_model_path, train_data_aft_norm)
    val_extracted = get_extracted_feats(NN_model_path, val_data_aft_norm)
    test_extracted = get_extracted_feats(NN_model_path, test_data_aft_norm)

    # train Regressor
    train_Regressor(train_extracted, train_labels, model_save_path)

    # Inference and save results
    predict(test_extracted, test_data[:,3], model_save_path, predict_results_save_path, test_save_name)
    predict(val_extracted, val_data[:,3], model_save_path, predict_results_save_path, val_save_name)



