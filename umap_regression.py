
import numpy as np
import matplotlib.pyplot as plt
import umap

def get_data_label_list(data_arr_path, label_dict_path):
    data_arr = np.load(data_arr_path)
    label_dict = np.load(label_dict_path,allow_pickle=True).item()
    label_list = []
    for ID in data_arr[:,3]:
        key = 'rs' + str(int(ID))
        label_list.append(abs(label_dict[key]))
    print('label_list:', 'max:', np.max(label_list), 'min:', np.min(label_list), 'mean:',
          np.mean(label_list), 'median:', np.median(label_list))
    return np.asarray(data_arr),np.asarray(label_list)

def get_n_top_IDs_dict(n,ID_dict):
    # Take the absolute values for computing the ranking indexes
    ID_arr = []
    value_arr = []
    for ID in ID_dict:
        ID_arr.append(ID)
        value_arr.append(abs(ID_dict[ID]))
    sort_index = sorted(range(len(value_arr)), key=lambda k: value_arr[k], reverse=True)
    del value_arr
    ID_arr = np.asarray(ID_arr)[sort_index]  # rank IDs
    n_top_IDs_dict = dict([(key, ID_dict[key]) for key in ID_arr[0:n-1]])
    return n_top_IDs_dict

def normalization(train_data_path, test_data):
    train_data = np.load(train_data_path)
    train_data = train_data.astype(float)
    test_data = test_data.astype(float)
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    test_data -= mean
    test_data /= std
    print('mean',mean)
    print('std',std)
    return test_data

if __name__ == '__main__':
    train_data_path = '/output/train_data.npy'
    test_data_path = '/output/test_data.npy'
    test_labels_path = '/output/test_chi_square.npy'
    model_path = '/NN/model.h5'
    save_fig_path = '/NN/umap.png'

    test_data, test_labels = get_data_label_list(test_data_path, test_labels_path)
    sort_index = sorted(range(len(test_labels)), key=lambda k: test_labels[k], reverse=True)
    test_data = test_data[sort_index,:]
    test_labels = test_labels[sort_index]

    test_data = normalization(train_data_path, test_data)

    len_test = len(test_labels)

    sig_index = np.random.choice(np.arange(40000), 5000, replace=False)
    insig_index = np.random.choice(np.arange(len_test)[-40000:], 5000, replace=False)
    # insig_index = np.random.choice(np.arange(len_test-int(len_test*0.01))+int(len_test*0.01), 2000, replace=False)

    # test_data = np.concatenate((test_data[sig_index,:],test_data[insig_index,:]),axis=0)
    # test_labels = np.concatenate((test_labels[sig_index],test_labels[insig_index]),axis=None)
    # test_words = np.concatenate((test_words[sig_index],test_words[insig_index]),axis=None)
    sig_data = test_data[sig_index,:]
    insig_data = test_data[insig_index,:]
    sig_labels = test_labels[sig_index]
    insig_labels = test_labels[insig_index]

    # data = np.concatenate((sig_data,insig_data),axis=0)
    # labels = np.concatenate((sig_labels,insig_labels),axis=None)

    from tensorflow.keras.models import load_model
    from tensorflow.keras.models import Model
    model = load_model(model_path)
    layer_model = Model(inputs=model.input, outputs=model.get_layer('dense_5').output)
    # feats = np.asarray(layer_model.predict(data))
    # embedd = umap.UMAP(n_neighbors=3).fit_transform(feats)
    print(sig_data.shape)
    print(insig_data.shape)

    sig_feats = np.asarray(layer_model.predict(sig_data))
    insig_feats = np.asarray(layer_model.predict(insig_data))

    sig_embedd = umap.UMAP(n_neighbors=100).fit_transform(sig_feats)
    insig_embedd = umap.UMAP(n_neighbors=100).fit_transform(insig_feats)

    # plt.scatter(sig_embedd[:, 0], sig_embedd[:, 1], s=10, c=test_labels, cmap='viridis')
    # plt.colorbar()
    plt.scatter(sig_embedd[:, 0], sig_embedd[:, 1], s=10, alpha=0.8, color='indigo', label='Significant SNPs')
    plt.scatter(insig_embedd[:, 0], insig_embedd[:, 1], s=10, color='darkgreen',  alpha=0.8, label='Insignificant SNPs')
    # plt.scatter(embedd[:len(sig_index), 0], embedd[:len(sig_index), 1], s=10, color='indigo', label='Frequent Words')
    # plt.scatter(embedd[-len(insig_data):, 0], embedd[-len(insig_data):, 1], s=10, color='darkgreen', label='Infrequent Words')
    plt.legend(loc='upper right')
    plt.savefig(save_fig_path)
    plt.show()

