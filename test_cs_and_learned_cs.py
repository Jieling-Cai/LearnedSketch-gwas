import numpy as np
from itertools import repeat
from multiprocessing import Pool
import argparse

def get_ID_label_arrays(label_dict):
    IDs = []
    labels = []
    for key in label_dict:
        IDs.append(key)
        labels.append(label_dict[key])
    return np.asarray(IDs),np.asarray(labels)

def learned_countsketch(predict_dict, bcut, label_dict,n_hashes,space_for_buckets,normal_bucket_bytes,space,name,save_path):
    unique_IDs = []
    y_unique = []
    cs_IDs = []
    y_cs = []

    predict_dict = get_n_top_IDs_dict(bcut, predict_dict)

    for ID in label_dict:
        if ID in predict_dict:
            unique_IDs.append(ID)
            y_unique.append(abs(label_dict[ID]))
        else:
            try:
                y_cs.append(label_dict[ID])
                cs_IDs.append(ID)
            except:
                pass

    n_cs_buckets = int((space_for_buckets - len(y_unique)*normal_bucket_bytes)/(n_hashes * normal_bucket_bytes))
    assert n_cs_buckets >0, 'number of count sketch buckets cannot be 0 or negative'

    y_est_cs = count_sketch_l(y_cs,n_cs_buckets,n_hashes)

    IDs_all = unique_IDs + cs_IDs
    y_est_all = y_unique + y_est_cs

    learned_sketch_dict = dict(zip(IDs_all,y_est_all))

    y_est_all = np.array(y_est_all)

    if name == 'learned':
        print('learned_cs: max:', np.max(y_est_all), 'mean:', np.mean(y_est_all), 'median:',
              np.median(y_est_all), 'min:', np.min(y_est_all))
        print('finish one learned sketch')
        np.save(save_path + 'learned_cs_result_%.2fMB.npy'%space,learned_sketch_dict)
    elif name == 'learned_ideal':
        print('ideal learned_cs: max:', np.max(y_est_all), 'mean:', np.mean(y_est_all), 'median:',
              np.median(y_est_all), 'min:', np.min(y_est_all))
        print('finish one ideal learned sketch')
        np.save(save_path + 'ideal_cs_result_%.2fMB.npy' %space, learned_sketch_dict)

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

def lookup_countsketch(best_bcut, train_Fi_dict, ID_byte, label_dict,n_hashes,space,normal_bucket_bytes,save_path):
    unique_IDs = []
    y_unique = []
    cs_IDs = []
    y_cs = []

    top_SNPs_in_train = get_n_top_IDs_dict(best_bcut, train_Fi_dict)

    for ID in label_dict:
        if ID in top_SNPs_in_train:
            unique_IDs.append(ID)
            y_unique.append(abs(label_dict[ID]))
        else:
            cs_IDs.append(ID)
            y_cs.append(label_dict[ID])

    n_cs_buckets = int((space*1e6 - len(y_unique)*normal_bucket_bytes - best_bcut*ID_byte)/(n_hashes * normal_bucket_bytes))
    assert n_cs_buckets >0, 'number of count sketch buckets cannot be 0 or negative'

    y_est_cs = count_sketch_l(y_cs,n_cs_buckets,n_hashes)

    IDs_all = unique_IDs + cs_IDs
    y_est_all = y_unique + y_est_cs

    lookup_sketch_dict = dict(zip(IDs_all,y_est_all))

    y_est_all = np.array(y_est_all)

    print('Table_lookup: max:', np.max(y_est_all), 'mean:', np.mean(y_est_all), 'median:',
          np.median(y_est_all), 'min:', np.min(y_est_all))
    print('finish one lookup')
    np.save(save_path + 'lookup_result_%.2fMB.npy'%space,lookup_sketch_dict)

def count_sketch_l(y, n_buckets, n_hash):
    if len(y) == 0:
        return 0    # avoid division of 0

    counts_all = np.zeros((n_hash, n_buckets))
    y_buckets_all = np.zeros((n_hash, len(y)), dtype=int)
    y_signs_all = np.zeros((n_hash, len(y)), dtype=int)
    for i in range(n_hash):
        counts, y_buckets, y_signs = random_hash_with_sign(y, n_buckets)
        counts_all[i] = counts
        y_buckets_all[i] = y_buckets
        y_signs_all[i] = y_signs

    y_est_all = []
    for i in range(len(y)):
        y_est = np.median(
            [y_signs_all[k, i] * counts_all[k, y_buckets_all[k, i]] for k in range(n_hash)])
        y_est_all.append(abs(y_est))

    return y_est_all

def count_sketch(IDs,space,y,n_buckets,n_hash,save_path):
    if len(y) == 0:
        return 0  # avoid division of 0

    counts_all = np.zeros((n_hash, n_buckets))
    y_buckets_all = np.zeros((n_hash, len(y)), dtype=int)
    y_signs_all = np.zeros((n_hash, len(y)), dtype=int)
    for i in range(n_hash):
        counts, y_buckets, y_signs = random_hash_with_sign(y, n_buckets)
        counts_all[i] = counts
        y_buckets_all[i] = y_buckets
        y_signs_all[i] = y_signs

    y_est_all = []
    for i in range(len(y)):
        y_est = np.median(
            [y_signs_all[k, i] * counts_all[k, y_buckets_all[k, i]] for k in range(n_hash)])
        y_est_all.append(abs(y_est))

    print('cs: max:', np.max(y_est_all), 'mean:', np.mean(y_est_all), 'median:',
          np.median(y_est_all), 'min:', np.min(y_est_all))
    print('finish one sketch')
    np.save(save_path + 'cs_result_%.2fMB.npy' %space, dict(zip(IDs,y_est_all)))

def random_hash_with_sign(y, n_buckets):
    counts = np.zeros(n_buckets)
    y_buckets = np.random.choice(np.arange(n_buckets), size=len(y))
    y_signs = np.random.choice([-1, 1], size=len(y))
    for i in range(len(y)):
        counts[y_buckets[i]] += (y[i] * y_signs[i])
    return counts, y_buckets, y_signs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-train_Fi","--train_Fi_dict_path", type=str, help="Path of training set's ID-Fi/CA Fi values dictionary", default=r'/output/train_Fi.npy')
    parser.add_argument("-test_Fi","--label_dict_path", type=str, help="Path of test set's ID-Fi/CA Fi values dictionary", default='/output/test_Fi.npy')
    parser.add_argument("-predict","--predict_dict_path", type=str, help="Path of predicted results obtained from 'NN.py'", default='/NN/test_predict_results.npy')
    parser.add_argument("-o","--save_path", type=str, help="The prefix of path to save all the hashing results (lookup table & count sketch & learned sketch & ideal learned sketch)", default='/NN/cs_results/')
    parser.add_argument("-model_size","--model_size", type=int, help="B, indicate the model(oracle) size", default=0)
    parser.add_argument("-byte","--normal_bucket_bytes", type=int, help="B, indicate the space each normal bucket occupies", default=4)
    parser.add_argument("-n_workers","--n_workers", type=int, help="the number of parallel threads", default=3)
    args = parser.parse_args()

    ID_byte = 4
    space_list = [4,5,6]
    lookup_best_hash = [2,2,2]
    lookup_best_bcut = [314948,401392,529261]
    learned_best_hash = [2,2,3]
    learned_best_bcut = [700000,875000,900000]
    ideal_best_hash = [2,2,2]
    ideal_best_bcut = [800000,1000000,1200000]
    cs_best_hash = [3,3,3]

    # load labels and predicted results
    label_dict = np.load(args.label_dict_path,allow_pickle=True).item()
    predict_dict = np.load(args.predict_dict_path,allow_pickle=True).item()
    train_Fi_dict = np.load(args.train_Fi_dict_path, allow_pickle=True).item()

    # test using lookup table
    pool = Pool(args.n_workers)
    pool.starmap(lookup_countsketch,zip(lookup_best_bcut,repeat(train_Fi_dict),repeat(ID_byte),repeat(label_dict),lookup_best_hash,space_list,repeat(args.normal_bucket_bytes),repeat(args.save_path)))
    pool.close()
    pool.join()

    # test using learned-cs
    name = 'learned'
    pool = Pool(args.n_workers)
    pool.starmap(learned_countsketch,zip(repeat(predict_dict),learned_best_bcut,repeat(label_dict),learned_best_hash,np.asarray(space_list) * 1e6 - args.model_size,repeat(args.normal_bucket_bytes),space_list,repeat(name),repeat(args.save_path)))
    pool.close()
    pool.join()

    # test using count-sketch
    IDs,labels = get_ID_label_arrays(label_dict)
    cs_buckets_list = np.array((np.asarray(space_list) * 1e6/(np.asarray(cs_best_hash)*args.normal_bucket_bytes)),dtype = int)
    pool = Pool(args.n_workers)
    pool.starmap(count_sketch,zip(repeat(IDs),space_list,repeat(labels),cs_buckets_list,cs_best_hash,repeat(args.save_path)))
    pool.close()
    pool.join()

    # test using learned-ideal
    name = 'learned_ideal'
    pool = Pool(args.n_workers)
    pool.starmap(learned_countsketch,zip(repeat(label_dict),ideal_best_bcut, repeat(label_dict),ideal_best_hash,np.asarray(space_list) * 1e6,repeat(args.normal_bucket_bytes),space_list,repeat(name),repeat(args.save_path)))
    pool.close()
    pool.join()


