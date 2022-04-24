import os
import sys
import time
import argparse
import numpy as np
from itertools import repeat
from multiprocessing import Pool

def get_data_label_list(counts_dict_path):
    counts_dict = np.load(counts_dict_path,allow_pickle=True).item()
    words = []
    counts = []
    for word in counts_dict:
        words.append(word)
        counts.append(counts_dict[word])
    return np.asarray(words), np.asarray(counts)

def myfunc(y, n_buckets, n_hash, name):
    start_t = time.time()
    loss = count_sketch(y, n_buckets, int(n_hash))
    print('%s: # hashes %d, # buckets %d - loss %.2f\t time: %.2f sec' % \
        (name, n_hash, n_buckets, loss, time.time() - start_t))
    return loss

def count_sketch(y, n_buckets, n_hash):
    """ Count-Sketch
    Args:
        y: true counts of each item, float - [num_items]
        n_buckets: number of buckets
        n_hash: number of hash functions

    Returns:
        Estimation error
    """
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

    loss = 0
    for i in range(len(y)):
        y_est = np.median(
            [y_signs_all[k, i] * counts_all[k, y_buckets_all[k, i]] for k in range(n_hash)])
        loss += np.abs(abs(y[i]) - abs(y_est)) * abs(y[i])
    return loss / np.sum(np.abs(y))

def random_hash_with_sign(y, n_buckets):
    """ Assign items in y into n_buckets, randomly pick a sign for each item
    Args:
        y: true counts of each item, float - [num_items]
        n_buckets: number of buckets

    Returns
        counts: estimated counts in each bucket, float - [num_buckets]
        loss: estimation error
        y_bueckets: item -> bucket mapping - [num_items]
    """
    counts = np.zeros(n_buckets)
    y_buckets = np.random.choice(np.arange(n_buckets), size=len(y))
    y_signs = np.random.choice([-1, 1], size=len(y))
    for i in range(len(y)):
        counts[y_buckets[i]] += (y[i] * y_signs[i])
    return counts, y_buckets, y_signs

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--test_data", type=str, nargs='*', help="input .npy data", default='/test_Fi.npy')
    argparser.add_argument("--n_workers", type=int, help="number of workers", default=120)
    argparser.add_argument("--seed", type=int, help="random state for sklearn", default=69)
    argparser.add_argument("--space_list", type=float, nargs='*', help="space in MB", default=[4,4.2,4.4,4.6,4.8,5,5.2,5.4,5.6,5.8,6,6.2,6.4,6.6,6.8,7])
    argparser.add_argument("--n_hashes_list", type=int, nargs='*', help="number of hashes", default=[2,3,4,5,6])
    argparser.add_argument("--save", type=str, help="prefix to save the results", default='SNPs')
    args = argparser.parse_args()
    bucket_byte = 4

    command = ' '.join(sys.argv) + '\n'
    log_str = command
    print(log_str)
    np.random.seed(args.seed)

    x, y = get_data_label_list(args.test_data)

    name = 'count_sketch'
    folder = os.path.join('param_results', name, '')
    if not os.path.exists(folder):
        os.makedirs(folder)

    nb_all = []
    nh_all = []
    for n_hash in args.n_hashes_list:
        for space in args.space_list:
            n_buckets = int(space * 1e6 / (n_hash * bucket_byte))
            nh_all.append(n_hash)
            nb_all.append(n_buckets)
    rshape = (len(args.n_hashes_list), len(args.space_list))

    start_t = time.time()
    pool = Pool(args.n_workers)
    results = pool.starmap(myfunc, zip(repeat(y), nb_all, nh_all,repeat(name)))
    pool.close()
    pool.join()

    results = np.reshape(results, rshape)
    nb_all = np.reshape(nb_all, rshape)
    nh_all = np.reshape(nh_all, rshape)

    log_str += '==== results ====\n'
    for i in range(len(results)):
        for j in range(len(results[i])):
            space = nh_all[i, j] * nb_all[i, j] * bucket_byte / 1e6
            log_str += '%s: # hashes %d, # buckets %d, space %.2f MB - loss %.2f\n' % \
                (name, nh_all[i, j], nb_all[i, j], space, results[i, j])
    log_str += 'total time: %.2f sec\n' % (time.time() - start_t)

    log_str += '==== best parameters ====\n'
    best_param_idx = np.argmin(results, axis=0)
    best_n_buckets = nb_all[best_param_idx, np.arange(len(nb_all[0]))]
    best_n_hashes  = nh_all[best_param_idx, np.arange(len(nb_all[0]))]
    best_loss      = results[best_param_idx, np.arange(len(nb_all[0]))]
    for i in range(len(best_loss)):
        log_str += 'space: %.2f, n_buckets %d, n_hashes %d - \tloss %.2f\n' % \
            (args.space_list[i], best_n_buckets[i], best_n_hashes[i], best_loss[i])
    log_str += 'total time: %.2f sec\n' % (time.time() - start_t)

    print(log_str)
    with open(os.path.join(folder, args.save+'.log'), 'w') as f:
        f.write(log_str)

    np.savez(os.path.join(folder, args.save),
        command=command,
        loss_all=results,
        n_hashes=nh_all,
        n_buckets=nb_all,
        space_list=args.space_list,
        )
