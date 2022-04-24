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

def get_data_label_predict_list(counts_dict_path,results_path):
    counts_dict = np.load(counts_dict_path,allow_pickle=True).item()
    results_dict = np.load(results_path,allow_pickle=True).item()
    words = []
    counts = []
    predicts = []
    for word in counts_dict:
        predicts.append(results_dict[word])
        words.append(word)
        counts.append(counts_dict[word])
    return np.asarray(words), np.asarray(counts),np.asarray(predicts)

def run_ccm(y, b_cutoff, n_hashes, n_buckets, name):
    start_t = time.time()
    loss, space = cutoff_countsketch(y, n_buckets, b_cutoff, n_hashes)
    print('%s: bcut: %d, # hashes %d, # buckets %d - loss %.2f\t time: %.2f sec' % \
        (name, b_cutoff, n_hashes, n_buckets, loss, time.time() - start_t))
    return loss, space

def cutoff_countsketch(y, n_buckets, b_cutoff, n_hashes):
    """ Learned Count-Sketch
    Args:
        y: true counts of each item (sorted, largest first), float - [num_items]
        n_buckets: number of total buckets
        b_cutoff: number of unique buckets
        n_hash: number of hash functions

    Returns:
        loss_avg: estimation error
        space: space usage in bytes
    """
    assert b_cutoff <= n_buckets, 'bucket cutoff cannot be greater than n_buckets'
    counts = np.zeros(n_buckets)
    if len(y) == 0:
        return 0            # avoid division of 0

    y_buckets = []
    for i in range(b_cutoff):
        if i >= len(y):
            break           # more unique buckets than # flows
        counts[i] += abs(y[i])   # unique bucket for each flow
        y_buckets.append(i)

    loss_cs = count_sketch(y[b_cutoff:], n_buckets - b_cutoff, n_hashes)

    loss_avg = (loss_cs * np.sum(np.abs(y[b_cutoff:]))) / np.sum(np.abs(y))
    print('\tloss_avg %.2f' % loss_avg)

    space = b_cutoff * 4 + (n_buckets - b_cutoff) * n_hashes * 4
    return loss_avg, space

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

def run_ccm_wscore(y, scores, score_cutoff, space, n_hashes, name):
    start_t = time.time()
    len_y_ccs = len(y[scores >  score_cutoff])
    n_cs_buckets = int((space*1e6 - len_y_ccs*4)/(n_hashes*4))
    loss, space = cutoff_countsketch_wscore(y, scores, score_cutoff, n_cs_buckets, n_hashes)
    print('%s: scut: %.3f, # hashes %d, # cm buckets %d - loss %.2f\t time: %.2f sec' % \
        (name, score_cutoff, n_hashes, n_cs_buckets, loss, time.time() - start_t))
    return loss, space

def cutoff_countsketch_wscore(y, scores, score_cutoff, n_cs_buckets, n_hashes):
    if len(y) == 0:
        return 0            # avoid division of 0

    y_ccs = y[scores >  score_cutoff]
    y_cs  = y[scores <= score_cutoff]

    loss_cf = 0  # put y_ccs into cutoff buckets, no loss
    loss_cs = count_sketch(y_cs, n_cs_buckets, n_hashes)

    assert len(y_ccs) + len(y_cs) == len(y)
    loss_avg = (loss_cf * np.sum(np.abs(y_ccs)) + loss_cs * np.sum(np.abs(y_cs))) / np.sum(np.abs(y))
    print('\tloss_avg %.2f' % loss_avg)

    space = len(y_ccs) * 4 + n_cs_buckets * n_hashes * 4
    return loss_avg, space

def run_ccm_lookup(x, y, n_hashes, n_cm_buckets, d_lookup, s_cutoff, name, bucket_byte, ID_byte, bcut):
    start_t = time.time()

    loss, space = cutoff_lookup(x, y, n_cm_buckets, n_hashes, d_lookup, s_cutoff,\
            bucket_byte, ID_byte, bcut)
    print('%s: s_cut: %d, # hashes %d, # cm buckets %d - loss %.2f\t time: %.2f sec' %\
        (name, s_cutoff, n_hashes, n_cm_buckets, loss, time.time() - start_t))
    return loss, space

def cutoff_lookup(x, y, n_cm_buckets, n_hashes, d_lookup, y_cutoff, bucket_byte, ID_byte, bcut):
    """ Learned Count-Min (use predicted scores to identify heavy hitters)
    Args:
        x: feature of each item - [num_items]
        y: true counts of each item, float - [num_items]
        n_cm_buckets: number of buckets of Count-Min
        n_hashes: number of hash functions
        d_lookup: x[i] -> y[i] look up table
        y_cutoff: threshold for heavy hitters

    Returns:
        loss_avg: estimation error
        space: space usage in bytes
    """
    if len(y) == 0:
        return 0            # avoid division of 0

    y_ccm = []
    y_cm = []
    for i in range(len(y)):
        if x[i] in d_lookup:
            if d_lookup[x[i]] > y_cutoff:
                y_ccm.append(y[i])
            else:
                y_cm.append(y[i])
        else:
            y_cm.append(y[i])

    loss_cf = 0 # put y_ccm into cutoff buckets, no loss
    loss_cm = count_sketch(y_cm, n_cm_buckets, n_hashes)

    assert len(y_ccm) + len(y_cm) == len(y)
    loss_avg = (loss_cf * np.sum(np.abs(y_ccm)) + loss_cm * np.sum(np.abs(y_cm))) / np.sum(np.abs(y))
    print('\tloss_avg %.2f' % loss_avg)
    print('\t# uniq', len(y_ccm), '# cm', len(y_cm))

    space = len(y_ccm) * bucket_byte + n_cm_buckets * n_hashes * bucket_byte + bcut * ID_byte
    return loss_avg, space

def get_great_cut(b_cut, y, max_bcut):
    assert b_cut <= max_bcut
    y_sorted = np.sort(y)[::-1]
    if b_cut < len(y_sorted):
        s_cut = y_sorted[b_cut]
    else:
        s_cut = y_sorted[-1]
    # return cut at the boundary of two frequencies
    n_after_same = np.argmax((y_sorted == s_cut)[::-1]) # items after items == s_cut
    if (len(y) - n_after_same) < max_bcut:
        b_cut_new = (len(y) - n_after_same)
        if n_after_same == 0:
            s_cut = s_cut - 1   # get every thing
        else:
            s_cut = y_sorted[b_cut_new] # item right after items == s_cut
    else:
        b_cut_new = np.argmax(y_sorted == s_cut) # first item that # items == s_cut
    return b_cut_new, s_cut

def order_y_wkey_list(y, pred):
    """ Order items based on the scores in results """
    idx = np.argsort(pred)[::-1]
    assert len(idx) == len(y)
    return y[idx], pred[idx]

def count_y_ccm_items(x_valid, scut, x_train, y_train):
    x_train = np.asarray(x_train)
    x_train_hh = x_train[y_train > scut]
    y_train_hh = y_train[y_train > scut]
    lookup_dict = dict(zip(x_train_hh, y_train_hh))
    counter = 0
    for i in range(len(x_valid)):
        if x_valid[i] in lookup_dict:
            if lookup_dict[x_valid[i]] > scut:
                counter+=1
    return counter

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--test_results", type=str, nargs='*', help="testing results of a model (.npy file)", default='/test_predict_results.npy')
    argparser.add_argument("--valid_results", type=str, nargs='*', help="validation results of a model (.npy file)", default='/val_predict_results.npy')
    argparser.add_argument("--test_data", type=str, nargs='*', help="input .npy data", default='/test_Fi.npy')
    argparser.add_argument("--valid_data", type=str, nargs='*', help="input .npy data", default='/val_Fi.npy')
    argparser.add_argument("--lookup_data", type=str, nargs='*', help="input .npy data", default=0)
    argparser.add_argument("--perfect_order", type=int, default=0)
    argparser.add_argument("--n_workers", type=int, help="number of workers", default=120)
    argparser.add_argument("--seed", type=int, help="random state for sklearn", default=69)
    argparser.add_argument("--space_list", type=float, nargs='*', help="space in MB", default=[4,4.2,4.4,4.6,4.8,5,5.2,5.4,5.6,5.8,6,6.2,6.4,6.6,6.8,7])
    argparser.add_argument("--n_hashes_list", type=int, nargs='*', help="number of hashes", default=[2,3,4,5,6])
    argparser.add_argument("--save", type=str, help="prefix to save the results", default='SNPs')
    args = argparser.parse_args()

    bucket_byte = 4
    ID_byte = 4
    sketch_type = 'count_sketch'
    assert not (args.perfect_order and args.lookup_data), "use either --perfect or --lookup"

    command = ' '.join(sys.argv) + '\n'
    log_str = command
    print(log_str)
    np.random.seed(args.seed)

    if args.perfect_order:
        name = 'cutoff_%s_param_perfect' % sketch_type
    elif args.lookup_data:
        name = 'lookup_table_%s' % sketch_type
    else:
        name = 'cutoff_%s_param' % sketch_type

    folder = os.path.join('param_results', name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    start_t = time.time()
    x_valid, y_valid, pred_valid = get_data_label_predict_list(args.valid_data,args.valid_results)
    x_test, y_test, pred_test = get_data_label_predict_list(args.test_data,args.test_results)

    if args.lookup_data:
        x_train, y_train = get_data_label_list(args.lookup_data)
    print('data loading time: %.1f sec' % (time.time() - start_t))

    if args.valid_results:
        y_valid_ordered, valid_scores = order_y_wkey_list(y_valid, pred_valid)

    if args.test_results:
        y_test_ordered, test_scores = order_y_wkey_list(y_test, pred_test)

    bcut_all = []
    scut_all = []
    nh_all = []
    nb_all = []

    for space in args.space_list:
        max_bcut = space * 1e6 / bucket_byte
        b_cutoffs = np.linspace(0.1, 0.9, 9) * max_bcut
        for i,bcut in enumerate(b_cutoffs):
            for n_hash in args.n_hashes_list:
                bcut = int(bcut)

                if args.lookup_data:
                    max_bcut = space * 1e6 / (bucket_byte + ID_byte)
                    bcut = np.linspace(0.1, 0.9, 9) * max_bcut
                    bcut = int(bcut[i])
                    bcut, scut = get_great_cut(bcut, np.abs(y_train), np.floor(max_bcut))    # this has to be y_train
                else:
                    if bcut < len(y_valid):
                        scut = valid_scores[bcut]
                    else:
                        scut = valid_scores[-1]

                if args.lookup_data:
                    y_ccm_counts = count_y_ccm_items(x_valid, scut, x_train, np.abs(y_train))
                    n_cmin_buckets = int((space * 1e6 - y_ccm_counts * bucket_byte - bcut * ID_byte) / (n_hash * bucket_byte))
                else:
                    n_cmin_buckets = int((space * 1e6 - bcut * bucket_byte) / (n_hash * bucket_byte))

                nb_all.append(bcut + n_cmin_buckets)
                bcut_all.append(bcut)
                scut_all.append(scut)
                nh_all.append(n_hash)

    rshape = (len(args.space_list), len(b_cutoffs), len(args.n_hashes_list))
    n_cm_all = np.array(nb_all) - np.array(bcut_all)

    if args.lookup_data:
        min_scut = np.min(scut_all) # no need to store elements that are smaller
        x_train = np.asarray(x_train)
        x_train_hh = x_train[np.abs(y_train) > min_scut]
        y_train_hh = np.abs(y_train)[np.abs(y_train) > min_scut]
        lookup_dict = dict(zip(x_train_hh, y_train_hh))

    start_t = time.time()
    pool = Pool(args.n_workers)
    if args.perfect_order:
        y_sorted = np.sort(np.abs(y_valid))[::-1]
        results = pool.starmap(run_ccm, zip(repeat(y_sorted), bcut_all, nh_all, nb_all, repeat(name)))
    elif args.lookup_data:
        results = pool.starmap(run_ccm_lookup, zip(repeat(x_valid), repeat(y_valid), nh_all, n_cm_all, repeat(lookup_dict), scut_all, repeat(name), repeat(bucket_byte), repeat(ID_byte), bcut_all))
    else:
        results = pool.starmap(run_ccm, zip(repeat(y_valid_ordered), bcut_all, nh_all, nb_all, repeat(name)))
    pool.close()
    pool.join()
    valid_results, space_actual = zip(*results)
    valid_results = np.reshape(valid_results, rshape)
    space_actual = np.reshape(space_actual, rshape)
    bcut_all = np.reshape(bcut_all, rshape)
    scut_all = np.reshape(scut_all, rshape)
    nh_all = np.reshape(nh_all, rshape)
    nb_all = np.reshape(nb_all, rshape)

    log_str += '==== valid_results ====\n'
    for i in range(len(valid_results)):
        log_str += 'space: %.2f\n' % args.space_list[i]
        for j in range(len(valid_results[i])):
            for k in range(len(valid_results[i, j])):
                log_str += '%s: bcut: %d, # hashes %d, # buckets %d - \tloss %.2f\tspace %.1f\n' % \
                    (name, bcut_all[i,j,k], nh_all[i,j,k], nb_all[i,j,k], valid_results[i,j,k], space_actual[i,j,k])
    log_str += 'param search done -- time: %.2f sec\n' % (time.time() - start_t)

    np.savez(os.path.join(folder, args.save+'_valid'),
        command=command,
        loss_all=valid_results,
        b_cutoffs=bcut_all,
        n_hashes=nh_all,
        n_buckets=nb_all,
        space_list=args.space_list,
        space_actual=space_actual,
        )

    log_str += '==== best parameters ====\n'
    rshape = (len(args.space_list), -1)
    best_param_idx = np.argmin(valid_results.reshape(rshape), axis=1)
    best_scuts     = scut_all.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    best_bcuts     = bcut_all.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    best_n_buckets = nb_all.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    best_n_hashes  = nh_all.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    best_valid_loss  = valid_results.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    best_valid_space = space_actual.reshape(rshape)[np.arange(rshape[0]), best_param_idx]

    for i in range(len(best_valid_loss)):
        log_str += 'space: %.2f, scut %.3f, bcut %d, n_buckets %d, n_hashes %d - \tloss %.2f\tspace %.1f\n' % \
            (args.space_list[i], best_scuts[i], best_bcuts[i], best_n_buckets[i], best_n_hashes[i], best_valid_loss[i], best_valid_space[i])

    # test data using best parameters
    pool = Pool(args.n_workers)
    if args.perfect_order:
        y_sorted = np.sort(np.abs(y_test))[::-1]
        results = pool.starmap(run_ccm, zip(repeat(y_sorted), best_bcuts, best_n_hashes, best_n_buckets, repeat(name)))
    elif args.lookup_data:
        results = pool.starmap(run_ccm_lookup,
            zip(repeat(x_test), repeat(y_test), best_n_hashes, best_n_buckets - best_bcuts, repeat(lookup_dict), best_scuts, repeat(name), repeat(bucket_byte), repeat(ID_byte), best_bcuts))
    else:
        results = pool.starmap(run_ccm_wscore,
            zip(repeat(y_test_ordered), repeat(test_scores), best_scuts, args.space_list, best_n_hashes, repeat(name)))
    pool.close()
    pool.join()
    test_results, space_test = zip(*results)

    log_str += '==== test test_results ====\n'
    for i in range(len(test_results)):
        log_str += 'space: %.2f, scut %.3f, bcut %d, n_buckets %d, n_hashes %d - \tloss %.2f\tspace %.1f\n' % \
               (args.space_list[i], best_scuts[i], best_bcuts[i], best_n_buckets[i], best_n_hashes[i], test_results[i], space_test[i])

    log_str += 'total time: %.2f sec\n' % (time.time() - start_t)
    print(log_str)
    with open(os.path.join(folder, args.save+'.log'), 'w') as f:
        f.write(log_str)

    np.savez(os.path.join(folder, args.save+'_test'),
        command=command,
        loss_all=test_results,
        s_cutoffs=best_scuts,
        b_cutoffs=best_bcuts,
        n_hashes=best_n_hashes,
        n_buckets=best_n_buckets,
        space_list=args.space_list,
        space_actual=space_test,
        )

