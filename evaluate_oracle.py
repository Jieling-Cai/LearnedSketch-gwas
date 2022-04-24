import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

def get_n_top_IDs_dict(n,ID_dict):
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

def compute_accuracy(true_top_k,k_true_top_IDs_dict,l_top_IDs_dict):
    ID_counter = 0
    for ID in k_true_top_IDs_dict:
        if ID in l_top_IDs_dict: ID_counter +=1
    return round(ID_counter/true_top_k,4)

def get_accuracy_array(true_top_k_list,true_IDs_dict,top_l,IDs_dict):
    accuracy_results = []
    l_top_IDs_dict = get_n_top_IDs_dict(top_l,IDs_dict)
    for true_top_k in true_top_k_list:
        k_true_top_IDs_dict = get_n_top_IDs_dict(true_top_k, true_IDs_dict)
        accuracy_results.append(compute_accuracy(true_top_k,k_true_top_IDs_dict,l_top_IDs_dict))
    return accuracy_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-k","--true_top_k_list", type=list, help="the list of top k items for plotting accuracy", default=list(np.arange(1,2100,100)))
    parser.add_argument("-l","--l_return", type=int, help="top 'l_return' items will be returned as candidates", default=10000)
    parser.add_argument("-result","--result_path", type=str, help="The prefix of path of test predicted results", default='/NN/test_predict_results.npy')
    parser.add_argument("-train_counts", "--train_counts_path", type=str, help="the path of true training counts' dictionary",
                        default='/output/train_Fi.npy')
    parser.add_argument("-test_counts","--test_counts_path", type=str, help="the path of true test counts' dictionary", default='/output/test_chi_square.npy')
    args = parser.parse_args()

    # save figure
    save_figure_path = '/NN/evaluate_oracle_4_feats.png' # path to save "accuracy" figure
    title = 'Candidates Returned: %d'%args.l_return

    # load true counts
    true_IDs_dict = np.load(args.test_counts_path,allow_pickle=True).item()
    lookup_dict = np.load(args.train_counts_path,allow_pickle=True).item()
    predicts_dict = np.load(args.result_path,allow_pickle=True).item()

    model_acc_list = get_accuracy_array(args.true_top_k_list,true_IDs_dict,args.l_return,predicts_dict)
    lookup_acc_list = get_accuracy_array(args.true_top_k_list,true_IDs_dict,args.l_return,lookup_dict)
    del true_IDs_dict,lookup_dict,predicts_dict

    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 14,
            }

    # plt.ylim(ymin=0.6,ymax=1.02)
    plt.title(title,font)
    plt.plot(args.true_top_k_list,model_acc_list, color='green', label='Model', marker='+',markersize=6)
    plt.plot(args.true_top_k_list,lookup_acc_list, color='blue', label='Lookup Table', marker='+',markersize=6)
    plt.legend(loc='lower left')
    plt.xlabel('k',font)
    plt.ylabel('Fraction of top k words included',font)
    plt.grid()
    plt.savefig(save_figure_path,bbox_inches="tight", pad_inches=0.1)


