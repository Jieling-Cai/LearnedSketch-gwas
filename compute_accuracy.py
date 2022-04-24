import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

def get_n_top_IDs_dict(n,ID_dict):
    ID_dict = dict(sorted(ID_dict.items(), key=lambda item: item[1], reverse=True))
    sub_keys = []
    counter = 0
    for key in ID_dict:
        counter+=1
        sub_keys.append(key)
        if counter == n: break
    assert len(sub_keys) == n, 'len of sub_keys is wrong'
    n_top_IDs_dict = dict([(key, ID_dict[key]) for key in sub_keys])
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
    parser.add_argument("-k","--true_top_k_list", type=list, help="the list of top k-th SNP for plotting accuracy", default=list(np.arange(1,2050,50)))
    parser.add_argument("-l","--l_return", type=int, help="top 'l_return' SNPs will be returned", default=5000)
    parser.add_argument("-space","--space", type=int, help="MB, total space previously used in hashing stage", default=4)
    parser.add_argument("-result","--result_path", type=str, help="The prefix of path of all the hashing results (lookup table & count sketch & learned sketch & ideal learned sketch)", default='/NN/cmin_results/')
    parser.add_argument("-test_chi","--test_chi_path", type=str, help="the path of test set's chi-square/CA chi-square values dictionary", default='/output/test_chi_square.npy')
    args = parser.parse_args()

    # load true statistics
    true_IDs_dict = np.load(args.test_chi_path,allow_pickle=True).item()

    # save figure
    save_figure_path = args.result_path+'result_%.2fMB.png'%args.space # path to save "accuracy" figure
    save_acc_path = args.result_path+'result_%.2fMB.npy'%args.space # path to save all the "accuracy" results. This may useful when you want to run multiple times to take the avarage results

    title = 'Space: %.2f (MB), Candidates returned: %d'%(args.space,args.l_return)
    learned_cs_IDs_dict = np.load(args.result_path+'learned_cmin_result_%.2fMB.npy' %args.space,allow_pickle=True).item()
    learned_cs_accuracy_list = get_accuracy_array(args.true_top_k_list,true_IDs_dict,args.l_return,learned_cs_IDs_dict)
    del learned_cs_IDs_dict

    ideal_cs_IDs_dict = np.load(args.result_path+'ideal_cmin_result_%.2fMB.npy' %args.space,allow_pickle=True).item()
    ideal_cs_accuracy_list = get_accuracy_array(args.true_top_k_list,true_IDs_dict,args.l_return,ideal_cs_IDs_dict)
    del ideal_cs_IDs_dict

    lookup_IDs_dict = np.load(args.result_path+'lookup_result_%.2fMB.npy' %args.space,allow_pickle=True).item()
    lookup_IDs_list = get_accuracy_array(args.true_top_k_list,true_IDs_dict,args.l_return,lookup_IDs_dict)
    del lookup_IDs_dict

    cs_IDs_dict = np.load(args.result_path+'cmin_result_%.2fMB.npy' %args.space,allow_pickle=True).item()
    cs_accuracy_list = get_accuracy_array(args.true_top_k_list,true_IDs_dict,args.l_return,cs_IDs_dict)
    del cs_IDs_dict

    All_acc = []
    All_acc.append(learned_cs_accuracy_list)
    All_acc.append(ideal_cs_accuracy_list)
    All_acc.append(lookup_IDs_list)
    All_acc.append(cs_accuracy_list)
    All_acc = np.asarray(All_acc)
    print(All_acc.shape)
    np.save(save_acc_path,All_acc)

    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 14,
            }

    plt.ylim(ymin=0,ymax=1.02)
    plt.title(title,font)
    plt.plot(args.true_top_k_list,learned_cs_accuracy_list, color='green', label='Learned CS')
    plt.plot(args.true_top_k_list,cs_accuracy_list, color='blue', linestyle='--', label='Count Sketch')
    plt.plot(args.true_top_k_list,ideal_cs_accuracy_list, color='firebrick', label='Ideal Learned CS')
    plt.plot(args.true_top_k_list,lookup_IDs_list, color='darkgoldenrod', linestyle='--', label='Table lookup CS')

    plt.legend(loc='lower left')
    plt.xlabel('k',font)
    plt.ylabel('Fraction of top k words included',font)
    plt.grid()
    plt.savefig(save_figure_path,bbox_inches="tight", pad_inches=0.1)


