import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_rank(values_dict,n):
    ID_arr = []
    value_arr = []
    for ID in values_dict:
        ID_arr.append(ID)
        value_arr.append(abs(values_dict[ID]))
    sort_index = sorted(range(len(value_arr)), key=lambda k: value_arr[k], reverse=True)
    del value_arr
    ID_arr = np.asarray(ID_arr)[sort_index]  # rank IDs
    rank = np.arange(1,n+1,1)
    rank_dict = dict(zip(ID_arr,rank))
    return rank_dict

def obtain_positions(CA_rank_dict,corrected_Fi_rank_dict):
    CA_rank = []
    corrected_Fi_rank = []
    for ID in CA_rank_dict:
        CA_rank.append(CA_rank_dict[ID])
        corrected_Fi_rank.append(corrected_Fi_rank_dict[ID])
    return np.asarray(CA_rank),np.asarray(corrected_Fi_rank)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--true_dict_path", type=str, default=r'\output\test_chi_square.npy')
    argparser.add_argument("--predict_dict_path", type=str, default=r'\NN\test_predict_results.npy')
    argparser.add_argument("--save_figure_path", type=str, default=r'\Rank_chi.png')
    args = argparser.parse_args()
    
    true_dict = np.load(true_dict_path,allow_pickle=True).item()
    n = len(true_dict)
    predict_dict = np.load(predict_dict_path,allow_pickle=True).item()

    true_dict = get_rank(true_dict,n)
    predict_dict = get_rank(predict_dict, n)

    true_dict,predict_dict = obtain_positions(true_dict,predict_dict)
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 20,
            }

    plt.figure(figsize=(8,8))
    plt.scatter(predict_dict,true_dict, s=3)
    plt.xlabel('Predicted Ranks of Words',font)
    plt.ylabel('True Ranks of Words',font)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(save_figure_path,bbox_inches="tight", pad_inches=0.1)
    plt.show()









