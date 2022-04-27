import numpy as np
import glob
import vcf
import time

def get_data_matrix(unique_IDs_path,vcfs_path,save_path):  # extract distinct information for learning
    unique_IDs = np.load(unique_IDs_path)
    occur_dict = dict(zip(unique_IDs,np.zeros((len(unique_IDs)),dtype=np.int)))
    features = []
    for i in range(len(vcfs_path)):
        ech_vcf = vcf.Reader(filename=vcfs_path[i])
        start_time = time.time()

        for snp in ech_vcf:
            if occur_dict[snp.ID] == 0:   # this snp ID has not been seen yet
                features.append(append_features(snp))
                occur_dict[snp.ID] += 1

        t = time.time()-start_time
        print("file {} takes: {}".format(i,t))

    features = np.array(features).reshape(-1,4)
    np.save(save_path,features)
    print(np.load(save_path))

def append_features(snp):
    features=[]

    # append POS
    features.append(snp.POS)
    # append REF
    if snp.REF == 'A':
        features.append(1)
    elif snp.REF == 'C':
        features.append(2)
    elif snp.REF == 'G':
        features.append(3)
    elif snp.REF == 'T':
        features.append(4)

    # append ALT
    if snp.ALT == ['A']:
        features.append(1)
    elif snp.ALT == ['C']:
        features.append(2)
    elif snp.ALT == ['G']:
        features.append(3)
    elif snp.ALT == ['T']:
        features.append(4)

    # append ID
    features.append(int(snp.ID[2:]))

    return features

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--vcf_dirt", type=str, default=r'\data\train\*.vcf')
    argparser.add_argument("--unique_IDs_path", type=str, default=r'\output\train_unique_IDs.npy')
    argparser.add_argument("--save_path", type=str, default=r'\output\train_data.npy')
    args = argparser.parse_args()
    
    vcfs_path = glob.glob(vcf_dirt)

    get_data_matrix(unique_IDs_path,vcfs_path,save_path)

