import numpy as np
import glob
import vcf
import time
import argparse

def extract_unique_IDs(vcfs_path,save_path,n_linked_lists):
    IDs_lists = [[] for i in range(n_linked_lists)]
    for i in range(len(vcfs_path)):
        ech_vcf = vcf.Reader(filename=vcfs_path[i])
        start_time = time.time()
        for snp in ech_vcf:
            if snp.ID not in IDs_lists[int(snp.ID[2:])%n_linked_lists]:
                IDs_lists[int(snp.ID[2:])%n_linked_lists].append(snp.ID)
        t = time.time() - start_time
        print("file {} takes: {}".format(i, t))

    unique_IDs_all = []
    for i in range(n_linked_lists):
        unique_IDs_all.extend(IDs_lists[i])
        IDs_lists[i].clear()

    np.save(save_path,np.array(unique_IDs_all))
    print(len(np.load(save_path)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i","--vcfs_path", type=str, help="Path of training/test set's VCF files (*.vcf files)", default='/data/test/*.vcf')
    parser.add_argument("-o","--save_path", type=str, help="Path to save training/test set's unique IDs (.npy file)", default='/output/test_unique_IDs.npy')
    parser.add_argument("-n","--n_linked_lists", type=int, help="The total number of linked lists for Hash Table with Linked List, which we use to maintain unique IDs", default=80000000)
    args = parser.parse_args()
    extract_unique_IDs(glob.glob(args.vcfs_path),args.save_path,args.n_linked_lists)

