import glob
import vcf
import time
import numpy as np
import re
import argparse

# This script allows to collect all the unique types of feature "annotation", "Transcript biotype", and "HGVS.p"
# then assign each type an index:
def index_features(vcfs_path,save_dict_path,num_effects):
    anno_types = []
    biotype_types = []
    amino_types = [] # types of amino acid (obtained from feature "HGVS.p")
    for i in range(len(vcfs_path)):
        ech_vcf = vcf.Reader(filename=vcfs_path[i])
        start_time = time.time()
        for snp in ech_vcf:
            num_effects_counters=0
            for annotation in snp.INFO['ANN']:
                num_effects_counters+=1
                anno_parse = annotation.split('|')
                # "annotation"
                if anno_parse[1] not in anno_types:
                    anno_types.append(anno_parse[1])
                # "biotype"
                if anno_parse[7] not in biotype_types:
                    biotype_types.append(anno_parse[7])
                # "HGVS.p"
                s = re.findall(r'[0-9]+|[a/A-z/Z]+', anno_parse[10])[1:]
                if len(s)==3:  # in order to exclude dirty data
                    if s[0] not in amino_types: amino_types.append(s[0])
                    if s[2] not in amino_types: amino_types.append(s[2])

                if num_effects_counters == num_effects: break  # only take the first few effects
        t = time.time()-start_time
        print("file {} takes: {}".format(i,t))
        print('Current annotation types collection:',anno_types)
        print('Current transcript biotype types collection:',biotype_types)
        print('Current amino acid types collection:', amino_types)

    feats = np.concatenate((np.asarray(anno_types),np.asarray(biotype_types), np.asarray(amino_types)), axis=0)
    indexes = np.concatenate((np.arange(1,len(anno_types)+1,1),np.arange(1,len(biotype_types)+1,1), np.arange(1,len(amino_types)+1,1)), axis=0)

    feats_dict = dict(zip(feats,indexes))

    np.save(save_dict_path,feats_dict)
    # check results
    print(np.load(save_dict_path,allow_pickle=True).item())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-num_eff","--num_effects", type=int, help="The max number of effects(annotations) extracted from each SNP", default=1)
    parser.add_argument("-i","--vcfs_path", type=str, help="Path of test set's VCF files", default='./data/*.vcf')
    parser.add_argument("-o", "--save_dict_path", type=str, help="Path to save feature types dictionary", default='./feat_types.npy')
    args = parser.parse_args()

    index_features(glob.glob(args.vcfs_path),args.save_dict_path,args.num_effects)


