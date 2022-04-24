import numpy as np
import glob
import vcf
import time
from scipy.sparse import csc_matrix,save_npz
import argparse

dict1={'heterozygous':True}
dict2={'homozygous':True}

def get_genotype_matrix(unique_IDs_path,case_path,control_path,save_genotype_path,save_phenotype_path,m):
    assert (len(case_path)+len(control_path)) == m, 'The number of individuals is wrong, or case/control path is wrong'
    unique_IDs = np.load(unique_IDs_path)
    indices = {}
    for col_index, ID in enumerate(unique_IDs):
        indices[ID] = col_index

    row_index,col_index,data = get_row_col_index_and_data(case_path,control_path,indices,save_phenotype_path)
    genotype_matrix = csc_matrix((data, (row_index, col_index)), shape=(m,len(unique_IDs)))
    save_npz(save_genotype_path,genotype_matrix)

def get_row_col_index_and_data(case_path,control_path,indices,save_phenotype_path):
    row_index = []
    col_index = []
    phenotype_vector = []
    data = []
    for i in range(len(case_path)):
        ech_vcf = vcf.Reader(filename=case_path[i])
        start_time = time.time()
        phenotype_vector.append(1)
        for snp in ech_vcf:
            if snp.INFO == dict1:
                row_index.append(i)
                col_index.append(indices[snp.ID])
                data.append(1)
            elif snp.INFO == dict2:
                row_index.append(i)
                col_index.append(indices[snp.ID])
                data.append(2)
        t = time.time()-start_time
        print("case file {} takes: {}".format(i,t))

    for i in range(len(control_path)):
        ech_vcf = vcf.Reader(filename=control_path[i])
        start_time = time.time()
        phenotype_vector.append(0)
        for snp in ech_vcf:
            if snp.INFO == dict1:
                row_index.append(i+len(case_path))
                col_index.append(indices[snp.ID])
                data.append(1)
            elif snp.INFO == dict2:
                row_index.append(i+len(case_path))
                col_index.append(indices[snp.ID])
                data.append(2)
        t = time.time()-start_time
        print("control file {} takes: {}".format(i,t))

    print('phenotype vector:',phenotype_vector)
    np.save(save_phenotype_path,np.asarray(phenotype_vector))
    return row_index,col_index,data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m","--num_individuals", type=int, help="The number of training/test set's total individuals (cases + controls)", default=20)
    parser.add_argument("-case","--case_path", type=str, help="Path of training/test set's VCF files from case cohort(*.vcf files)", default='./data/case/*.vcf')
    parser.add_argument("-control","--control_path", type=str, help="Path of training/test set's VCF files from control cohort(*.vcf files)", default='./data/control/*.vcf')
    parser.add_argument("-save_g","--save_genotype_path", type=str, help="Path to save training/test set's genotype matrix (.npz file)", default='./genotype.npz')
    parser.add_argument("-save_p","--save_phenotype_path", type=str, help="Path to save training/test set's phenotype vector (.npy file)", default='./phenotype.npy')
    parser.add_argument("-ID_path","--unique_IDs_path", type=str, help="Path of training/test set's unique IDs(.npy file)", default='./unique_IDs.npy')
    args = parser.parse_args()

    get_genotype_matrix(args.unique_IDs_path,glob.glob(args.case_path),glob.glob(args.control_path),args.save_genotype_path,args.save_phenotype_path,args.num_individuals)



