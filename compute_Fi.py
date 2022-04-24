import numpy as np
import vcf
import glob
import time
import argparse

dict1={'heterozygous':True}
dict2={'homozygous':True}

def compute_Fi_value(save_path,case_paths,control_paths,unique_IDs_array):
    Fi_dict = dict(zip(unique_IDs_array, np.zeros((len(unique_IDs_array)), dtype=np.int)))
    case_paths = glob.glob(case_paths)
    control_paths = glob.glob(control_paths)
    assert case_paths, 'The VCF files\' paths of case cohort loaded are empty'
    assert control_paths, 'The VCF files\' paths of control cohort loaded are empty'

    start_time_first = time.time()

    for i in range(len(case_paths)):
        start_time = time.time()
        ech_vcf = vcf.Reader(filename=case_paths[i])
        for snp in ech_vcf:
            if snp.INFO == dict1:
                v_snp=1
            elif snp.INFO == dict2:
                v_snp=2
            else:
                raise Exception('The format/info of \'heterozygous/homozygous\' related column is wrong')
            try:
                Fi_dict[snp.ID] += v_snp
            except:
                pass
        t = time.time()-start_time
        print("case file_{} takes:{}".format(i,t))

    for i in range(len(control_paths)):
        start_time = time.time()
        ech_vcf = vcf.Reader(filename=control_paths[i])
        for snp in ech_vcf:
            if snp.INFO == dict1:
                v_snp=1
            elif snp.INFO == dict2:
                v_snp=2
            else:
                raise Exception('The format/info of \'heterozygous/homozygous\' related column is wrong')
            try:
                Fi_dict[snp.ID] -= v_snp
            except:
                pass
        t = time.time()-start_time
        print("control file_{} takes:{}".format(i,t))

    np.save(save_path,Fi_dict)
    t = time.time() - start_time_first
    print("total time takes:%d"%t)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-case","--case_path", type=str, help="Path of training/test set's VCF files from case cohort(*.vcf files)", default='/data/test/case/*.vcf')
    parser.add_argument("-control","--control_path", type=str, help="Path of training/test set's VCF files from control cohort(*.vcf files)", default='/data/test/control/*.vcf')
    parser.add_argument("-o","--save_path", type=str, help=" Path to save training/test set's ID-Fi value dictionary (.npy file)", default='/output/test_Fi.npy')
    parser.add_argument("-ID_path","--unique_IDs_path", type=str, help="Path of training/test set's unique IDs(.npy file)", default='/output/test_unique_IDs.npy')
    args = parser.parse_args()

    compute_Fi_value(args.save_path,args.case_path,args.control_path, np.load(args.unique_IDs_path))




