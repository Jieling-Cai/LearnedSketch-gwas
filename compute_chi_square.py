import numpy as np
import vcf
import glob
import time
import argparse

dict1={'heterozygous':True}
dict2={'homozygous':True}

def compute_chi_square_value(case_or_control_number,save_path,case_path,control_path,unique_IDs_array):
    chi_square = []
    case = glob.glob(case_path)
    control = glob.glob(control_path)
    assert case, 'The case cohort\'s VCF files\' paths loaded are empty'
    assert control, 'The control cohort\'s VCF files\' paths loaded are empty'

    start_time = time.time()
    # observed snp:
    case_snp_allele = compute_snp_alleles(case,unique_IDs_array)
    control_snp_allele = compute_snp_alleles(control,unique_IDs_array)

    for key in unique_IDs_array:
        numerator = int((case_or_control_number * case_snp_allele[key] - case_or_control_number * control_snp_allele[key])) ** 2
        assert numerator>=0,'numerator cannot be negative'
        denominator = int((case_snp_allele[key] + control_snp_allele[key]) * (
                        4 * case_or_control_number - (case_snp_allele[key] + control_snp_allele[key])))

        if denominator == 0: denominator+=1 # avoid division of 0, this does not affect the result since numerator equals 0 when denominator equals 0
        assert denominator>0,'denominator cannot be zero or negative'
        chi_square.append((numerator / denominator)*(4/case_or_control_number))

    chi_square_dict = dict(zip(unique_IDs_array,np.array(chi_square)))

    # save results
    np.save(save_path,chi_square_dict)

    t = time.time() - start_time
    print("total time takes:%d"%t)


def compute_snp_alleles(path,unique_IDs_array):
    snp_allele_dict = dict(zip(unique_IDs_array,np.zeros((len(unique_IDs_array)),dtype=np.int)))
    for i in range(len(path)):
        start_time = time.time()
        ech_vcf = vcf.Reader(filename=path[i])
        for snp in ech_vcf:
            if snp.INFO == dict1:
                v_snp=1
            elif snp.INFO == dict2:
                v_snp=2
            else:
                raise Exception('The format/info of \'heterozygous/homozygous\' related column is wrong')
            try:
                snp_allele_dict[snp.ID] +=v_snp
            except:
                pass
        t = time.time()-start_time
        print("file_{} takes:{}".format(i,t))
    return snp_allele_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n","--case_or_control_number", type=int, help="The number of training/test set's cases or controls (Note that cases and controls must have to be equal", default=50)
    parser.add_argument("-case","--case_path", type=str, help="Path of training/test set's VCF files from case cohort(*.vcf files)", default='/data/train/case/*.vcf')
    parser.add_argument("-control","--control_path", type=str, help="Path of training/test set's VCF files from control cohort(*.vcf files)", default='/data/train/control/*.vcf')
    parser.add_argument("-o","--save_path", type=str, help="Path to save training/test set's ID-chi square value dictionary (.npy file)", default='/output/train_chi_square.npy')
    parser.add_argument("-ID_path","--unique_IDs_path", type=str, help="Path of training/test set's unique IDs(.npy file)", default='/output/train_unique_IDs.npy')
    args = parser.parse_args()

    compute_chi_square_value(args.case_or_control_number,args.save_path,args.case_path,args.control_path,np.load(args.unique_IDs_path))
