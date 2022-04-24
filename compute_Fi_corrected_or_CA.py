import numpy as np
from scipy.stats.stats import pearsonr
from scipy.sparse import load_npz
import argparse

def computeQ(U, m):
    I = np.asarray(np.identity(m))
    Q = I - np.dot(U, np.transpose(U))
    return Q

def computeCA(Q, A, y, m, k, n):
    ca_chisq = []
    y = np.dot(Q, y)
    for i in range(0, n):
        r = A.getcol(i).toarray()
        r = np.dot(Q, r).reshape(-1,)
        ca_chisq.append((m-k-1)*(pearsonr(r, y)[0]**2))

    ca_chisq = np.asarray(ca_chisq)
    print('len of CA trend chi-square:', len(ca_chisq))
    print('CA-chi values: mean:', np.mean(ca_chisq), 'min:', np.min(ca_chisq), 'max:', np.max(ca_chisq), 'median:',
          np.median(ca_chisq))
    return ca_chisq

def computeDot_1(Q, A, y, m, n):
    tilde_y = np.dot(Q, y)
    tilde_1 = np.dot(Q, np.ones(m))
    R = sum(tilde_y)
    tilde_y = (m / R) * tilde_y - tilde_1
    dotprod = []
    for i in range(0, n):
        r = A.getcol(i).toarray().reshape(-1,)
        dotprod.append(np.dot(r, tilde_y))
    dotprod = np.asarray(dotprod)
    print('len of CA trend Fi:', len(dotprod))
    return dotprod

def compute_simple_Dot(Q, A, y, n):
    tilde_y = np.dot(Q, y)
    dotprod = []
    for i in range(0, n):
        r = A.getcol(i).toarray().reshape(-1,)
        dotprod.append(np.dot(r, tilde_y))
    dotprod = np.asarray(dotprod)
    print('len of CA trend Fi:', len(dotprod))
    return dotprod

def computeDot(Q, A, y, n):  # This one was used finally
    y[y==0] = -1
    y = y.reshape(1,-1)
    assert 0 not in y,'phenotype vector is wrong'
    y = np.dot(y,Q).reshape(-1,)
    dotprod = []
    for i in range(0, n):
        r = A.getcol(i).toarray().reshape(-1,)
        dotprod.append(np.dot(r,y))
    dotprod = np.asarray(dotprod)
    print('len of CA trend Fi:', len(dotprod))
    abs_dotprod = np.asarray([abs(i) for i in dotprod])
    print('absolute CA-Fi values: mean:', np.mean(abs_dotprod), 'min:', np.min(abs_dotprod), 'max:', np.max(abs_dotprod), 'median:',
          np.median(abs_dotprod))
    del abs_dotprod
    return dotprod

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m","--num_individuals", type=int, help="The number of training/test set's total individuals (cases + controls)", default=20)
    parser.add_argument("-ID_path","--unique_IDs_path", type=str, help="Path of training/test set's unique IDs(.npy file)", default='./unique_IDs.npy')
    parser.add_argument("-k","--num_pc", type=int, help="The number of principal components previously used", default=2)
    parser.add_argument("-name","--sketch_name", type=str, help="The type of sketch matrix previously used", default='sparse_sketch')
    parser.add_argument("-eigens", "--eigens_prefix_path", type=str, help="Prefix of path of training/test set's singular vectors", default='./')
    parser.add_argument("-g","--genotype_path", type=str, help="Path of training/test set's genotype matrix (.npz file)", default='./genotype.npz')
    parser.add_argument("-p","--phenotype_path", type=str, help="Path of training/test set's phenotype vector (.npy file)", default='./phenotype.npy')
    parser.add_argument("-CA_chi", "--save_CA_chi_path", type=str, help="Path to save training/test set's CA trend chi square values", default='./CA_chi.npy')
    parser.add_argument("-CA_Fi", "--save_CA_Fi_path", type=str, help="Path to save training/test set's CA trend Fi values", default='./CA_Fi.npy')
    args = parser.parse_args()

    unique_IDs = np.load(args.unique_IDs_path)  # load unique IDs
    n = len(unique_IDs)  # the number of total unique SNPs
    U = np.load(args.eigens_prefix_path+'%s_eigens.npz'%args.sketch_name) # load singular vectors
    A = load_npz(args.genotype_path) # load genotype matrix
    y = np.load(args.phenotype_path) # load phenotype vector

    U1 = U['true_eig']
    U2 = U['approximate_eig']
    Q1 = computeQ(U1, args.num_individuals) # true
    Q2 = computeQ(U2, args.num_individuals) # approximate (obtained from sketch)

    ca_chisq = computeCA(Q1, A, y, args.num_individuals, args.num_pc, n)
    np.save(args.save_CA_chi_path, dict(zip(unique_IDs, ca_chisq)))

    approximate_Fi = computeDot(Q2, A, y, n)
    np.save(args.save_CA_Fi_path, dict(zip(unique_IDs, approximate_Fi)))


