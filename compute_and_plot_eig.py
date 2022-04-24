
import numpy as np
from scipy.sparse import load_npz,csr_matrix
import argparse

def generate_sparse_sketch_matrix(n_row,n_col):
    assert n_col <= n_row
    row_index = []
    col_index = []
    data = []
    col_bucket = np.random.choice(np.arange(n_col), size=n_row)
    for i in range(n_row):
        row_index.append(i)
        col_index.append(col_bucket[i])
        data.append(np.random.choice([-1,1], size=1)[0])
    sketch_matrix = csr_matrix((data, (row_index, col_index)), shape=(n_row,n_col))
    return sketch_matrix

def get_normalized_matrix(A,m):
    A = A.toarray().astype(float)
    mean = A.mean(axis=0)
    std = (1+A.sum(axis=0))/(2+2*m)
    std = np.sqrt(std * (1 - std))
    A -= mean
    A /= std
    return np.asarray(A)

def compute_eigens(A,k):
    lamb,vectors = np.linalg.eig(np.dot(A,np.transpose(A)))
    idx = lamb.argsort()[-k:][::-1]
    vectors = vectors[:, idx]
    norm = np.linalg.norm(vectors,axis=0,ord=2,keepdims=True)
    assert norm.shape[1] == vectors.shape[1]
    vectors /= norm
    return np.asarray(vectors)

def compute_normalized_sketch_matrix(A,m,n,k,eps,name):
    matrix = []
    if name == 'sparse_sketch':
        sketch_matrix = generate_sparse_sketch_matrix(n, round(k ** 2 / eps ** 2))
        print('The shape of sketch matrix is ',sketch_matrix.shape)
        AT = A.dot(sketch_matrix)
        MT = csr_matrix((np.asarray(A.mean(axis=0)).reshape(-1, ), (np.arange(0, n, 1), np.arange(0, n, 1))),
                        shape=(n, n)).dot(sketch_matrix)
        JMT = csr_matrix(np.ones((1, n))).dot(MT).toarray()
        for i in range(m):
            matrix.append(AT.getrow(i).toarray() - JMT)
            print('Finish %d-th row'%i)
        del AT, MT, JMT
        matrix = np.asarray(matrix).reshape(m, -1)

    elif name == 'Gaussian':
        sketch_matrix = np.random.normal(size=[n, round(k ** 2 / eps ** 2)])
        print('The shape of sketch matrix is ',sketch_matrix.shape)
        mean = np.asarray(A.mean(axis=0))
        matrix = []
        for i in range(m):
            matrix.append(np.dot((A.getrow(i).toarray() - mean),sketch_matrix))
            print('Finish %d-th row'%i)
        del sketch_matrix,mean
        matrix = np.asarray(matrix).reshape(m,-1)

    elif name == 'random_sign':
        sketch_matrix = np.random.choice([-1,1], size=(n, round(k ** 2 / eps ** 2)))
        print('The shape of sketch matrix is ',sketch_matrix.shape)
        mean = np.asarray(A.mean(axis=0))
        matrix = []
        for i in range(m):
            matrix.append(np.dot((A.getrow(i).toarray() - mean),sketch_matrix))
            print('Finish %d-th row'%i)
        del sketch_matrix,mean
        matrix = np.asarray(matrix).reshape(m,-1)

    return matrix

def plot_eigens(A,U,save_figure_path,name):
    PC1 = U[:,0].reshape(1,-1)
    PC2 = U[:,1].reshape(1,-1)

    A = np.dot(A,np.transpose(A))
    A -= A.mean(axis=1)

    if name == 'true':
        PC1 = -1*np.dot(PC1,A)/1e8
        PC2 = -1*np.dot(PC2,A)/1e8
    elif name == 'Gaussian':
        PC1 = np.dot(PC1,A)/1e8
        PC2 = -1*np.dot(PC2,A)/1e8
    elif name == 'random_sign':
        PC1 = np.dot(PC1,A)/1e8
        PC2 = np.dot(PC2,A)/1e8
    elif name == 'sparse_sketch':
        PC1 = -1*np.dot(PC1,A)/1e8
        PC2 = -1*np.dot(PC2,A)/1e8

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,8))
    plt.cla()
    plt.scatter(PC1,PC2,s=8)
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 15,
            }
    if name == 'true':
        plt.xlabel('PC1:  true', font)
        plt.ylabel('PC2:  true', font)
    else:
        plt.xlabel('PC1:  approximate', font)
        plt.ylabel('PC2:  approximate', font)

    plt.savefig(save_figure_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m","--num_individuals", type=int, help="The number of training/test set's total individuals (cases + controls)", default=20)
    parser.add_argument("-k","--num_pc", type=int, help="The number of principal components you want to get", default=2)
    parser.add_argument("-eps","--eps", type=float, help="This is just a parameter for determing the number of columns of a sketch matrix: "
                                                         "round(k^2/eps^2) is the number of columns of a sketch matrix", default=0.05)
    parser.add_argument("-g","--genotype_path", type=str, help="Path of training/test set's genotype matrix (.npz file)", default='./genotype.npz')
    parser.add_argument("-ID_path","--unique_IDs_path", type=str, help="Path of training/test set's unique IDs(.npy file)", default='./unique_IDs.npy')
    parser.add_argument("-name","--sketch_name", type=str, help="The type of sketch matrix you want to use: there are three options——'sparse_sketch'，"
                                                                "'Gaussian'，'random_sign'", default='sparse_sketch')
    parser.add_argument("-eigens", "--save_eigens_path", type=str, help="Prefix of path to save training/test set's singular vectors (finally saved as 'XX_eigens.npz', XX refers to sketch matrix name)", default='./')
    parser.add_argument("-fig_approx", "--save_approx_fig_path", type=str, help="Prefix of path to save training/test set's figure of approximate singular vectors"
                                                                                       "(finally saved as 'XX_PCA_approx.png', XX refers to sketch matrix name)", default='./')
    parser.add_argument("-fig_true", "--save_true_fig_path", type=str, help="Path to save training/test set's figure of true singular vectors", default='./PCA_true.png')
    args = parser.parse_args()
    n = len(np.load(args.unique_IDs_path))  # the number of total unique SNPs
    A = load_npz(args.genotype_path) # load genotype matrix
    save_eigens_path = args.save_eigens_path+'%s_eigens.npz'%args.sketch_name # path to save singular vectors (both true and approximate)
    save_approx_figure_path = args.save_approx_fig_path+'%s_PCA_approx.png'% args.sketch_name  # path to save the figure of approximate singular vectors

    X = compute_normalized_sketch_matrix(A,args.num_individuals,n,args.num_pc,args.eps,args.sketch_name)
    approximate_eigens = compute_eigens(X,args.num_pc)
    plot_eigens(X, approximate_eigens, save_approx_figure_path, args.sketch_name)
    del X

    name = 'true'
    norm_A = get_normalized_matrix(A,args.num_individuals)
    true_eigens = compute_eigens(norm_A,args.num_pc)
    plot_eigens(norm_A,true_eigens,args.save_true_fig_path, args.sketch_name)
    del norm_A

    print('true eigenvectors:',true_eigens)
    print('approximate eigenvectors:',approximate_eigens)
    np.savez(save_eigens_path, true_eig=np.asarray(true_eigens), approximate_eig=np.asarray(approximate_eigens))
    print('Finish.')






