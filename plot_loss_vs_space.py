import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

plt.style.use('seaborn-whitegrid')

def get_best_loss_space(data):
    rshape = (len(data['space_list']), -1)
    best_param_idx = np.argmin(data['loss_all'].reshape(rshape), axis=1)
    loss = data['loss_all'].reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    space_actual = data['space_actual'].reshape(rshape)[np.arange(rshape[0]), best_param_idx] / 1e6
    return loss, space_actual

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--count_min", type=str, default='/param_results/count_min/SNPs.npz')
    argparser.add_argument("--lookup_table_ccm", type=str, default='/param_results/lookup_table_count_min/SNPs_test.npz')
    argparser.add_argument("--perfect_ccm", type=str, default='/param_results/cutoff_count_min_param_perfect/SNPs_test.npz')
    argparser.add_argument("--learned_cmin", type=str, nargs='*', default=['/param_results/cutoff_count_min_param/SNPs_test.npz'])
    argparser.add_argument("--model_name", type=str, nargs='*', default=["Learned Count-Min Sketch"])
    argparser.add_argument("--model_size", type=float, nargs='*', default=[])
    argparser.add_argument("--x_lim", type=float, nargs='*', default=[])
    argparser.add_argument("--y_lim", type=float, nargs='*', default=[])
    argparser.add_argument("--title", type=str, default='(Count-Min Sketch)')
    argparser.add_argument("--algo", type=str, default='Count-Min Sketch')
    args = argparser.parse_args()

    save_figure_path = r'\param_results\cmin_results.png'

    if args.learned_cmin:
        if not args.model_size:
            args.model_size = np.zeros(len(args.learned_cmin))
        assert len(args.learned_cmin) == len(args.model_name), "provide names for the learned_cmin results"
        assert len(args.learned_cmin) == len(args.model_size), "provide model sizes for the learned_cmin results"

    # fig = plt.figure(figsize=(6, 3))
    fig = plt.figure()
    ax = fig.gca()

    if args.count_min:
        data = np.load(args.count_min)
        space_cmin = data['space_list']
        loss_cmin = np.amin(data['loss_all'], axis=0)
        ax.plot(space_cmin, loss_cmin, label=args.algo)

    if args.lookup_table_ccm:
        data = np.load(args.lookup_table_ccm)
        if len(data['loss_all'].shape) == 1:
            print('plot testing results for lookup table')
            ax.plot(data['space_actual'] / 1e6, data['loss_all'], linestyle='-.', label='Table Lookup ' + args.algo)
        else:
            loss_lookup, space_actual = get_best_loss_space(data)
            ax.plot(space_actual, loss_lookup, linestyle='-.', label='Table lookup ' + args.algo)

    if args.perfect_ccm:
        data = np.load(args.perfect_ccm)
        if len(data['loss_all'].shape) == 1:
            print('plot testing results for perfect CCM')
            ax.plot(data['space_actual'] / 1e6, data['loss_all'], linestyle='-.', label='Learned ' + args.algo + ' (Ideal)')
        else:
            loss_cutoff_pf, space_actual = get_best_loss_space(data)
            ax.plot(space_cmin, loss_cutoff_pf, linestyle='--', label='Learned ' + args.algo + ' (Ideal)')

    if args.learned_cmin:
        for i, cmin_result in enumerate(args.learned_cmin):
            data = np.load(cmin_result)
            if len(data['loss_all'].shape) == 1:
                print('plot testing results for cutoff cmin')
                ax.plot(data['space_actual'] / 1e6 + args.model_size[i], data['loss_all'], label=args.model_name[i])
            else:
                loss_cutoff, space_actual = get_best_loss_space(data)
                ax.plot(space_actual + args.model_size[i], loss_cutoff, label=args.model_name[i])

    ax.set_ylabel('Estimation  Error',fontsize=16)
    ax.set_xlabel('Space (MB)',fontsize=16)
    if args.y_lim:
        ax.set_ylim(args.y_lim)
    if args.x_lim:
        ax.set_xlim(args.x_lim)

    # title = 'Estimation Error vs Space - %s' % args.title
    # ax.set_title(title)
    plt.legend(loc="upper right",fontsize=12)
    plt.savefig(save_figure_path,bbox_inches="tight", pad_inches=0.1)
    plt.show()

