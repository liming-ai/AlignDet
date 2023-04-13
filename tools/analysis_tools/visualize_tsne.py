import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

import seaborn as sns
sns.set_style("whitegrid")  # 白底黑线
colors = sns.color_palette()


# All the extracted feats can be download in $ssl_dir/I2B/extracted_feats_mask-rcnn


def parse_args():
    parser = argparse.ArgumentParser(description='Save TSNE')

    parser.add_argument('pretrained_feats')
    parser.add_argument('pretrained_labels')

    parser.add_argument('random_feats')
    parser.add_argument('random_labels')

    # t-SNE settings
    parser.add_argument(
        '--n-components', type=int, default=2, help='the dimension of results')
    parser.add_argument(
        '--perplexity',
        type=float,
        default=30.0,
        help='The perplexity is related to the number of nearest neighbors'
        'that is used in other manifold learning algorithms.')
    parser.add_argument(
        '--early-exaggeration',
        type=float,
        default=12.0,
        help='Controls how tight natural clusters in the original space are in'
        'the embedded space and how much space will be between them.')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=200.0,
        help='The learning rate for t-SNE is usually in the range'
        '[10.0, 1000.0]. If the learning rate is too high, the data may look'
        'like a ball with any point approximately equidistant from its nearest'
        'neighbours. If the learning rate is too low, most points may look'
        'compressed in a dense cloud with few outliers.')
    parser.add_argument(
        '--n-iter',
        type=int,
        default=1000,
        help='Maximum number of iterations for the optimization. Should be at'
        'least 250.')
    parser.add_argument(
        '--n-iter-without-progress',
        type=int,
        default=300,
        help='Maximum number of iterations without progress before we abort'
        'the optimization.')
    parser.add_argument(
        '--init', type=str, default='pca', help='The init method')

    args = parser.parse_args()
    return args


def tsne(args, feats, labels):
    classes = [1, 18, 53]

    feats = np.load(feats)
    labels = np.load(labels)

    feats = np.concatenate([feats[labels == i][:5000] for i in classes])
    labels = np.concatenate([labels[labels == i][:5000] for i in classes])

    tsne_model = TSNE(
        n_components=args.n_components,
        perplexity=args.perplexity,
        early_exaggeration=args.early_exaggeration,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        n_iter_without_progress=args.n_iter_without_progress,
        init=args.init,
        n_jobs=64)

    print("Start TSNE")
    result = tsne_model.fit_transform(feats)
    res_min, res_max = result.min(0), result.max(0)
    res_norm = (result - res_min) / (res_max - res_min)

    return res_norm, labels


def main():
    args = parse_args()

    pretrain_feats, pretrain_labels = tsne(args, args.pretrained_feats, args.pretrained_labels)
    random_feats, random_labels = tsne(args, args.random_feats, args.random_labels)

    print("Drawing pictures")
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(random_feats[:5000, 0], random_feats[:5000, 1], alpha=0.7, s=15, c=colors[0], label='Person', marker='o')
    plt.scatter(random_feats[5000:10000, 0], random_feats[5000:10000, 1], alpha=0.7, s=15, c=colors[1], label='Dog', marker='*')
    plt.scatter(random_feats[10000:, 0], random_feats[10000:, 1], alpha=0.7, s=15, c=colors[2], label='Apple', marker='x')
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.xlabel('Clustering Based on Box Sizes', fontsize=12)
    plt.title('Random Initialization', fontsize=16)

    plt.subplot(1, 2, 2)
    plt.scatter(pretrain_feats[:5000, 0], pretrain_feats[:5000, 1], alpha=0.7, s=15, c=colors[0], label='Person', marker='o')
    plt.scatter(pretrain_feats[5000:10000, 0], pretrain_feats[5000:10000, 1], alpha=0.7, s=15, c=colors[1], label='Dog', marker='*')
    plt.scatter(pretrain_feats[10000:, 0], pretrain_feats[10000:, 1], alpha=0.7, s=15, c=colors[2], label='Apple', marker='x')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Clustering Based on Semantics', fontsize=12)
    plt.title('Ours', fontsize=16)

    plt.tight_layout()
    plt.savefig(f'tsne_perplexity-30_person-dog-apple.pdf', dpi=256)
    plt.savefig(f'tsne_perplexity-30_person-dog-apple.png', dpi=256)

if __name__ == '__main__':
    main()
