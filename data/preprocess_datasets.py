import numpy as np
import os
import random
import sys
import time
import argparse
import bz2
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
try:
        from urllib import urlretrieve
        from urllib import urlopen
except ImportError:
        from urllib.request import urlretrieve
        from urllib.request import urlopen

DATA_DIRECTORY = '.'

def download(src, dst):
    if not os.path.exists(dst):
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)


def covtype_preprocess(fn):
    import gzip
    X = []
    with gzip.open(fn, 'rt') as t:
        for line in t.readlines():
            X.append([int(x) for x in line.strip().split(",")][:-1])
    return np.array(X) # ,dtype=np.float64)


def blobs50_preprocess(_):
    from sklearn.datasets import make_blobs
    X, Y = make_blobs(n_samples=20000, n_features=50, centers=10,
                      random_state=1)
    return X, Y


def unzip(bz2_filename, new_filename):
    with open(new_filename, 'wb') as new_file, bz2.BZ2File(bz2_filename, 'rb') as file:
        for data in iter(lambda : file.read(100 * 1024), b''):
            new_file.write(data)


def cifar10_preprocess(filename):
    unzip(filename, "cifar10.npz")
    data_loaded = np.load("cifar10.npz")
    x_train, y_train, _, _ = data_loaded['x_train'], data_loaded['y_train'], data_loaded['x_test'], data_loaded['y_test']
    Y = np.argmax(y_train, axis=1, out=None).astype(int)
    return x_train, Y


def census_preprocess(filename):
    X = []
    with open(filename) as f:
        # skip headerline, drop caseid
        for line in f.readlines()[1:]:
            X.append(list(map(int, line.split(",")[1:])))
    return np.array(X) #, dtype=np.float64)


def shuttle_preprocess(filenames):
    import zipfile
    X = []
    for fn in filenames:
        if fn.endswith(".Z"):
            if not os.path.exists(fn[:-2]):
                os.system("uncompress " + fn)
            fn = fn[:-2]
        with open(fn) as f:
            for line in f:
                # drop the class label
                X.append([int(x) for x in line.split()][:-1])
    return np.array(X)#,dtype=np.float64)


def glove_preprocess(fn):
    import zipfile
    d=100

    with zipfile.ZipFile(fn) as z:
        # print('preparing %s' % out_fn)
        z_fn = 'glove.twitter.27B.%dd.txt' % d
        X = []
        for line in z.open(z_fn):
            v = [float(x) for x in line.strip().split()[1:]]
            X.append(np.array(v))
    return np.array(X)


def mnist_preprocess(fn):
    mnist = fetch_openml('mnist_784')
    X = mnist.data
    Y = mnist.target.astype(int)
    return X.to_numpy(), Y.to_numpy()


def aloi_preprocess(fn):
    # AR14a
    # Anderson Rocha and Siome Goldenstein.
    # Multiclass from binary: Expanding one-vs-all, one-vs-one and ECOC-based approaches.
    # IEEE Transactions on Neural Networks and Learning Systems, 25(2):289–302, 2014.
    import tarfile
    with tarfile.open(fn, 'r') as f:
        with f.extractfile('aloi.data') as d:
            data = d.read().decode('ascii').strip().split('\n')
            X = [list(map(int,data[i].split(' ')[1:-1])) \
                     for i in range(1,len(data))]
    return np.array(X)


def msd_preprocess(fn):
    import zipfile
    X = list()
    with zipfile.ZipFile(fn,'r') as z:
        with z.open('YearPredictionMSD.txt','r') as f:
            X = [list(map(float, line.decode('ascii').strip().split(',')[1:])) \
                              for line in f]
    return np.array(X)


# url, filename_prefix, raw_filename, preprocess_function, save_labels?
DATASETS = {
    'glove' : ('http://nlp.stanford.edu/data/glove.twitter.27B.zip', 'glove',
               'glove.twitter.27B.zip', glove_preprocess, False),
    'mnist' : (None, 'mnist', None, mnist_preprocess, True),
    'covtype' : ('https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz',
                 'covtype', 'covtype.gz', covtype_preprocess, False),
    'census' : ('https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt',
                'census', 'census.txt', census_preprocess, False),
    'shuttle' : (list("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/%s" % dn \
                      for dn in ("shuttle.trn.Z", "shuttle.tst")),
                 'shuttle', ["shuttle.trn.Z", "shuttle.tst"], shuttle_preprocess, False),
    'aloi' : ('https://ic.unicamp.br/~rocha/pub/downloads/2014-tnnls/aloi.tar.gz',
              'aloi', 'aloi.tar.gz', aloi_preprocess, False),
    'msd'  : ('https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip',
              'msd', 'YearPredictionMSD.txt.zip', msd_preprocess, False),
    'cifar10': ('https://www.dropbox.com/s/nflroijnpeqn3b2/cifar10_embeddings.npz.bz2?dl=1',
                'cifar10', "cifar10.npz.bz2", cifar10_preprocess, True),
    'blobs': (None, 'blobs', None, blobs50_preprocess, True),
}


def create_dataset(dataset):
    (url, filename_prefix, raw_filename, preprocess_function, save_labels) = DATASETS[dataset]
    if raw_filename is None:
        download_filename = None
    elif isinstance(url,list):
        assert isinstance(raw_filename,list)
        assert len(url) == len(raw_filename)
        download_filename = list()
        for (u,dfn) in zip(url,map(lambda f: f'{DATA_DIRECTORY}/{f}', raw_filename)):
            download(u,dfn)
            download_filename.append(dfn)
    else:
        assert isinstance(raw_filename,str)
        download_filename = f'{DATA_DIRECTORY}/{raw_filename}'
        download(url, download_filename)

    if save_labels:
        start = time.time()
        X, Y = preprocess_function(download_filename)
        end = time.time()
        print(f'data preprocessing took {end - start} s')

        output_filename = f'{DATA_DIRECTORY}/{filename_prefix}.txt'
        labels_filename = f'{DATA_DIRECTORY}/{filename_prefix}_labels.txt'
        with open(output_filename, 'w') as fout:
            with open(labels_filename, 'w') as fout_labs:
                for cluster in range(np.max(Y) + 1):
                    print(f"Writing cluster {cluster}...")
                    for i in range(X.shape[0]):
                        if str(Y[i]) == str(cluster):
                            for j in range(X.shape[1]):
                                fout.write(str(X[i, j]))
                                fout.write(" ")
                            fout.write("\n")
                            fout_labs.write(str(Y[i]))
                            fout_labs.write("\n")
    else:
        start = time.time()
        X = preprocess_function(download_filename).astype(np.float64)
        end = time.time()
        print(f'data preprocessing took {end-start} s')

        output_filename = f'{DATA_DIRECTORY}/{filename_prefix}.txt'
        np.savetxt(output_filename, X)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        required=True)
    args = parser.parse_args()

    create_dataset(args.dataset)


if __name__ == "__main__":
    main()

