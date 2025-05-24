# Dynamic Similarity Graph Construction with Kernel Density Estimation
This repo provides the code necessary to repeat the experiments in the paper. "Dynamic Similarity Graph Construction with Kernel Density Estimation"

The algorithms are implemented in C++, and build with the cmake build tool.

## Installing Dependencies
This project has the following dependencies:
- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) (version >= 3.1)
- [Spectra](https://spectralib.org/) (version >= 1.0.1)
- [STAG](https://staglibrary.io/) (version >= 2.0.0)

You should refer to their documentation for installation instructions,
although the following should work on a standard linux system.

```bash
# Create a directory to work in
mkdir libraries
cd libraries

# Install Eigen
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xzvf eigen-3.4.0.tar.gz
cd eigen-3.4.0
mkdir build_dir
cd build_dir
cmake ..
sudo make install
cd ../..

# Install Spectra
wget https://github.com/yixuan/spectra/archive/v1.0.1.tar.gz
tar xzvf v1.0.1.tar.gz
cd spectra-1.0.1
mkdir build_dir
cd build_dir
cmake ..
sudo make install
cd ../..

# Install STAG
wget https://github.com/staglibrary/stag/archive/refs/tags/v2.0.0.tar.gz
tar xzvf v2.0.0.tar.gz
cd stag-2.0.0
mkdir build_dir
cd build_dir
cmake .. -DCMAKE_BUILD_TYPE=Release
sudo make install
cd ../..
```
## Building the dynamic algorithms code

The experimental code are compiled with two cmake build targets:

- dynamic-kde
- dynamic-similarity-graph

From the root directory of the code repository, the following commands will compile
both targets.

```bash
cmake -B build_dir -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build_dir
```

## Datasets

We evaluate the algorithms on the following datasets.
For each dataset, we show the parameter a of the Gaussian keernel used for our
experiments, and the batch size used to stream the data into the dynamic algorithms.

| dataset | n       | d    | experiment       | a        | batch size |
|---------|---------|------|------------------|----------|------------|
| blobs   | 20000   | 10   | Similarity Graph | 0.01     | 1000       |
| cifar10 | 50000   | 2048 | Similarity Graph | 0.0001   | 1000       |
| mnist   | 70000   | 728  | Similarity Graph | 0.000001 | 1000       |
| shuttle | 58000   | 9    | KDE              | 0.01     | 1000       |
| aloi    | 108000  | 128  | KDE              | 0.01     | 1000       |
| msd     | 515345  | 90   | KDE              | 0.000001 | 1000       |
| covtype | 581012  | 54   | KDE              | 0.000005 | 1000       |
| glove   | 1193514 | 100  | KDE              | 0.1      | 10000      |
| census  | 2458285 | 68   | KDE              | 0.01     | 10000      |

### Downloading the data
Each dataset can be downloaded using the `preprocess_datasets.py` python script in the
`data/` directory.
First, navigate to the `data/` directory, then
the python dependencies can be installed with
```
python -m pip install -r requirements.txt
```
and each dataset can be downloaded with
```
python preprocess_datasets.py --dataset <dataset_name>
```
This will create a text file named `<dataset>.txt` in the data directory.

## Running the Dynamic KDE Experiments
For the dynamic KDE experiments, we compare the following algorithms.

| Algorithm | Description                                 |
|-----------|---------------------------------------------|
| exact     | The exact KDE.                              |
| rs        | KDE estimates bsed on random sampling.      |
| CKNS      | The CKNS algorithm, recomputed every batch. |
| new       | The new algorithm proposed in the paper.    |

Once the c++ code has been compiled, and a dataset downloaded, you can run
the dynamic KDE experiment for a given dataset and algorithm with the following
command from the `build_dir` directory.

```bash
./dynamic-kde ../data/<dataset>.txt <results_filename> ../results/<dataset>_kde/gt.txt <algorithm> <a> <batch_size>
```

For example, to run the exact KDE algorithm on the shuttle dataset, we run

```bash
./dynamic-kde ../data/shuttle.txt ../results/shuttle_kde/exact.csv ../results/shuttle_kde/gt.txt exact 0.01 1000
```

Please note that running the experiments for the `exact` and `CKNS` algorithms on
the large datasets can take several hours.

## Running the dynamic similarity graph experiments
For the dynamic similarity graph experiments, we compare the following algorithms.

| Algorithm | Description                                     |
|-----------|-------------------------------------------------|
| fc        | Construct the fully connected similarity graph. |
| knn       | Construct the k nearest neighbor graph.         |
| new       | The new algorithm proposed in the paper.        |

You can run the dynamic similarity graph experiment for a given dataset and algorithm
with the following command from the `build_dir` directory.

```bash
./dynamic-similarity-graph ../data/<dataaset>.txt ../data/<dataset>_labels.txt <result_filename> <algorithm> <a> <batch_size> 
```

Forr example, to run the new algorithm on the blobs dataset, we run

```bash
./dynamic-similarity-graph ../data/blobs.txt ../data/blobs_labels.txt ../results/blobs_sg/new.01.csv new 0.01 1000
```
