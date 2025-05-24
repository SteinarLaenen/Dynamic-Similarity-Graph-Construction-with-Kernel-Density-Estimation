#include <iostream>

#include <Eigen/Dense>
#include <graph.h>
#include <cluster.h>
#include <data.h>
#include <kde.h>

#include "kde.h"
#include "lsh.h"
#include "timing.h"
#include "cluster.h"

StagReal get_relative_error(std::vector<StagReal> estimates, std::vector<StagReal> exact) {
  StagReal total_error = 0;
  for (auto i = 0; i < estimates.size(); i++) {
    total_error += abs(estimates.at(i) - exact.at(i)) / exact.at(i);
  }
  return  total_error / (StagReal) estimates.size();
}

std::vector<stag::DataPoint> create_batch(DenseMat& data, StagInt from_idx, StagInt to_idx) {
  to_idx = MIN(to_idx, data.rows());
  std::vector<stag::DataPoint> data_batch;
  for (auto i = from_idx; i < to_idx; i++) {
    data_batch.emplace_back(data, i);
  }
  return data_batch;
}

void update_estimates(kde::AbstractDynamicKDE& kde, std::vector<StagInt>& ids,
                      std::vector<StagReal>& estimates) {
  // Get the updated estimates
  for (auto id : ids) {
    estimates.at(id) = kde.get_estimate(id);
  }
}

StagReal mean(const std::vector<StagReal>& values) {
  StagReal total = 0;
  for (auto val : values) {
    total += val;
  }
  return total / values.size();
}

long getMemoryUsage() {
  std::ifstream statusFile("/proc/self/status");
  std::string line;
  long memoryUsage = 0;

  while (std::getline(statusFile, line)) {
    if (line.substr(0, 6) == "VmRSS:") {
      std::istringstream iss(line);
      std::string key;
      long value;
      std::string unit;
      iss >> key >> value >> unit;
      memoryUsage = value;
      break;
    }
  }

  return memoryUsage; // RSS in kilobytes
}

void  run_one_experiment(cluster::AbstractDynamicSimilarityGraph& sg,
                         DenseMat& data,
                         const std::vector<StagInt>& labels,
                         StagInt batch_size,
                         std::ofstream& results_fstream,
                         StagInt base_memory,
                         StagInt max_memory) {
  StagInt n = data.rows();
  SimpleTimer timer{};

  StagInt total_time = 0;
  StagInt this_iter = 0;

  std::vector<StagInt> labels_so_far;
  std::unordered_set<StagInt> unique_labels;
  StagInt num_clusters;
  StagReal ari;
  StagReal nmi;

  StagInt memory_usage = 0;

  StagInt batch_start_index = 0;
  while (batch_start_index < data.rows()) {
    StagInt this_batch_size = MIN(batch_size, data.rows() - batch_start_index);

    std::vector<stag::DataPoint> data_batch = create_batch(
        data, batch_start_index, batch_start_index + this_batch_size);
    for (auto i = 0; i < this_batch_size; i++) {
      labels_so_far.push_back(labels.at(batch_start_index + i));
      unique_labels.insert(labels.at(batch_start_index + i));
      num_clusters = (StagInt) unique_labels.size();
    }

    timer.start();
    sg.add_data(data_batch);
    timer.stop();
    StagInt graph_update_time = timer.elapsed_ms();

    timer.start();
    stag::Graph sim_graph = sg.get_graph();
    std::vector<StagInt> clusters = stag::spectral_cluster(
        &sim_graph, num_clusters);
    timer.stop();
    StagReal avg_degree = sim_graph.average_degree();
    StagInt cluster_time = timer.elapsed_ms();
    assert(labels_so_far.size() == clusters.size());
    if (num_clusters == 1) {
      ari = 1;
      nmi = 1;
    } else {
      ari = stag::adjusted_rand_index(labels_so_far, clusters);
      nmi = stag::normalised_mutual_information(labels_so_far, clusters);
    }

    // Update the run info
    memory_usage = getMemoryUsage() - base_memory;
    total_time += graph_update_time + cluster_time;
    results_fstream << sg.algorithm_name() << ", " << this_iter;
    results_fstream << ", " << batch_start_index + this_batch_size;
    results_fstream << ", " << num_clusters;
    results_fstream << ", " << this_batch_size;
    results_fstream << ", " << memory_usage;
    results_fstream << ", " << avg_degree;
    results_fstream << ", " << total_time;
    results_fstream << ", " << graph_update_time;
    results_fstream << ", " << cluster_time;
    results_fstream << ", " << ari;
    results_fstream << ", " << nmi;
    results_fstream << std::endl;
    results_fstream.flush();
    this_iter++;

    // Update the start index for the next iteration.
    batch_start_index += batch_size;
    if (memory_usage > max_memory) break;
  }

  if (memory_usage <= max_memory) {
    std::cout << sg.algorithm_name() << " completed in " << total_time << " ms";
    std::cout << std::endl;
  } else {
    std::cout << sg.algorithm_name() << " terminated after " << total_time << " ms";
    std::cout << std::endl;
  }
}

void run_experiment(std::string data_filename,
                    std::string labels_filename,
                    const std::string& results_filename,
                    std::string algorithm,
                    StagReal a,
                    StagInt batch_size,
                    StagInt max_memory) {
  // Load some data
  DenseMat data = stag::load_matrix(data_filename);
  std::cout << "n: " << data.rows() << std::endl;
  std::cout << "d: " << data.cols() << std::endl;
  DenseMat labels = stag::load_matrix(labels_filename);
  std::vector<StagInt> labels_vec;
  for (auto i = 0; i < labels.rows(); i++) {
    labels_vec.push_back((StagInt) labels.coeff(i, 0));
  }
  assert(labels_vec.size() >= data.rows());

  // Open the results file
  std::ofstream results_fstream;
  results_fstream.open(results_filename);
  results_fstream << "alg, iter, n, k, batch_size, mem, avg_deg, total_time, update_time, cluster_time, ari, nmi" << std::endl;

  StagInt base_memory = getMemoryUsage();

  std::cout << "Running " << algorithm << " experiment..." << std::endl;
  if (algorithm == "fc") {
    cluster::DynamicFCSimGraph fcsg(a, 0.1);
    run_one_experiment(fcsg, data, labels_vec, batch_size, results_fstream,
                       base_memory, max_memory);
  } else if (algorithm == "knn"){
    cluster::DynamickNNSimGraph knnsg(20);
    run_one_experiment(knnsg, data, labels_vec, batch_size, results_fstream,
                       base_memory, max_memory);
  } else if (algorithm == "CPS"){
    cluster::NaiveDynamicCPS cps(a, data.cols());
    run_one_experiment(cps, data, labels_vec, batch_size, results_fstream,
                       base_memory, max_memory);
  } else if (algorithm == "new"){
    cluster::DynamicCPS sg(a, data.cols(),
                           (StagInt) log((StagReal) data.rows()) - 10);
    run_one_experiment(sg, data, labels_vec, batch_size, results_fstream,
                       base_memory, max_memory);
  } else {
    std::cout << "Algorithm should be one of: " << std::endl;
    std::cout << "    FC - fully connected similarity graph" << std::endl;
    std::cout << "    kNN - dynamic k-NN similarity graph" << std::endl;
    std::cout << "    NaiveCPS - reconstruct the CPS every batch" << std::endl;
    std::cout << "    DynamicCPS - our new dynamic algorithm" << std::endl;
  }

  // Close the output filestream
  results_fstream.close();
}

int main(int argc, char** args) {
#ifndef NDEBUG
  std::cerr << "Warning: Compiled in debug mode. For optimal performance, compile with -DCMAKE_BUILD_TYPE=Release." << std::endl;
#endif
  if (argc != 7) {
    std::cerr << "Wrong number of agruments." << std::endl;
    return EINVAL;
  } else {
    std::string data_filename = std::string(args[1]);
    std::string labels_filename = std::string(args[2]);
    std::string results_filename = std::string(args[3]);
    std::string algorithm = std::string(args[4]);
    StagReal a = std::stod(args[5]);
    StagInt batch_size = std::stoi(args[6]);

    std::cout << "Data filename: " << data_filename << std::endl;
    std::cout << "Labels filename: " << labels_filename << std::endl;
    std::cout << "Results filename: " << results_filename << std::endl;
    std::cout << "Algorithm: " << algorithm << std::endl;
    std::cout << "a: " << a << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    StagInt max_memory = 100000000;

    run_experiment(data_filename, labels_filename, results_filename,
                   algorithm, a,
                   batch_size, max_memory);
  }
  return 0;
}
