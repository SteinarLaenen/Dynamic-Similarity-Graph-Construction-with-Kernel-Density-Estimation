#include <iostream>

#include <Eigen/Dense>
#include <graph.h>
#include <cluster.h>
#include <data.h>
#include <kde.h>

#include "kde.h"
#include "lsh.h"
#include "timing.h"

void save_gt(const std::string& gt_filename,
             std::vector<std::vector<StagReal>>& true_kde_values){
  std::ofstream outFile(gt_filename);

  if (!outFile) {
    std::cerr << "Error opening file: " << gt_filename << std::endl;
    return;
  }

  for (const auto& row : true_kde_values) {
    for (const auto& value : row) {
      outFile << value << ' '; // Use a space as a delimiter between values
    }
    outFile << '\n'; // Use a newline to separate rows
  }

  outFile.close();
}

std::vector<std::vector<StagReal>> load_gt(const std::string& gt_filename) {
  std::ifstream inFile(gt_filename);
  std::vector<std::vector<StagReal>> data;

  if (!inFile) {
    std::cerr << "Error opening file: " << gt_filename << std::endl;
    return data;
  }

  std::string line;
  while (std::getline(inFile, line)) {
    std::istringstream lineStream(line);
    std::vector<StagReal> row;
    StagReal value;
    while (lineStream >> value) {
      row.push_back(value);
    }
    data.push_back(row);
  }

  inFile.close();

  return data;
}

StagReal get_relative_error(std::vector<StagReal> estimates, std::vector<StagReal> exact) {
  StagReal total_error = 0;
  for (auto i = 0; i < estimates.size(); i++) {
    total_error += abs(estimates.at(i) - exact.at(i)) / exact.at(i);
  }
  return  total_error / (StagReal) estimates.size();
}

StagReal get_square_error(std::vector<StagReal> estimates, std::vector<StagReal> exact) {
  StagReal total_sq_error = 0;
  for (auto i = 0; i < estimates.size(); i++) {
    total_sq_error += SQR((estimates.at(i) - exact.at(i)) / exact.at(i));
  }
  return  total_sq_error / (StagReal) estimates.size();
}

StagReal get_max_error(std::vector<StagReal> estimates, std::vector<StagReal> exact) {
  StagReal max_error = 0;
  for (auto i = 0; i < estimates.size(); i++) {
    StagReal this_error = abs(estimates.at(i) - exact.at(i)) / exact.at(i);
    if (this_error > max_error) max_error = this_error;
  }
  return  max_error;
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

std::vector<std::vector<StagReal>> compute_dummy_true_kdes(DenseMat& data,
                                                           StagInt initial_batch,
                                                           StagInt batch_size) {
  std::vector<std::vector<StagReal>> results;
  StagInt n = data.rows();

  // Compute the kdes for the initial batch
  std::vector<StagReal> estimates(data.rows(), 0.0);
  results.push_back(estimates);

  StagInt batch_start_index = initial_batch;
  while (batch_start_index < data.rows()) {
    StagInt this_batch_size = MIN(batch_size, data.rows() - batch_start_index);

    // Update the estimates
    results.push_back(estimates);

    // Update the start index for the next iteration.
    batch_start_index += batch_size;
  }

  return results;
}

StagReal mean(const std::vector<StagReal>& values) {
  StagReal total = 0;
  for (auto val : values) {
    total += val;
  }
  return total / values.size();
}

std::vector<std::vector<StagReal>> compute_true_kdes(DenseMat& data, StagReal a,
                                                     StagInt batch_size,
                                                     std::ofstream& results_fstream) {
  std::vector<std::vector<StagReal>> results;
  SimpleTimer timer{};

  StagInt n = data.rows();

  // Compute the kdes for the initial batch
  std::vector<StagReal> estimates;
  kde::DynamicExactKDE kde(a);

  StagInt this_iter = 0;
  StagInt total_time = 0;

  std::vector<stag::DataPoint> data_batch = create_batch(data, 0, batch_size);
  timer.start();
  std::vector<StagInt> updated_query_ids = kde.add_data(data_batch);
  update_estimates(kde, updated_query_ids, estimates);
  std::vector<StagInt> new_query_ids = kde.add_query(data_batch);
  estimates.resize(estimates.size() + new_query_ids.size());
  update_estimates(kde, new_query_ids, estimates);
  timer.stop();
  results.push_back(estimates);

  // Update the run info
  total_time += timer.elapsed_ms();
  results_fstream << kde.algorithm_name() << ", " << this_iter;
  results_fstream << ", " << batch_size;
  results_fstream << ", " << batch_size;
  results_fstream << ", " << mean(estimates);
  results_fstream << ", " << total_time;
  results_fstream << ", " << timer.elapsed_ms();
  results_fstream << ", " << 0;
  results_fstream << ", " << 0;
  results_fstream << ", " << 0;
  results_fstream << std::endl;
  results_fstream.flush();
  this_iter++;

  StagInt batch_start_index = batch_size;
  while (batch_start_index < data.rows()) {
    StagInt this_batch_size = MIN(batch_size, data.rows() - batch_start_index);

    data_batch = create_batch(data, batch_start_index, batch_start_index + this_batch_size);

    // Update the KDE
    timer.start();
    updated_query_ids = kde.add_data(data_batch);
    update_estimates(kde, updated_query_ids, estimates);
    new_query_ids = kde.add_query(data_batch);
    estimates.resize(estimates.size() + new_query_ids.size());
    update_estimates(kde, new_query_ids, estimates);
    timer.stop();

    // Update the run info
    total_time += timer.elapsed_ms();
    results_fstream << kde.algorithm_name() << ", " << this_iter;
    results_fstream << ", " << batch_start_index + this_batch_size;
    results_fstream << ", " << this_batch_size;
    results_fstream << ", " << mean(estimates);
    results_fstream << ", " << total_time;
    results_fstream << ", " << timer.elapsed_ms();
    results_fstream << ", " << 0;
    results_fstream << ", " << 0;
    results_fstream << ", " << 0;
    results_fstream << std::endl;
    results_fstream.flush();
    this_iter++;

    // Update the estimates
    results.push_back(estimates);

    // Update the start index for the next iteration.
    batch_start_index += batch_size;

  }

  std::cout << kde.algorithm_name() << " completed in " << total_time << " ms";
  std::cout << std::endl;

  return results;
}

void  run_one_experiment(kde::AbstractDynamicKDE& kde,
                         DenseMat& data,
                         StagInt batch_size,
                         const std::vector<std::vector<StagReal>>& true_kdes,
                         std::ofstream& results_fstream) {
  StagInt n = data.rows();
  SimpleTimer timer{};

  StagInt total_time = 0;

  StagInt this_iter = 0;

  // Add an initial batch of data points
  std::vector<StagReal> estimates;
  std::vector<stag::DataPoint> data_batch;
  std::vector<StagInt> updated_query_ids;
  std::vector<StagInt> new_query_ids;

  data_batch = create_batch(data, 0, batch_size);
  timer.start();
  updated_query_ids = kde.add_data(data_batch);
  update_estimates(kde, updated_query_ids, estimates);
  new_query_ids = kde.add_query(data_batch);
  estimates.resize(estimates.size() + new_query_ids.size());
  update_estimates(kde, new_query_ids, estimates);
  timer.stop();

  // Update the run info
  total_time += timer.elapsed_ms();
  results_fstream << kde.algorithm_name() << ", " << this_iter;
  results_fstream << ", " << batch_size;
  results_fstream << ", " << batch_size;
  results_fstream << ", " << mean(estimates);
  results_fstream << ", " << total_time;
  results_fstream << ", " << timer.elapsed_ms();
  results_fstream << ", " << get_relative_error(estimates, true_kdes.at(this_iter));
  results_fstream << ", " << get_square_error(estimates, true_kdes.at(this_iter));
  results_fstream << ", " << get_max_error(estimates, true_kdes.at(this_iter));
  results_fstream << std::endl;
  results_fstream.flush();
  this_iter++;

  StagInt batch_start_index = batch_size;
  while (batch_start_index < data.rows()) {
    StagInt this_batch_size = MIN(batch_size, data.rows() - batch_start_index);

    data_batch = create_batch(data, batch_start_index, batch_start_index + this_batch_size);

    timer.start();
    updated_query_ids = kde.add_data(data_batch);
    update_estimates(kde, updated_query_ids, estimates);
    new_query_ids = kde.add_query(data_batch);
    estimates.resize(estimates.size() + new_query_ids.size());
    update_estimates(kde, new_query_ids, estimates);
    timer.stop();

    // Update the run info
    total_time += timer.elapsed_ms();
    results_fstream << kde.algorithm_name() << ", " << this_iter;
    results_fstream << ", " << batch_start_index + this_batch_size;
    results_fstream << ", " << this_batch_size;
    results_fstream << ", " << mean(estimates);
    results_fstream << ", " << total_time;
    results_fstream << ", " << timer.elapsed_ms();
    results_fstream << ", " << get_relative_error(estimates, true_kdes.at(this_iter));
    results_fstream << ", " << get_square_error(estimates, true_kdes.at(this_iter));
    results_fstream << ", " << get_max_error(estimates, true_kdes.at(this_iter));
    results_fstream << std::endl;
    results_fstream.flush();
    this_iter++;

    // Update the start index for the next iteration.
    batch_start_index += batch_size;
  }

  std::cout << kde.algorithm_name() << " completed in " << total_time << " ms";
  std::cout << std::endl;
}

void run_experiment(std::string data_filename,
                    const std::string& results_filename,
                    const std::string& gt_filename,
                    const std::string& algorithm,
                    StagReal a,
                    StagInt batch_size) {
  // Load some data
  DenseMat data = stag::load_matrix(data_filename);

  // Open the results file
  std::ofstream results_fstream;
  results_fstream.open(results_filename);
  results_fstream << "alg, iter, n, batch_size, avg_kde, total_time, update_time, rel_err, sq_err, max_err" << std::endl;

  std::cout << "Running " << algorithm << " experiment..." << std::endl;
  if (algorithm == "exact") {
    // Compute the true KDE values for each batch
    std::vector<std::vector<StagReal>> true_kde_values = compute_true_kdes(
        data, a, batch_size, results_fstream);
    save_gt(gt_filename, true_kde_values);
  } else if (algorithm == "rs") {
    // Run the random sampling dynamic KDE algorithm
    std::vector<std::vector<StagReal>> true_kde_values = load_gt(gt_filename);
    kde::DynamicRandomSamplingKDE rs_kde(a, 0.1);
    run_one_experiment(rs_kde, data, batch_size,
                       true_kde_values, results_fstream);
  } else if (algorithm == "new") {
    // Run the dynamic KDE algorithm
    std::vector<std::vector<StagReal>> true_kde_values = load_gt(gt_filename);
    kde::DynamicKDEWithResizing kde(a, data.cols());
    run_one_experiment(kde, data, batch_size, true_kde_values,
                       results_fstream);
  } else if (algorithm == "CKNS") {
    // Run the naive dynamic KDE algorithm
    std::vector<std::vector<StagReal>> true_kde_values = load_gt(gt_filename);
    kde::NaiveDynamicCKNSWithResizing naive_kde(a, data.cols());
    run_one_experiment(naive_kde, data, batch_size,
                       true_kde_values, results_fstream);
  } else {
    std::cout << "Algorithm should be one of: " << std::endl;
    std::cout << "    exact - compute the exact KDE dynamically" << std::endl;
    std::cout << "    rs - dynamic random sampling" << std::endl;
    std::cout << "    naiveCKNS - naive recomputation of CKNS estimates" << std::endl;
    std::cout << "    dynamicCKNS - our new dynamic CKNS algorithm" << std::endl;
  }

  // Close the output filestream
  results_fstream.close();
}

int main(int argc, char** args) {
#ifndef NDEBUG
  std::cerr << "Warning: Compiled in debug mode. For optimal performance, compile with -DCMAKE_BUILD_TYPE=Release." << std::endl;
#endif
  if (argc != 7) {
    std::cerr << "Wrong number of arguments." << std::endl;
    return EINVAL;
  } else {
    std::string data_filename = std::string(args[1]);
    std::string results_filename = std::string(args[2]);
    std::string gt_filename = std::string(args[3]);
    std::string algorithm = std::string(args[4]);
    StagReal a = std::stod(args[5]);
    StagInt batch_size = std::stoi(args[6]);

    std::cout << "Data filename: " << data_filename << std::endl;
    std::cout << "Results filename: " << results_filename << std::endl;
    std::cout << "Ground truth filename: " << gt_filename << std::endl;
    std::cout << "Algorithm: " << algorithm << std::endl;
    std::cout << "a: " << a << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;

    run_experiment(data_filename, results_filename, gt_filename,
                   algorithm, a, batch_size);
  }
  return 0;
}
