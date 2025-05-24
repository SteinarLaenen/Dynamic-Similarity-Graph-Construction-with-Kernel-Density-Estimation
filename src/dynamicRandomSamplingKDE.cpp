#include <iostream>
#include <random>
#include <unordered_set>
#include <set>

#include <random.h>
#include <data.h>
#include <kde.h>

#include "multithreading/ctpl_stl.h"

#include "kde.h"

StagReal kde::unnormalised_kde(StagReal a,
                               const std::vector<stag::DataPoint>& data,
                               const stag::DataPoint& query) {
  StagReal total = 0;
  for (const auto& i : data) {
    total += stag::gaussian_kernel(a, query, i);
  }
  return total;
}

kde::DynamicRandomSamplingKDE::DynamicRandomSamplingKDE(StagReal a_input,
                                                        StagReal p_input) {
  a = a_input;
  p = p_input;
  num_sampled_points = 0;
}

std::vector<StagInt> kde::DynamicRandomSamplingKDE::add_data(const std::vector<stag::DataPoint>& dps) {
  // Sub sample the data
  std::uniform_real_distribution<StagReal> dist(0, 1);
  std::vector<stag::DataPoint> subsampled_dps;

  // Sub-sample the data matrix.
  for (auto& dp : dps) {
    if (dist(*stag::get_global_rng()) <= p) {
      num_sampled_points++;
      subsampled_dps.push_back(dp);
      data_points.push_back(dp);
    }
  }

  // Update the existing estimates with the new data points
  StagInt num_threads = std::thread::hardware_concurrency();
  if (q_ids.size() < num_threads) {
    for (auto& i : q_ids) {
      estimates.at(i) += kde::unnormalised_kde(a, subsampled_dps, query_points.at(i));
    }
  } else {
    // Start the thread pool
    ctpl::thread_pool pool((int) num_threads);

    StagInt chunk_size = floor((StagReal) q_ids.size() / (StagReal) num_threads);

    // The query size is large enough to be worth splitting.
    std::vector<std::future<void>> futures;
    for (auto chunk_id = 0; chunk_id < num_threads; chunk_id++) {
      futures.push_back(
          pool.push(
              [&, chunk_size, chunk_id, num_threads](int id) {
                StagInt this_chunk_start = chunk_id * chunk_size;
                StagInt this_chunk_end = this_chunk_start + chunk_size;
                if (chunk_id == num_threads - 1) {
                  this_chunk_end = q_ids.size();
                }

                for (auto i = this_chunk_start; i < this_chunk_end; i++) {
                  estimates.at(i) += kde::unnormalised_kde(a, subsampled_dps, query_points.at(i));
                }
              }
          )
      );
    }

    for (auto& future : futures) {
      future.get();
    }
  }

  // Return the vector of all query points
  return q_ids;
}

StagReal kde::DynamicRandomSamplingKDE::get_estimate(StagInt qp_id) {
  assert(qp_id < estimates.size());
  return estimates.at(qp_id) / (StagReal) num_sampled_points;
}

std::vector<StagInt> kde::DynamicRandomSamplingKDE::add_query(
    const std::vector<stag::DataPoint>& query_mat) {
  auto first_new_id = (StagInt) estimates.size();

  std::vector<StagInt> ids(query_mat.size());
  for (auto i = 0; i < query_mat.size(); i++) {
    estimates.push_back(0);
    q_ids.push_back(first_new_id + i);
    ids.at(i) = first_new_id + i;
  }
  query_points.insert(query_points.end(), query_mat.begin(), query_mat.end());

  // Parallelize the queries for the new query point
  StagInt num_threads = std::thread::hardware_concurrency();
  if (ids.size() < num_threads) {
    for (auto& i : ids) {
      estimates.at(i) += kde::unnormalised_kde(a, data_points, query_points.at(i));
    }
  } else {
    // Start the thread pool
    ctpl::thread_pool pool((int) num_threads);

    StagInt chunk_size = floor((StagReal) ids.size() / (StagReal) num_threads);

    // The query size is large enough to be worth splitting.
    std::vector<std::future<void>> futures;
    for (auto chunk_id = 0; chunk_id < num_threads; chunk_id++) {
      futures.push_back(
          pool.push(
              [&, chunk_size, chunk_id, num_threads](int id) {
                StagInt this_chunk_start = chunk_id * chunk_size;
                StagInt this_chunk_end = this_chunk_start + chunk_size;
                if (chunk_id == num_threads - 1) {
                  this_chunk_end = ids.size();
                }

                for (auto i = this_chunk_start; i < this_chunk_end; i++) {
                  estimates.at(ids.at(i)) += kde::unnormalised_kde(a, data_points, query_points.at(ids.at(i)));
                }
              }
          )
      );
    }

    for (auto& future : futures) {
      future.get();
    }
  }

  return ids;
}

std::string kde::DynamicRandomSamplingKDE::algorithm_name() {
  return {"DynamicRandomSamplingKDE"};
}
