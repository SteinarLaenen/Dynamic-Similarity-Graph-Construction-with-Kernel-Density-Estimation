#include <iostream>
#include <random>
#include <unordered_set>
#include <set>

#include <random.h>
#include <data.h>
#include <kde.h>

#include "multithreading/ctpl_stl.h"

#include "kde.h"

kde::DynamicExactKDE::DynamicExactKDE(StagReal a_input) {
  a = a_input;
  num_data_points = 0;
}

std::vector<StagInt> kde::DynamicExactKDE::add_data(const std::vector<stag::DataPoint>& dps) {
  data_points.insert(data_points.end(), dps.begin(), dps.end());

  // Update the existing estimates with the new data points
  StagInt num_threads = std::thread::hardware_concurrency();
  if (q_ids.size() < num_threads) {
    for (auto& i : q_ids) {
      estimates.at(i) += kde::unnormalised_kde(a, dps, query_points.at(i));
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
                  estimates.at(i) += kde::unnormalised_kde(a, dps, query_points.at(i));
                }
              }
          )
      );
    }

    for (auto& future : futures) {
      future.get();
    }
  }

  num_data_points += dps.size();

  // Return the vector of all query points
  return q_ids;
}

StagReal kde::DynamicExactKDE::get_estimate(StagInt qp_id) {
  assert(qp_id < estimates.size());
  return estimates.at(qp_id) / (StagReal) num_data_points;
}

std::vector<StagInt> kde::DynamicExactKDE::add_query(
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
                  estimates.at(ids.at(i)) += kde::unnormalised_kde(
                      a, data_points, query_points.at(ids.at(i)));
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

std::vector<StagInt> kde::DynamicExactKDE::sample_neighbors(
    StagInt q_id, std::vector<StagReal> rs) {
  std::vector<StagInt> samples;
  StagReal degree = num_data_points * get_estimate(q_id);

  std::deque<StagReal> targets;
  for (auto r : rs) {
    targets.push_back(degree * r);
  }
  std::sort(targets.begin(), targets.end());

  StagReal total = 0;
  for (StagInt i = 0; i < num_data_points; i++) {
    total += gaussian_kernel(a, query_points.at(q_id), data_points.at(i));

    // Get an iterator to the first element more than the total
    auto it = std::lower_bound(targets.begin(), targets.end(), total);

    // Count the elements less than the total
    StagInt count = std::distance(targets.begin(), it);

    // Remove the satisfied targets
    targets.erase(targets.begin(), it);

    // Add the samples
    for (auto j = 0; j < count; j++) {
      samples.push_back(i);
    }

    if (targets.empty()) break;
  }

  assert(targets.empty());
  assert(samples.size() == rs.size());

  return samples;
}

std::string kde::DynamicExactKDE::algorithm_name() {
  return {"DynamicExactKDE"};
}
