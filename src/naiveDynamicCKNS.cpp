#include <iostream>
#include <random>
#include <unordered_set>
#include <set>

#include <random.h>
#include <data.h>
#include <kde.h>

#include "multithreading/ctpl_stl.h"

#include "kde.h"

#define TWO_ROOT_TWO 2.828427124
#define TWO_ROOT_TWOPI 5.0132565
#define LOG_TWO 0.69314718056

// The CKNS algorithm has a couple of 'magic constants' which control the
// probability guarantee and variance bounds.
#define K2_DEFAULT_CONSTANT 0.1     // K_2 = C log(n) p^{-k_j}
#define K1_DEFAULT_CONSTANT 1       // K_1 = C log(n) / eps^2
#define EPS_DEFAULT 1               // K_1 = C log(n) / eps^2
#define CKNS_DEFAULT_OFFSET 0

// At a certain number of sampled points, we might as well brute-force the hash
// unit.
#define HASH_UNIT_CUTOFF 1000

#ifndef NDEBUG
#  define LOG_DEBUG(x) do { std::cerr << x; } while (0)
#else
#  define LOG_DEBUG(x)
#endif

//------------------------------------------------------------------------------
// Beginning of Dynamic KDE Implementation
//------------------------------------------------------------------------------
/**
 * Compute the value of J for the CKNS algorithm, given
 *   - n, the number of data points
 *   - log2(n * mu), as an integer (at most log2(n)).
 *
 * @param n the number of data points
 * @param log_nmu the value of log2(n * mu)
 * @return
 */
StagInt ckns_J(StagInt n, StagInt log_nmu) {
  assert((StagReal) log_nmu <= ceil(log2((StagReal) n)));
  return ((StagInt) ceil(log2((StagReal) n))) - log_nmu;
}

//------------------------------------------------------------------------------
// Implementation of the naive dynamic KDE
//------------------------------------------------------------------------------
kde::NaiveDynamicKDEHashUnit::NaiveDynamicKDEHashUnit(
    StagReal kern_param, StagInt lognmu, StagInt j_small,
    StagReal K2_constant, StagInt prob_offset, StagInt n_input, StagInt d,
    StagInt log_nmu_iter_input, StagInt iter_input) {
  log_nmu_iter = log_nmu_iter_input;
  iter = iter_input;
  n = n_input;
  a = kern_param;
  log_nmu = lognmu;
  j = j_small;
  sampling_offset = prob_offset;
  p_sampling = ckns_p_sampling(j, log_nmu, n, sampling_offset);
  num_sampled_points = 0;

  // Get the J parameter from the value of n and log(n * mu)
  J = ckns_J(n, log_nmu);
  assert(j <= J);

  below_cutoff = true;
  shell_max_dist = ckns_gaussian_rj_squared(j, a);
  shell_min_dist = 0;
  final_shell = false;
  if (j > 1) {
    shell_min_dist= ckns_gaussian_rj_squared(j-1, a);
  } else if (j == 0) {
    final_shell = true;
    // If j = 0, then this is the 'outer' shell
    shell_min_dist = ckns_gaussian_rj_squared(J, a);
  }

  if (!final_shell) {
    // Create the E2LSH hash table.
    std::vector<StagUInt> lsh_parameters = ckns_gaussian_create_lsh_params(
        J, j, a, K2_constant);
    LSH_buckets = lsh::LSHDynamic(lsh_parameters[0], lsh_parameters[1], d);
  }
}

void kde::NaiveDynamicKDEHashUnit::add_data(
    const std::vector<stag::DataPoint>& data) {
  // Create an array of DataPoint structures which will be used to point to the
  // Eigen data matrix.
  std::vector<stag::DataPoint> new_data;

  // Create the sampling distribution
  std::uniform_real_distribution<StagReal> dist(0, 1);

  // Sub-sample the data matrix.
  for (auto& dp : data) {
    if (dist(*stag::get_global_rng()) <= p_sampling) {
      num_sampled_points++;
      new_data.push_back(dp);
    }
  }

  if (!new_data.empty()) {
    for (auto& dp : new_data) {
      add_sampled_dp(dp);
    }
  }
}

void kde::NaiveDynamicKDEHashUnit::add_sampled_dp(const stag::DataPoint &dp) {
  num_sampled_points++;

  // If the number of sampled points is below the cutoff, or this is the 'outer
  // ring' don't create an LSH table, just store the points and we'll search
  // through them at query time.
  if (final_shell) {
    all_data.push_back(dp);
  } else if (num_sampled_points <= HASH_UNIT_CUTOFF) {
    below_cutoff = true;
    all_data.push_back(dp);
  } else {
    below_cutoff = false;
  }

  // Add the newly sampled point to the lsh table
  if (!final_shell) {
    // We add the data to the LSH buckets even if we are still below the
    // cutoff.
    LSH_buckets.add_data(dp);
  }
}

void kde::NaiveDynamicKDEHashUnit::add_data(const stag::DataPoint& dp) {
  // Create the sampling distribution
  std::uniform_real_distribution<StagReal> dist(0, 1);

  // Check whether to sample this point
  if (dist(*stag::get_global_rng()) <= p_sampling) {
    add_sampled_dp(dp);
  }
}

StagReal kde::NaiveDynamicKDEHashUnit::query_neighbors(
    const stag::DataPoint& q, const std::vector<stag::DataPoint>& neighbors) const {
  // Separate computation if this is the final shell - no maximum distance.
  if (final_shell) {
    StagReal total = 0;
    for (const auto& neighbor : neighbors) {
      // We use all the points in the final shell.
      StagReal d_sq = squared_distance(q, neighbor);
      assert(shell_min_dist > 0);
      if (d_sq > shell_min_dist) {
        // Include this point in the estimate
        total += stag::gaussian_kernel(a, d_sq) / p_sampling;
      }
    }
    return total;
  } else {
    StagReal total = 0;
    for (const auto& neighbor : neighbors) {
      // We return only points that are in L_j - that is, in the annulus between
      // r_{j-1} and r_j.
      StagReal d_sq = squared_distance_at_most(q, neighbor, shell_max_dist);
      if (d_sq > shell_min_dist) {
        // Include this point in the estimate
        total += stag::gaussian_kernel(a, d_sq) / p_sampling;
      }
    }
    return total;
  }
}

StagReal kde::NaiveDynamicKDEHashUnit::query(const stag::DataPoint& q) {
  if (below_cutoff || final_shell) {
    return query_neighbors(q, all_data);
  } else {
    std::vector<stag::DataPoint> near_neighbours = LSH_buckets.get_near_neighbors(q);
    return query_neighbors(q, near_neighbours);
  }
}

void kde::NaiveDynamicCKNS::initialize(StagReal gaussian_param,
                                       StagInt K1,
                                       StagReal K2_constant,
                                       StagInt prob_offset,
                                       StagInt n_input,
                                       StagInt d_input) {
  max_n = n_input;
  num_data_points = 0;
  num_query_points = 0;
  d = d_input;
  a = gaussian_param;
  sampling_offset = prob_offset;

  // We are going to create a grid of LSH data structures:
  //   log2(n * mu) ranges from 0 to floor(log2(n))
  //   i ranges from 1 to k1.
  //   j ranges from 1 to J.
  min_log_nmu = (StagInt) floor(log2((StagReal) max_n * 0.001));
  max_log_nmu = (StagInt) ceil(log2((StagReal) max_n));
  assert(min_log_nmu <= max_log_nmu);

  num_log_nmu_iterations = (StagInt) floor((StagReal) (max_log_nmu - min_log_nmu) / 2) + 1;

  k1 = K1;
  k2_constant = K2_constant;

  hash_units.resize(num_log_nmu_iterations);
  for (StagInt log_nmu_iter = 0;
       log_nmu_iter < num_log_nmu_iterations;
       log_nmu_iter++){
    hash_units[log_nmu_iter].resize(k1);
  }

  // For each value of n * mu, we'll create an array of LSH data structures.
  std::mutex hash_units_mutex;
  for (StagInt iter = 0; iter < k1; iter++) {
    for (StagInt log_nmu_iter = 0;
         log_nmu_iter < num_log_nmu_iterations;
         log_nmu_iter++) {
      StagInt log_nmu = max_log_nmu - (log_nmu_iter * 2);
      StagInt J = ckns_J(max_n, log_nmu);
      assert(J >= 0);

      // Make sure everything works like we expect.
      assert(log_nmu <= max_log_nmu);
      assert(log_nmu >= min_log_nmu - 1);

      // j = 0 is the special random sampling hash unit.
      for (StagInt j = 0; j <= J; j++) {
        add_hash_unit(log_nmu_iter, log_nmu, iter, j, hash_units_mutex);
      }
    }
  }
}

kde::NaiveDynamicCKNS::NaiveDynamicCKNS(StagReal a,
                                        StagReal eps,
                                        StagInt n,
                                        StagInt d_input) {
  StagInt K1 = ceil(K1_DEFAULT_CONSTANT * log((StagReal) n) / SQR(eps));
  StagReal K2_constant = K2_DEFAULT_CONSTANT * log((StagReal) n);
  initialize(a, K1, K2_constant, CKNS_DEFAULT_OFFSET, n, d_input);
}

kde::NaiveDynamicCKNS::NaiveDynamicCKNS(StagReal a, StagInt n, StagInt d_input) {
  StagInt K1 = ceil(K1_DEFAULT_CONSTANT * log((StagReal) n) / SQR(EPS_DEFAULT));
  StagReal K2_constant = K2_DEFAULT_CONSTANT * log((StagReal) n);
  initialize(a, K1, K2_constant, CKNS_DEFAULT_OFFSET, n, d_input);
}

kde::NaiveDynamicCKNS::NaiveDynamicCKNS(StagReal a,
                                        StagInt K1,
                                        StagReal K2_constant,
                                        StagInt prob_offset,
                                        StagInt n,
                                        StagInt d_input) {
  initialize(a, K1, K2_constant, prob_offset, n, d_input);
}

StagInt kde::NaiveDynamicCKNS::add_hash_unit(StagInt log_nmu_iter,
                                             StagInt log_nmu,
                                             StagInt iter,
                                             StagInt j,
                                             std::mutex& hash_units_mutex) {
  assert(log_nmu <= max_log_nmu);
  assert(log_nmu >= min_log_nmu - 1);
  NaiveDynamicKDEHashUnit new_hash_unit = NaiveDynamicKDEHashUnit(
      a, log_nmu, j, k2_constant, sampling_offset, max_n, d, log_nmu_iter, iter);
  hash_units_mutex.lock();
  hash_units[log_nmu_iter][iter].push_back(new_hash_unit);
  hash_units_mutex.unlock();
  return 0;
}

std::vector<StagInt> kde::NaiveDynamicCKNS::add_data(const std::vector<stag::DataPoint>& dps) {
  // For each hash unit, add the data to the hash unit.
  for (StagInt log_nmu_iter = 0;
       log_nmu_iter < num_log_nmu_iterations;
       log_nmu_iter++) {
    StagInt log_nmu = max_log_nmu - (log_nmu_iter * 2);
    StagInt J = ckns_J(max_n, log_nmu);
    for (auto iter = 0; iter < k1; iter++) {
      // Iterate through the shells for each value of j
      // Recall that j = 0 is the special random sampling unit.
      for (auto j = 0; j <= J; j++) {
        hash_units[log_nmu_iter][iter][j].add_data(dps);
      }
    }
  }

  num_data_points += dps.size();

  // Update the estimates of all the query points
  this->update_estimates();

  // Return the vector of all query points
  return q_ids;
}

StagReal kde::NaiveDynamicCKNS::get_estimate(StagInt qp_id) {
  return estimates.at(qp_id) / (StagReal) num_data_points;
}

void kde::NaiveDynamicCKNS::update_estimates() {
  this->full_query(query_points, q_ids);
}

void kde::NaiveDynamicCKNS::full_query(
    const std::vector<stag::DataPoint>& query, StagInt base_q_id) {
  std::vector<StagInt> q_ids;
  for (auto i = 0; i < query.size(); i++) {
    q_ids.push_back(base_q_id + i);
  }
  this->full_query(query, q_ids);
}

void kde::NaiveDynamicCKNS::full_query(const std::vector<stag::DataPoint>& query,
                                       const std::vector<StagInt>& q_ids) {
  StagInt num_threads = std::thread::hardware_concurrency();

  // Create a bunch of mutexes to make sure we don't run into race conditions
  // in the hash units.
  StagInt max_J = ckns_J(max_n, 0);
  std::vector<std::mutex> hash_unit_mutexes(
      num_log_nmu_iterations * k1 * max_J + 1);

  // Split the query into num_threads chunks.
  if (query.size() < num_threads) {
    return this->chunk_query(query, 0, (StagInt) query.size(), q_ids,
                             hash_unit_mutexes);
  } else {
    // Start the thread pool
    ctpl::thread_pool pool((int) num_threads);

    // The query size is large enough to be worth splitting.
    StagInt chunk_size = floor((StagReal) query.size() / (StagReal) num_threads);
    std::vector<std::future<void>> futures;
    for (auto chunk_id = 0; chunk_id < num_threads; chunk_id++) {
      futures.push_back(
          pool.push(
              [&, chunk_size, chunk_id, num_threads, query] (int id) {
                assert(chunk_id < num_threads);
                StagInt this_chunk_start = chunk_id * chunk_size;
                StagInt this_chunk_end = this_chunk_start + chunk_size;
                if (chunk_id == num_threads - 1) {
                  this_chunk_end = query.size();
                }

                assert(this_chunk_start <= (StagInt) query.size());
                assert(this_chunk_end <= (StagInt) query.size());
                assert(this_chunk_end >= this_chunk_start);

                this->chunk_query(query,
                                  this_chunk_start,
                                  this_chunk_end,
                                  q_ids,
                                  hash_unit_mutexes);
              }
          )
      );
    }

    assert((StagInt) futures.size() == num_threads);
    for (auto chunk_id = 0; chunk_id < num_threads; chunk_id++) {
      futures[chunk_id].get();
    }

    pool.stop();
  }
}

void kde::NaiveDynamicCKNS::chunk_query(
    const std::vector<stag::DataPoint>& query, StagInt chunk_start, StagInt chunk_end,
    const std::vector<StagInt>& q_ids, std::vector<std::mutex>& hum) {
  StagInt max_J = ckns_J(max_n, 0);
  // Iterate through possible values of mu , until we find a correct one for
  // each query point.
  std::vector<StagReal> results(chunk_end - chunk_start, 0);
  std::unordered_set<StagInt> unsolved_queries;
  std::unordered_map<StagInt, StagReal> last_mu_estimates;
  assert(chunk_start < chunk_end);
  for (StagInt i = chunk_start; i < chunk_end; i++) {
    assert(i < chunk_end);
    unsolved_queries.insert(i);
    last_mu_estimates.insert(std::pair<StagInt, StagReal>(i, 0));
  }
  assert((StagInt) last_mu_estimates.size() == (chunk_end - chunk_start));

  std::unordered_map<StagInt, std::vector<StagReal>> iter_estimates;
  for (StagInt log_nmu_iter = 0;
       log_nmu_iter < num_log_nmu_iterations;
       log_nmu_iter++) {
    StagInt log_nmu = max_log_nmu - (log_nmu_iter * 2);
    assert(log_nmu >= 0);
    StagInt J = ckns_J(max_n, log_nmu);

    // Clear iter_estimates
    iter_estimates.clear();

    // Get an estimate from k1 copies of the CKNS data structure.
    // Take the median one to be the true estimate.
    for (StagInt i : unsolved_queries) {
      assert(i < chunk_end);
      std::vector<StagReal> temp(k1, 0);
      iter_estimates.insert(std::pair<StagInt, std::vector<StagReal>>(i, temp));
    }

    for (auto iter = 0; iter < k1; iter++) {
      // Iterate through the shells for each value of j
      // Recall that j = 0 is the special random sampling unit.
      for (auto j = 0; j <= J; j++) {
        assert(j <= max_J);
        for (auto i : unsolved_queries) {
          assert(i < chunk_end);
          hum.at((log_nmu_iter * (k1 * max_J)) + (iter * max_J) + j).lock();
          iter_estimates[i][iter] += hash_units[log_nmu_iter][iter][j]
              .query(query.at(i));
          hum.at((log_nmu_iter * (k1 * max_J)) + (iter * max_J) + j).unlock();
        }
      }
    }

    std::vector<StagInt> newly_solved;
    for (StagInt i : unsolved_queries) {
      assert(i < chunk_end);
      StagReal this_mu_estimate = median(iter_estimates[i]);

      // Check whether the estimate is at least mu, in which case we
      // return it.
      if (log(this_mu_estimate) >= (StagReal) 1.3 * log_nmu) {
        // Add the result to the official estimates
        estimates.at(q_ids.at(i)) = this_mu_estimate;
        newly_solved.push_back(i);
      }

      assert(i < chunk_end);
      assert(last_mu_estimates.find(i) != last_mu_estimates.end());
      last_mu_estimates[i] = this_mu_estimate;
    }

    for (StagInt i : newly_solved) {
      assert(i < chunk_end);
      assert(unsolved_queries.find(i) != unsolved_queries.end());
      unsolved_queries.erase(i);
    }
  }

  // Didn't find a good answer, return the last estimate, or 0.
  for (auto i : unsolved_queries) {
    assert(i < chunk_end);
    // Add the result to the official estimates
    estimates.at(q_ids.at(i)) = last_mu_estimates[i];
  }
}

void kde::NaiveDynamicCKNS::full_query(const stag::DataPoint &query, StagInt q_id) {
  // Iterate through possible values of mu , until we find a correct one for
  // the query.
  StagReal last_mu_estimate = 0;

  std::vector<StagReal> iter_estimates(k1, 0);
  for (auto log_nmu_iter = 0;
       log_nmu_iter < num_log_nmu_iterations;
       log_nmu_iter++) {
    StagInt log_nmu = max_log_nmu - (log_nmu_iter * 2);
    StagInt J = ckns_J(max_n, log_nmu);
    StagReal this_mu_estimate;

    // Get an estimate from k1 copies of the CKNS data structure.
    // Take the median one to be the true estimate.
    for (auto iter = 0; iter < k1; iter++) {
      // Iterate through the shells for each value of j
      // Recall that j = 0 is the special random sampling hash unit.
      for (auto j = 0; j <= J; j++) {
        iter_estimates[iter] += hash_units[log_nmu_iter][iter][j].query(query);
      }
    }

    this_mu_estimate = median(iter_estimates);

    // Check whether the estimate is at least mu, in which case we
    // return it.
    if (log(this_mu_estimate) >= (StagReal) 1.3 * log_nmu) {
      estimates.at(q_id) = this_mu_estimate;
      return;
    }

    last_mu_estimate = this_mu_estimate;
  }

  // Didn't find a good answer, return the last estimate, or 0.
  estimates.at(q_id) = last_mu_estimate;
}

std::vector<StagInt> kde::NaiveDynamicCKNS::add_query(
    const std::vector<stag::DataPoint>& query_mat) {
  auto first_new_id = (StagInt) estimates.size();

  // Initialise the query point data
  std::vector<StagInt> ids(query_mat.size());
  for (auto i = 0; i < query_mat.size(); i++) {
    estimates.push_back(0);
    q_ids.push_back(first_new_id + i);
    ids.at(i) = first_new_id + i;
  }
  query_points.insert(query_points.end(), query_mat.begin(), query_mat.end());

  // Perform a full query to initialise the estimates
  num_query_points += query_mat.size();
  assert(estimates.size() == num_query_points);
  full_query(query_mat, first_new_id);

  return ids;
}

std::string kde::NaiveDynamicCKNS::algorithm_name() {
  return {"NaiveDynamicCKNS"};
}
