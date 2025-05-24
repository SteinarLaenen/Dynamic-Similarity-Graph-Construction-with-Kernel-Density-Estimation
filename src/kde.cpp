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
#define K2_DEFAULT_CONSTANT 1     // K_2 = C log(n) p^{-k_j}
#define K1_DEFAULT_CONSTANT 1       // K_1 = C log(n) / eps^2
#define EPS_DEFAULT 1               // K_1 = C log(n) / eps^2
#define CKNS_DEFAULT_OFFSET 0

// At a certain number of sampled points, we might as well brute-force the hash
// unit.
#define HASH_UNIT_CUTOFF 3000

#ifndef NDEBUG
#  define LOG_DEBUG(x) do { std::cerr << x; } while (0)
#else
#  define LOG_DEBUG(x)
#endif

/*
 * Used to disable compiler warning for unused variable.
 */
template<class T> void ignore_warning(const T&){}

StagReal kde::median(std::vector<StagReal> &v)
{
  size_t n = v.size() / 2;
  std::nth_element(v.begin(), v.begin()+n, v.end());
  return v[n];
}

StagReal kde::squared_distance(const stag::DataPoint& u, const stag::DataPoint& v) {
  assert(u.dimension == v.dimension);
  StagReal result = 0;
  StagReal diff;
  for (StagUInt i = 0; i < u.dimension; i++) {
    diff = u.coordinates[i] - v.coordinates[i];
    result += diff * diff;
  }
  return result;
}

StagReal kde::squared_distance_at_most(const stag::DataPoint& u,
                                       const stag::DataPoint& v,
                                       StagReal max_dist) {
  assert(u.dimension == v.dimension);
  StagReal result = 0;
  StagReal diff;
  for (StagUInt i = 0; i < u.dimension; i++) {
    diff = u.coordinates[i] - v.coordinates[i];
    result += diff * diff;
    if (result > max_dist) return -1;
  }
  return result;
}

//------------------------------------------------------------------------------
// Beginning of Dynamic KDE Implementation
//------------------------------------------------------------------------------
StagInt kde::ckns_J(StagInt n, StagInt log_nmu) {
  assert((StagReal) log_nmu <= ceil(log2((StagReal) n)));
  return ((StagInt) ceil(log2((StagReal) n))) - log_nmu;
}

StagReal kde::ckns_p_sampling(StagInt j, StagInt log_nmu, StagInt n, StagInt sampling_offset) {
  // j = 0 is the special random sampling hash unit. Sample with probability
  // 2^-offset / n.
  if (j == 0) return MIN((StagReal) 1, pow(2, -sampling_offset) / n);
  else return MIN((StagReal) 1, pow(2, (StagReal) -(j+sampling_offset)) * pow(2, (StagReal) -log_nmu));
}

StagReal kde::ckns_gaussian_rj_squared(StagInt j, StagReal a) {
  return (StagReal) j * LOG_TWO / a;
}

std::vector<StagUInt> kde::ckns_gaussian_create_lsh_params(
    StagInt J, StagInt j, StagReal a, StagReal K2_constant) {
  StagReal r_j = sqrt((StagReal) j * log(2) / a);
  StagReal p_j = lsh::LSHFunction::collision_probability(r_j);
  StagReal phi_j = ceil((((StagReal) j)/((StagReal) J)) * (StagReal) (J - j + 1));
  StagUInt k_j = MAX(1, floor(- phi_j / log2(p_j)));
  StagUInt K_2 = ceil(K2_constant * pow(2, phi_j));
  return {
      k_j, // parameter K
      K_2, // parameter L
  };
}

StagInt kde::AbstractDynamicKDE::add_query(const stag::DataPoint &q) {
  std::vector<stag::DataPoint> qs = {q};
  std::vector<StagInt> ids = this->add_query(qs);
  return ids.at(0);
}

//------------------------------------------------------------------------------
// Implementation of the CKNS Gaussian KDE Hash Unit
//
// The CKNS KDE data structure is made up of several E2LSH hashes of different
// subsets of the dataset. We define a data structure (class since we're in C++
// land) which represents one such E2LSH hash as part of the CKNS algorithm.
//
// Each unit corresponds to a given 'guess' of the value of a query KDE value,
// and a distance level from the query point. The KDE value guess is referred to
// as mu in the analysis of the CKNS algorithm, and the level is referred to by
// the index j.
//------------------------------------------------------------------------------
kde::DynamicKDEHashUnit::DynamicKDEHashUnit(
    StagReal kern_param, StagInt lognmu, StagInt j_small,
    StagReal K2_constant, StagInt prob_offset, StagInt n_input, StagInt d,
    StagInt log_nmu_iter_input, StagInt iter_input, DynamicKDE* parent_input) {
  log_nmu_iter = log_nmu_iter_input;
  iter = iter_input;
  parentKDE = parent_input;
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
    LSH_buckets = lsh::LSHWithFullHash(lsh_parameters[0], lsh_parameters[1], d);
  }

  assert(parentKDE->num_data_points == 0);
  assert(parentKDE->num_query_points == 0);
}

std::unordered_set<StagInt> kde::DynamicKDEHashUnit::add_data(
    const std::vector<stag::DataPoint>& data, std::vector<std::mutex>& hum) {
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

  std::unordered_set<StagInt> modified_q_ids;
  if (!new_data.empty()) {
    for (auto& dp : new_data) {
      std::unordered_set<StagInt> this_modified_qp = add_sampled_dp(dp, hum);
      modified_q_ids.insert(this_modified_qp.begin(), this_modified_qp.end());
    }
  }

  return modified_q_ids;
}

bool kde::DynamicKDEHashUnit::possibly_update_estimate(
    StagInt q_id, StagReal distance, std::vector<std::mutex>& hum) {
  if (distance >= shell_min_dist) {
    // Update the appropriate iter estimate in the parent kde structure
    assert(iter < hum.size());
    hum.at(iter).lock();
    assert(q_id < parentKDE->get_iter_estimates()->size());
    assert(iter < parentKDE->get_iter_estimates()->at(q_id).size());
    parentKDE->get_iter_estimates()->at(q_id).at(iter) +=
        stag::gaussian_kernel(a, distance) / p_sampling;
    hum.at(iter).unlock();
    return true;
  } else {
    return false;
  }
}

std::unordered_set<StagInt> kde::DynamicKDEHashUnit::add_sampled_dp(
    const stag::DataPoint &dp, std::vector<std::mutex>& hum) {
  // Check whether we are below the sampling cutoff before adding the new data
  bool below_cutoff_before = below_cutoff;

  num_sampled_points++;

  // If the number of sampled points is below the cutoff, or this is the 'outer
  // ring' don't create an LSH table, just store the points and we'll search
  // through them at query time.
  if (j == 0) {
    all_data.push_back(dp);
  } else if (num_sampled_points <= HASH_UNIT_CUTOFF) {
    below_cutoff = true;
    all_data.push_back(dp);
  } else {
    below_cutoff = false;
  }

  // Add the newly sampled point to the lsh table, and return the q_ids to
  // be re-sampled.
  std::unordered_set<StagInt> modified_q_ids;
  if (final_shell) {
    if (!all_q_ids.empty())  {
      for (auto q_id : all_q_ids) {
        // We only consider points whose log_nmu matches this hash function
        assert(q_id < parentKDE->get_qp_mus()->size());
        if (parentKDE->get_qp_mus()->at(q_id) == log_nmu_iter) {
          assert(q_id < parentKDE->get_query_dps()->size());
          StagReal d_sq = squared_distance(dp, parentKDE->get_query_dps()->at(q_id));
          if (possibly_update_estimate(q_id, d_sq, hum)) {
            modified_q_ids.insert(q_id);
          }
        }
      }
    }
  } else {
    // We add the data to the LSH buckets even if we are still below the
    // cutoff.
    std::unordered_set<StagInt> this_mod_q_ids = LSH_buckets.add_data(dp);

    // Check any collision points
    if (!below_cutoff_before) {
      for (auto q_id : this_mod_q_ids) {
        // We only consider points whose log_nmu matches this hash function
        assert(q_id < parentKDE->get_qp_mus()->size());
        if (parentKDE->get_qp_mus()->at(q_id) == log_nmu_iter) {
          assert(q_id < parentKDE->get_query_dps()->size());
          StagReal distance = squared_distance_at_most(
              dp, parentKDE->get_query_dps()->at(q_id), shell_max_dist);
          // Update the estimate and if it has changed, add it to the list of
          // updated points.
          if (possibly_update_estimate(q_id, distance, hum)) {
            modified_q_ids.insert(q_id);
          }
        }
      }
    } else {
      // If we are below the cutoff then we return all the query ids.
      for (auto q_id : all_q_ids) {
        // We only consider points whose log_nmu matches this hash function
        assert(q_id < parentKDE->get_qp_mus()->size());
        if (parentKDE->get_qp_mus()->at(q_id) == log_nmu_iter) {
          assert(q_id < parentKDE->get_query_dps()->size());
          StagReal distance = squared_distance_at_most(
              dp, parentKDE->get_query_dps()->at(q_id), shell_max_dist);
          if (distance > shell_min_dist) {
            // Update the estimate and if it has changed, add it to the list of
            // updated points.
            if (possibly_update_estimate(q_id, distance, hum)) {
              modified_q_ids.insert(q_id);
            }
          }
        }
      }
    }
  }

  return modified_q_ids;
}

std::unordered_set<StagInt> kde::DynamicKDEHashUnit::add_data(
    const stag::DataPoint& dp, std::vector<std::mutex>& hum) {
  // Create the sampling distribution
  std::uniform_real_distribution<StagReal> dist(0, 1);

  // Check whether to sample this point
  if (dist(*stag::get_global_rng()) <= p_sampling) {
    return add_sampled_dp(dp, hum);
  } else {
    // If we did not sample the point, return the empty set as nothing has
    // been updated.
    return {};
  }
}

StagReal kde::DynamicKDEHashUnit::query_neighbors(const stag::DataPoint& q,
                                                  const std::vector<stag::DataPoint>& neighbors) const {
  // Separate computation if this is the final shell - no maximum distance.
  if (final_shell) {
    StagReal total = 0;
    for (const auto& neighbor : neighbors) {
      // We use all the points in the final shell.
      StagReal d_sq = squared_distance(q, neighbor);
      assert(shell_min_dist >= 0);
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

StagReal kde::DynamicKDEHashUnit::query(const stag::DataPoint& q, StagInt q_id) {
  if (below_cutoff || final_shell) {
    all_q_ids.insert(q_id);
    return query_neighbors(q, all_data);
  } else {
    std::vector<stag::DataPoint> near_neighbours = LSH_buckets.get_near_neighbors(q, q_id);
    return query_neighbors(q, near_neighbours);
  }
}

//------------------------------------------------------------------------------
// Dynamic Gaussian KDE
//
// We now come to the implementation of the full Dynamic KDE data structure.
//------------------------------------------------------------------------------
void kde::DynamicKDE::initialize(StagReal gaussian_param,
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
  min_log_nmu = (StagInt) MAX(0, floor(log2((StagReal) max_n * 0.001)));
  max_log_nmu = (StagInt) ceil(log2((StagReal) max_n));
  assert(min_log_nmu <= max_log_nmu);
  assert(min_log_nmu >= 0);

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

kde::DynamicKDE::DynamicKDE(StagReal a,
                            StagReal eps,
                            StagInt n,
                            StagInt d_input) {
  StagInt K1 = ceil(K1_DEFAULT_CONSTANT * log((StagReal) n) / SQR(eps));
  StagReal K2_constant = K2_DEFAULT_CONSTANT * log((StagReal) n);
  initialize(a, K1, K2_constant, CKNS_DEFAULT_OFFSET, n, d_input);
}

kde::DynamicKDE::DynamicKDE(StagReal a, StagInt n, StagInt d_input) {
  StagInt K1 = ceil(K1_DEFAULT_CONSTANT * log((StagReal) n) / SQR(EPS_DEFAULT));
  StagReal K2_constant = K2_DEFAULT_CONSTANT * log((StagReal) n);
  initialize(a, K1, K2_constant, CKNS_DEFAULT_OFFSET, n, d_input);
}

kde::DynamicKDE::DynamicKDE(StagReal a,
                            StagInt K1,
                            StagReal K2_constant,
                            StagInt prob_offset,
                            StagInt n,
                            StagInt d_input) {
  initialize(a, K1, K2_constant, prob_offset, n, d_input);
}

StagInt kde::DynamicKDE::add_hash_unit(StagInt log_nmu_iter,
                                       StagInt log_nmu,
                                       StagInt iter,
                                       StagInt j,
                                       std::mutex& hash_units_mutex) {
  assert(log_nmu <= max_log_nmu);
  assert(log_nmu >= min_log_nmu - 1);
  DynamicKDEHashUnit new_hash_unit = DynamicKDEHashUnit(
      a, log_nmu, j, k2_constant, sampling_offset, max_n, d, log_nmu_iter, iter, this);
  hash_units_mutex.lock();
  hash_units[log_nmu_iter][iter].push_back(new_hash_unit);
  hash_units_mutex.unlock();
  return 0;
}

std::vector<stag::DataPoint>* kde::DynamicKDE::get_query_dps() {
  return &query_points;
}

std::vector<StagInt>* kde::DynamicKDE::get_qp_mus() {
  return &qp_mus;
}

std::vector<std::vector<StagReal>>* kde::DynamicKDE::get_iter_estimates() {
  return &all_iteration_estimates;
}

std::vector<StagInt> kde::DynamicKDE::add_data(
    const std::vector<stag::DataPoint>& dps) {
  std::unordered_set<StagInt> q_ids_to_update;

  // Create some mutexes to control data accesses.
  // This is to control writing to the all_iteration_estimates vector
  std::vector<std::mutex> hash_unit_mutexes(k1);

  // For each hash unit, add the data to the hash unit.
  StagInt num_threads = std::thread::hardware_concurrency();
  ctpl::thread_pool pool((int) num_threads);
  std::vector<std::future<void>> futures;
  std::mutex updates_mutex;
  for (StagInt log_nmu_iter = 0;
       log_nmu_iter < num_log_nmu_iterations;
       log_nmu_iter++) {
    StagInt log_nmu = max_log_nmu - (log_nmu_iter * 2);
    StagInt J = ckns_J(max_n, log_nmu);
    for (auto iter = 0; iter < k1; iter++) {
      // Iterate through the shells for each value of j
      // Recall that j = 0 is the special random sampling unit.
      for (auto j = 0; j <= J; j++) {
        futures.push_back(
            pool.push(
                [&, log_nmu_iter, iter, j](int id) {
                  std::unordered_set<StagInt> new_qs_to_update =
                      hash_units[log_nmu_iter][iter][j].add_data(
                          dps, hash_unit_mutexes);
                  updates_mutex.lock();
                  q_ids_to_update.insert(new_qs_to_update.begin(), new_qs_to_update.end());
                  updates_mutex.unlock();
                }
            )
        );
      }
    }
  }

  // Join all the threads
  for (auto& future: futures) {
    future.get();
  }
  pool.stop();

  num_data_points += (StagInt) dps.size();

  // Update the estimates of the changed query points
  std::vector<StagInt> ids_to_update{q_ids_to_update.begin(), q_ids_to_update.end()};
  this->update_estimates(ids_to_update);
  return ids_to_update;
}

StagReal kde::DynamicKDE::get_estimate(StagInt qp_id) {
  assert(qp_id < estimates.size());
  return estimates.at(qp_id) / (StagReal) num_data_points;
}

void kde::DynamicKDE::update_estimates(const std::vector<StagInt>& q_ids) {
  if (!q_ids.empty()) {
    std::vector<stag::DataPoint> qps;
    std::vector<StagInt> ids_for_full_update;
    for (auto id : q_ids) {
      assert(id < estimates.size());
      assert(id < all_iteration_estimates.size());
      estimates.at(id) = median(all_iteration_estimates.at(id));
      // Only run a full query when the estimate has changed by enough
      if (estimates.at(id) >= 2 * full_update_estimates.at(id)) {
        assert(id < query_points.size());
        qps.push_back(query_points.at(id));
        ids_for_full_update.push_back(id);
      }
    }

    if (!qps.empty()) {
      this->full_query(qps, ids_for_full_update);
    }
  }
}

void kde::DynamicKDE::full_query(const std::vector<stag::DataPoint>& query,
                                 StagInt base_q_id) {
  std::vector<StagInt> q_ids;
  for (auto i = 0; i < query.size(); i++) {
    q_ids.push_back(base_q_id + i);
  }
  this->full_query(query, q_ids);
}

void kde::DynamicKDE::full_query(const std::vector<stag::DataPoint>& query,
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
                ignore_warning(id);
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

void kde::DynamicKDE::chunk_query(
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
          assert((log_nmu_iter * (k1 * max_J)) + (iter * max_J) + j < hum.size());
          assert(i < query.size());
          assert(i < q_ids.size());
          hum.at((log_nmu_iter * (k1 * max_J)) + (iter * max_J) + j).lock();
          assert(iter < iter_estimates[i].size());
          assert(log_nmu_iter < hash_units.size());
          assert(iter < hash_units.at(log_nmu_iter).size());
          assert(j < hash_units.at(log_nmu_iter).at(iter).size());
          iter_estimates[i].at(iter) += hash_units.at(log_nmu_iter).at(iter).at(j)
              .query(query.at(i), q_ids.at(i));
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
        // Update the iteration estimates for this query point
        assert(i < q_ids.size());
        assert(q_ids.at(i) < all_iteration_estimates.size());
        all_iteration_estimates.at(q_ids.at(i)) = iter_estimates[i];

        // Update the value of log_nmu for this estimate
        assert(q_ids.at(i) < qp_mus.size());
        qp_mus.at(q_ids.at(i)) = log_nmu_iter;

        // Add the result to the official estimates
        assert(q_ids.at(i) < estimates.size());
        assert(q_ids.at(i) < full_update_estimates.size());
        estimates.at(q_ids.at(i)) = this_mu_estimate;
        full_update_estimates.at(q_ids.at(i)) = this_mu_estimate;
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
    assert(i < q_ids.size());
    assert(q_ids.at(i) < all_iteration_estimates.size());
    // Update the iteration estimates for this query point
    all_iteration_estimates.at(q_ids.at(i)) = iter_estimates[i];

    // Update the value of log_nmu for this estimate
    assert(q_ids.at(i) < qp_mus.size());
    qp_mus.at(q_ids.at(i)) = num_log_nmu_iterations - 1;

    // Add the result to the official estimates
    assert(q_ids.at(i) < estimates.size());
    assert(q_ids.at(i) < full_update_estimates.size());
    estimates.at(q_ids.at(i)) = last_mu_estimates[i];
    full_update_estimates.at(q_ids.at(i)) = last_mu_estimates[i];
  }
}

void kde::DynamicKDE::full_query(const stag::DataPoint &query, StagInt q_id) {
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
        iter_estimates[iter] += hash_units[log_nmu_iter][iter][j].query(query, q_id);
      }
    }

    this_mu_estimate = median(iter_estimates);

    // Check whether the estimate is at least mu, in which case we
    // return it.
    if (log(this_mu_estimate) >= (StagReal) 1.3 * log_nmu) {
      assert(q_id < all_iteration_estimates.size());
      assert(q_id < qp_mus.size());
      assert(q_id < estimates.size());
      assert(q_id < full_update_estimates.size());
      all_iteration_estimates.at(q_id) = iter_estimates;
      qp_mus.at(q_id) = log_nmu_iter;
      estimates.at(q_id) = this_mu_estimate;
      full_update_estimates.at(q_id) = this_mu_estimate;
      return;
    }

    last_mu_estimate = this_mu_estimate;
  }

  // Didn't find a good answer, return the last estimate, or 0.
  assert(q_id < all_iteration_estimates.size());
  assert(q_id < qp_mus.size());
  assert(q_id < estimates.size());
  assert(q_id < full_update_estimates.size());
  all_iteration_estimates.at(q_id) = iter_estimates;
  qp_mus.at(q_id) = 0;
  estimates.at(q_id) = last_mu_estimate;
  full_update_estimates.at(q_id) = last_mu_estimate;
}

std::vector<StagInt> kde::DynamicKDE::add_query(
    const std::vector<stag::DataPoint>& new_query_points) {
  auto first_new_id = (StagInt) estimates.size();

  // Initialise the query point data
  std::vector<StagInt> ids(new_query_points.size());
  for (auto i = 0; i < new_query_points.size(); i++) {
    all_iteration_estimates.emplace_back();
    qp_mus.push_back(0);
    estimates.emplace_back(0);
    full_update_estimates.push_back(0);
    assert(i < ids.size());
    ids.at(i) = first_new_id + i;
  }
  query_points.insert(query_points.end(), new_query_points.begin(),
                      new_query_points.end());

  // Perform a full query to initialise the estimates
  num_query_points += new_query_points.size();
  assert(all_iteration_estimates.size() == num_query_points);
  assert(qp_mus.size() == num_query_points);
  assert(estimates.size() == num_query_points);
  full_query(new_query_points, first_new_id);

  return ids;
}

std::string kde::DynamicKDE::algorithm_name() {
  return {"DynamicCKNS"};
}

//------------------------------------------------------------------------------
// Dynamic Gaussian KDE with resizing
//------------------------------------------------------------------------------
kde::DynamicKDEWithResizing::DynamicKDEWithResizing(
    StagReal a_input, StagInt d_input) {
  a = a_input;
  d = d_input;
  current_max_n = 1000;
  internal_kde = new DynamicKDE(a, current_max_n, d);
}

std::vector<StagInt> kde::DynamicKDEWithResizing::add_data(
    const std::vector<stag::DataPoint> &dps) {
  data_points.insert(data_points.end(), dps.begin(), dps.end());

  if (internal_kde->num_data_points >= current_max_n) {
    // Re-initialise
    current_max_n = 2 * internal_kde->num_data_points;
    delete internal_kde;
    internal_kde = new DynamicKDE(a, current_max_n, d);
    internal_kde->add_data(data_points);

    if (!query_points.empty()) {
      std::vector<StagInt> new_qp_ids = internal_kde->add_query(query_points);
      return new_qp_ids;
    } else {
      return {};
    }
  } else {
    return internal_kde->add_data(dps);
  }
}

std::vector<StagInt> kde::DynamicKDEWithResizing::add_query(
    const std::vector<stag::DataPoint>& q) {
  query_points.insert(query_points.end(), q.begin(), q.end());
  return internal_kde->add_query(q);
}

StagReal kde::DynamicKDEWithResizing::get_estimate(StagInt qp_id) {
  return internal_kde->get_estimate(qp_id);
}

std::string kde::DynamicKDEWithResizing::algorithm_name() {
  return "DynamicCKNS";
}

//------------------------------------------------------------------------------
// Dynamic Naive CKNS KDE with resizing
//------------------------------------------------------------------------------
kde::NaiveDynamicCKNSWithResizing::NaiveDynamicCKNSWithResizing(
    StagReal a_input, StagInt d_input) {
  a = a_input;
  d = d_input;
  current_max_n = 1000;
  internal_kde = new NaiveDynamicCKNS(a, current_max_n, d);
}

std::vector<StagInt> kde::NaiveDynamicCKNSWithResizing::add_data(
    const std::vector<stag::DataPoint> &dps) {
  data_points.insert(data_points.end(), dps.begin(), dps.end());

  if (internal_kde->num_data_points >= current_max_n) {
    // Re-initialise
    current_max_n = 2 * internal_kde->num_data_points;
    delete internal_kde;
    internal_kde = new NaiveDynamicCKNS(a, current_max_n, d);
    internal_kde->add_data(data_points);

    if (!query_points.empty()) {
      std::vector<StagInt> new_qp_ids = internal_kde->add_query(query_points);
      return new_qp_ids;
    } else {
      return {};
    }
  } else {
    return internal_kde->add_data(dps);
  }
}

std::vector<StagInt> kde::NaiveDynamicCKNSWithResizing::add_query(
    const std::vector<stag::DataPoint>& q) {
  query_points.insert(query_points.end(), q.begin(), q.end());
  return internal_kde->add_query(q);
}

StagReal kde::NaiveDynamicCKNSWithResizing::get_estimate(StagInt qp_id) {
  return internal_kde->get_estimate(qp_id);
}

std::string kde::NaiveDynamicCKNSWithResizing::algorithm_name() {
  return "NaiveDynamicCKNS";
}

