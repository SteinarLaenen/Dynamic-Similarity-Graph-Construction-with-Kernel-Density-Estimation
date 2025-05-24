#include <random>
#include <random.h>
#include <cluster.h>
#include <iostream>
#include "kde.h"
#include "cluster.h"
#include "multithreading/ctpl_stl.h"

#define ASG_TREE_CUTOFF 5000

cluster::DynamicKDETreeEntry::DynamicKDETreeEntry(StagReal a_input, StagInt d_input,
                                                  StagInt this_depth_input,
                                                  StagInt max_depth_input)
    : sampling_dist(0.0, 1.0),
      this_estimator(a_input, d_input),
      exact_kde(a_input)
{
  a = a_input;
  d = d_input;
  n_node = 0;
  n_left = 0;
  n_right = 0;
  this_depth = this_depth_input;
  max_depth = max_depth_input;

  // We always begin below the cutoff since the number of nodes is 0.
  below_cutoff = true;

  // Add children up to the max depth
  if (this_depth < max_depth) {
    left_child = new DynamicKDETreeEntry(a, d, this_depth + 1, max_depth);
    right_child = new DynamicKDETreeEntry(a, d, this_depth + 1, max_depth);
  }
}

void cluster::DynamicKDETreeEntry::possibly_add_query_point(
    const stag::DataPoint& q, const StagInt global_id) {
  if (!global_to_local_q_id.contains(global_id)) {
    std::vector<stag::DataPoint> qs = {q};
    std::vector<StagInt> ids = this_estimator.add_query(qs);
    local_to_global_q_id[ids.at(0)] = global_id;
    global_to_local_q_id[global_id] = ids.at(0);

    if (below_cutoff) {
      // If we're below the cutoff, we need to add the query point to the exact
      // estimator.
      ids = exact_kde.add_query(qs);
      assert(ids.at(0) == global_to_local_q_id[global_id]);
    }
  }
}

void cluster::DynamicKDETreeEntry::possibly_add_query_points(
    const std::vector<stag::DataPoint>& qs,
    const std::vector<StagInt>& global_ids) {
  assert(qs.size() == global_ids.size());
  std::vector<StagInt> globals_to_add;
  std::vector<stag::DataPoint> qs_to_add;
  for (auto i = 0; i < qs.size(); i++) {
    if (!global_to_local_q_id.contains(global_ids.at(i))) {
      globals_to_add.push_back(global_ids.at(i));
      qs_to_add.push_back(qs.at(i));
    }
  }
  assert(globals_to_add.size() == qs_to_add.size());
  if (!globals_to_add.empty()) {
    std::vector<StagInt> local_ids = this_estimator.add_query(qs_to_add);

    assert(local_ids.size() == globals_to_add.size());
    for (auto i = 0; i < local_ids.size(); i++) {
      local_to_global_q_id[local_ids.at(i)] = globals_to_add.at(i);
      global_to_local_q_id[globals_to_add.at(i)] = local_ids.at(i);
    }

    if (below_cutoff) {
      // If we're below the cutoff, we need to add the query point to the exact
      // estimator.
      local_ids = exact_kde.add_query(qs_to_add);
      assert(local_ids.at(0) == global_to_local_q_id[globals_to_add.at(0)]);
    }
  }
}

std::unordered_set<StagInt> cluster::DynamicKDETreeEntry::add_data(
    const std::vector<stag::DataPoint>& data,
    StagInt base_global_id) {
  // Add the data to the child node with the lowest number of data points
  std::unordered_set<StagInt> changed_estimates_children;
  if (this_depth < max_depth) {
    if (n_left < n_right) {
      changed_estimates_children = left_child->add_data(data, base_global_id);
      n_left += (StagInt) data.size();
    } else {
      changed_estimates_children = right_child->add_data(data, base_global_id);
      n_right += (StagInt) data.size();
    }
  }

  // Add the data to the current node
  std::vector<StagInt> changed_estimates_local;
  if (below_cutoff) {
    changed_estimates_local = exact_kde.add_data(data);
    this_estimator.add_data(data);
  } else {
    // We stop updating the exact_kde estimator once we're above the cutoff.
    changed_estimates_local = this_estimator.add_data(data);
  }

  // Check whether the changed estimates have changed enough
  std::unordered_set<StagInt> global_ids_changed_enough;
  StagInt global_q_id;
  StagReal last_estimate;
  StagReal change_ratio;
  for (auto local_q_id : changed_estimates_local) {
    global_q_id = local_to_global_q_id[local_q_id];
    if (edge_source_nodes.contains(global_q_id)) {
      last_estimate = last_estimates[global_q_id];
      if (below_cutoff) {
        change_ratio = abs(last_estimate -
                           (n_node * exact_kde.get_estimate(local_q_id))) / last_estimate;
      } else {
        change_ratio = abs(last_estimate -
                           (n_node * this_estimator.get_estimate(local_q_id))) / last_estimate;
      }
      if (change_ratio >= 0.5) {
        global_ids_changed_enough.insert(global_q_id);
      }
    }
  }

  // Update the map of local to global ids
  StagInt next_local_id = n_node;
  for (auto i = 0; i < data.size(); i++) {
    local_to_global_d_id[next_local_id] = base_global_id + i;
    next_local_id++;
  }

  // Update the number of data points at this node
  n_node += (StagInt) data.size();

  // Check if we are no longer below the cutoff
  if (n_node > ASG_TREE_CUTOFF && this_depth < max_depth) {
    below_cutoff = false;
  }

  // Return the set of changed nodes
  global_ids_changed_enough.insert(changed_estimates_children.begin(),
                                   changed_estimates_children.end());
  return global_ids_changed_enough;
}

void cluster::DynamicKDETreeEntry::remove_q_ids(
    const std::vector<StagInt>& q_ids) {
  std::vector<StagInt> found_q_ids;
  for (auto q_id : q_ids) {
    if (edge_source_nodes.contains(q_id)) {
      found_q_ids.push_back(q_id);
    }
  }
  if (!found_q_ids.empty()) {
    for (auto q_id : found_q_ids) {
      edge_source_nodes.erase(q_id);
      last_estimates.erase(q_id);
    }
    if (this_depth < max_depth) {
      left_child->remove_q_ids(found_q_ids);
      right_child->remove_q_ids(found_q_ids);
    }
  }
}

StagReal cluster::DynamicKDETreeEntry::estimate_weight(
    const stag::DataPoint& q, const StagInt q_id) {
  possibly_add_query_point(q, q_id);

  StagReal weight;
  if (!below_cutoff) {
    weight = (StagReal) n_node * this_estimator.get_estimate(
        global_to_local_q_id[q_id]);
  } else {
    weight = (StagReal) n_node * exact_kde.get_estimate(
        global_to_local_q_id[q_id]);
  }

  // Start 'watching' this node
  if (!edge_source_nodes.contains(q_id)) {
    edge_source_nodes.insert(q_id);
    last_estimates[q_id] = weight;
  }

  return weight;
}

std::vector<StagReal> cluster::DynamicKDETreeEntry::estimate_weights(
    const std::vector<stag::DataPoint>& qs,
    const std::vector<StagInt>& q_ids) {
  possibly_add_query_points(qs, q_ids);
  assert(qs.size() == q_ids.size());
  std::vector<StagReal> weights;
  for (auto i = 0; i < qs.size(); i++) {
    weights.push_back(estimate_weight(qs.at(i), q_ids.at(i)));
  }
  return weights;
}

std::vector<EdgeTriplet> cluster::DynamicKDETreeEntry::sample_neighbors(
    const std::vector<stag::DataPoint>& qs,
    const std::vector<StagInt>& q_ids,
    const std::vector<StagInt>& nums_to_sample) {
  possibly_add_query_points(qs, q_ids);
  if (below_cutoff) {
    // Do the sampling here using the exact KDE. Start a thread pool
    StagInt num_threads = std::thread::hardware_concurrency();
    if (qs.size() < num_threads) {
      std::vector<EdgeTriplet> samples;
      for (auto i = 0; i < qs.size(); i++) {
        std::vector<StagReal> rs;
        for (auto j = 0; j < nums_to_sample.at(i); j++) {
          rs.push_back(sampling_dist(*stag::get_global_rng()));
        }
        std::vector<StagInt> local_d_ids = exact_kde.sample_neighbors(
            global_to_local_q_id[q_ids.at(i)], rs);
        std::vector<StagInt> global_d_ids;
        for (auto id : local_d_ids) {
          samples.emplace_back(q_ids.at(i), local_to_global_d_id[id], 1);
        }
      }
      return samples;
    } else {
      // Worth threading
      ctpl::thread_pool pool((int) num_threads);

      StagInt chunk_size = floor((StagReal) qs.size() / (StagReal) num_threads);
      std::vector<std::future<std::vector<EdgeTriplet>>> futures;
      for (auto chunk_id = 0; chunk_id < num_threads; chunk_id++) {
        futures.push_back(
            pool.push(
                [&, chunk_size, chunk_id, num_threads](int id) {
                  StagInt this_chunk_start = chunk_id * chunk_size;
                  StagInt this_chunk_end = this_chunk_start + chunk_size;
                  if (chunk_id == num_threads - 1) {
                    this_chunk_end = qs.size();
                  }
                  std::vector<EdgeTriplet> samples;
                  for (auto i = this_chunk_start; i < this_chunk_end; i++) {
                    std::vector<StagReal> rs;
                    for (auto j = 0; j < nums_to_sample.at(i); j++) {
                      rs.push_back(sampling_dist(*stag::get_global_rng()));
                    }
                    std::vector<StagInt> local_d_ids = exact_kde.sample_neighbors(
                        global_to_local_q_id[q_ids.at(i)], rs);
                    std::vector<StagInt> global_d_ids;
                    for (auto id : local_d_ids) {
                      samples.emplace_back(q_ids.at(i), local_to_global_d_id[id], 1);
                    }
                  }
                  return samples;
                }
            )
        );
      }

      std::vector<EdgeTriplet> all_samples;
      for (auto& future : futures) {
        std::vector<EdgeTriplet> these_samples = future.get();
        all_samples.insert(all_samples.end(), these_samples.begin(), these_samples.end());
      }
      return all_samples;
    }
  } else {
    std::vector<StagReal> left_ests = left_child->estimate_weights(
        qs, q_ids);
    std::vector<StagReal> right_ests = right_child->estimate_weights(
        qs, q_ids);
    std::vector<StagReal> my_ests;
    for (auto i = 0; i < left_ests.size(); i++) {
      my_ests.push_back(left_ests.at(i) + right_ests.at(i));
    }

    std::vector<StagInt> nums_left_samples;
    std::vector<StagInt> nums_right_samples;
    std::vector<stag::DataPoint> left_qs;
    std::vector<stag::DataPoint> right_qs;
    std::vector<StagInt> left_q_ids;
    std::vector<StagInt> right_q_ids;
    for (auto i = 0; i < left_ests.size(); i++) {
      // Get the number of left samples from a binomial distribution.
      std::binomial_distribution<> binom_dist(
          nums_to_sample.at(i), left_ests.at(i) / my_ests.at(i));
      StagInt num_left_samples = binom_dist(*stag::get_global_rng());
      if (num_left_samples > 0) {
        nums_left_samples.push_back(num_left_samples);
        left_qs.push_back(qs.at(i));
        left_q_ids.push_back(q_ids.at(i));
      }
      if (num_left_samples < nums_to_sample.at(i)) {
        nums_right_samples.push_back(
            nums_to_sample.at(i) - num_left_samples);
        right_qs.push_back(qs.at(i));
        right_q_ids.push_back(q_ids.at(i));
      }
    }

    std::vector<EdgeTriplet> left_samples;
    std::vector<EdgeTriplet> right_samples;
    if (!nums_left_samples.empty()) left_samples =
        left_child->sample_neighbors(left_qs, left_q_ids,
                                     nums_left_samples);
    if (!nums_right_samples.empty()) right_samples =
        right_child->sample_neighbors(right_qs, right_q_ids,
                                      nums_right_samples);

    left_samples.insert(left_samples.end(),
                        right_samples.begin(),
                        right_samples.end());
    return left_samples;
  }
}

cluster::DynamicKDETreeEntry::~DynamicKDETreeEntry() {
  if (this_depth < max_depth) {
    delete left_child;
    delete right_child;
  }
}

//------------------------------------------------------------------------------
// Implementation of the dynamic cluster-preserving sparsifier data structure.
//------------------------------------------------------------------------------
cluster::DynamicCPS::DynamicCPS(StagReal a_input, StagInt d_input, StagInt max_depth)
    : tree_root(a_input, d_input, 0, max_depth) {
  a = a_input;
  d = d_input;
  n_nodes = 0;
  n_nodes_at_last_edges_update = 0;

  // Start by sampling one edge per node - this will slowly increase as we add
  // more data points.
  edges_per_node = 1;
}

void cluster::DynamicCPS::add_data(const std::vector<stag::DataPoint> &dps) {
  // Add the data to the sampling tree, and get a list of nodes which need
  // their edges re-sampled.
  std::unordered_set<StagInt> ids_needing_resampled =
      tree_root.add_data(dps, n_nodes);

  // Add the data to the internal data store
  std::vector<StagInt> to_sample;
  for (auto i = 0; i < dps.size(); i++) {
    data.push_back(dps.at(i));
    to_sample.push_back(n_nodes + i);
  }
  n_nodes += (StagInt) dps.size();

  // Remove previously sampled edges for ids needing resampled
  for (auto id : ids_needing_resampled) {
    graph_edges[id] = {};
    to_sample.push_back(id);
  }

  // Check whether we need to increase the number of sampled edges per node.
  // If so, we just re-sample everything.
  if (n_nodes > 2 * n_nodes_at_last_edges_update) {
    n_nodes_at_last_edges_update = n_nodes;
    edges_per_node = 30 * (StagInt) log((StagReal) n_nodes);
    graph_edges.clear();
    to_sample = {};
    for (auto i = 0; i < n_nodes; i++) {
      to_sample.push_back(i);
    }
  }

  // Remove the nodes to sample from the tree cache
  tree_root.remove_q_ids(to_sample);

  // Sample new edges for added data points and those needing resampled
  sample_new_edges(to_sample);
}

void cluster::DynamicCPS::sample_new_edges(const std::vector<StagInt> &dp_ids) {
  std::vector<stag::DataPoint> qps;
  std::vector<StagInt> num_samples;
  for (auto q_id : dp_ids) {
    qps.push_back(data.at(q_id));
    num_samples.push_back(edges_per_node);
  }

  std::vector<EdgeTriplet> sampled_edges = tree_root.sample_neighbors(
      qps, dp_ids, num_samples);

  for (EdgeTriplet& edge : sampled_edges) {
    edge = EdgeTriplet(edge.row(), edge.col(),
                       tree_root.estimate_weight(data.at(edge.row()), edge.row()) / (StagReal) edges_per_node);

    if (!graph_edges.contains(edge.row())) {
      graph_edges[edge.row()] = {};
    }
    graph_edges[edge.row()].push_back(edge);
  }
}

stag::Graph cluster::DynamicCPS::get_graph() {
  std::vector<EdgeTriplet> all_triplets;
  for (auto i = 0; i < n_nodes; i++) {
    for (EdgeTriplet& edge : graph_edges[i]) {
      all_triplets.push_back(edge);
      all_triplets.emplace_back(edge.col(), edge.row(), edge.value());
    }
  }
  SprsMat adj_mat(n_nodes, n_nodes);
  adj_mat.setFromTriplets(all_triplets.begin(), all_triplets.end());
  return stag::Graph(adj_mat);
}

std::string cluster::DynamicCPS::algorithm_name() {
  return "DynamicCPS";
}
