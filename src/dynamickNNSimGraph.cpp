#include <random>

#include <random.h>
#include <kde.h>
#include <cluster.h>
#include <iostream>

#include "multithreading/ctpl_stl.h"
#include "kde.h"
#include "cluster.h"

cluster::DynamickNNSimGraph::DynamickNNSimGraph(StagInt k_input) {
  k = k_input;
  n_nodes = 0;
}

void cluster::DynamickNNSimGraph::possibly_replace_nn(StagInt this_node,
                                                      StagInt other_node,
                                                      std::mutex& danger_mutex) {
  StagReal distance = kde::squared_distance(data.at(this_node),
                                            data.at(other_node));

  danger_mutex.lock();
  if (!nearest_distances.contains(this_node)) {
    nearest_distances[this_node] = {};
  }

  if (nearest_distances[this_node].size() < k) {
    nearest_distances[this_node].insert(distance);
    nearest_neighbors[this_node].insert(other_node);
  } else {
    auto max_elem = std::max_element(nearest_distances[this_node].begin(),
                                     nearest_distances[this_node].end());
    if (distance < *max_elem) {
      nearest_distances[this_node].erase(max_elem);
      nearest_distances[this_node].insert(distance);
      nearest_neighbors[this_node].insert(other_node);
      StagInt max_neighbor;
      for (auto neighbour : nearest_neighbors[this_node]) {
        if (kde::squared_distance(data.at(this_node), data.at(neighbour)) >= *max_elem) {
          max_neighbor = neighbour;
        }
      }
      nearest_neighbors[this_node].erase(max_neighbor);
    }
  }
  danger_mutex.unlock();
}

void cluster::DynamickNNSimGraph::add_data(const std::vector<stag::DataPoint> &dps) {
  // Add the new data
  data.insert(data.end(), dps.begin(), dps.end());

  StagInt num_threads = std::thread::hardware_concurrency();
  std::mutex danger_mutex;

  // Add new edges
  if (n_nodes < num_threads) {
    for (auto i = 0; i < n_nodes + dps.size(); i++) {
      for (auto j = MAX(n_nodes, i); j < n_nodes + dps.size(); j++) {
        possibly_replace_nn(i, j, danger_mutex);
        possibly_replace_nn(j, i, danger_mutex);
      }
    }
  } else {
    ctpl::thread_pool pool((int) num_threads);

    StagInt chunk_size = floor((StagReal) (n_nodes + dps.size()) /
                               (StagReal) num_threads);
    std::vector<std::future<void>> futures;
    for (auto chunk_id = 0; chunk_id < num_threads; chunk_id++) {
      futures.push_back(
          pool.push(
              [&, chunk_size, chunk_id, num_threads] (int id) {
                assert(chunk_id < num_threads);
                StagInt this_chunk_start = chunk_id * chunk_size;
                StagInt this_chunk_end = this_chunk_start + chunk_size;
                if (chunk_id == num_threads - 1) {
                  this_chunk_end = n_nodes + dps.size();
                }

                assert(this_chunk_end >= this_chunk_start);

                for (auto i = this_chunk_start; i < this_chunk_end; i++) {
                  for (auto j = MAX(n_nodes, i); j < n_nodes + dps.size(); j++) {
                    possibly_replace_nn(i, j, danger_mutex);
                    possibly_replace_nn(j, i, danger_mutex);
                  }
                }
              }
          )
      );
    }
  }

  // Update the data vectors
  n_nodes += dps.size();
}

stag::Graph cluster::DynamickNNSimGraph::get_graph() {
  SprsMat adj_mat(n_nodes, n_nodes);
  std::vector<EdgeTriplet> graph_edges;
  for (auto i = 0; i < n_nodes; i++) {
    for (auto j : nearest_neighbors[i]) {
      graph_edges.emplace_back(i, j, 1);
      graph_edges.emplace_back(j, i, 1);
    }
  }
  adj_mat.setFromTriplets(graph_edges.begin(), graph_edges.end());
  return stag::Graph(adj_mat);
}

std::string cluster::DynamickNNSimGraph::algorithm_name() {
  return "DynamickNNSimGraph";
}
