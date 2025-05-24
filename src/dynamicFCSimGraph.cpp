#include <random>

#include <random.h>
#include <kde.h>
#include <cluster.h>
#include <iostream>

#include "multithreading/ctpl_stl.h"
#include "kde.h"
#include "cluster.h"

cluster::DynamicFCSimGraph::DynamicFCSimGraph(StagReal a_input,
                                              StagReal trunc) {
  a = a_input;
  n_nodes = 0;
  thresh = trunc;
}

void cluster::DynamicFCSimGraph::add_data(const std::vector<stag::DataPoint> &dps) {
  StagInt num_threads = std::thread::hardware_concurrency();
  std::mutex danger_mutex;

  if (dps.size() < num_threads) {
    for (auto j = 0; j < dps.size(); j++) {
      // Add new edges for existing data
      for (auto i = 0; i < n_nodes; i++) {
        StagReal weight = stag::gaussian_kernel(a, data.at(i), dps.at(j));
        if (weight > thresh) {
          graph_edges.emplace_back(i, n_nodes + j, weight);
          graph_edges.emplace_back(n_nodes + j, i, weight);
        }
      }

      // Add edges for new data
      for (auto i = 0; i < dps.size(); i++) {
        if (i != j) {
          StagReal weight = stag::gaussian_kernel(a, dps.at(i), dps.at(j));
          if (weight > thresh) {
            graph_edges.emplace_back(n_nodes + i, n_nodes + j, weight);
            graph_edges.emplace_back(n_nodes + j, n_nodes + i, weight);
          }
        }
      }
    }
  } else {
    ctpl::thread_pool pool((int) num_threads);

    StagInt chunk_size = floor((StagReal) dps.size() / (StagReal) num_threads);
    std::vector<std::future<void>> futures;
    for (auto chunk_id = 0; chunk_id < num_threads; chunk_id++) {
      futures.push_back(
          pool.push(
              [&, chunk_size, chunk_id, num_threads] (int id) {
                assert(chunk_id < num_threads);
                StagInt this_chunk_start = chunk_id * chunk_size;
                StagInt this_chunk_end = this_chunk_start + chunk_size;
                if (chunk_id == num_threads - 1) {
                  this_chunk_end = dps.size();
                }

                assert(this_chunk_end >= this_chunk_start);

                for (auto j = this_chunk_start; j < this_chunk_end; j++) {
                  // Add new edges for existing data
                  for (auto i = 0; i < n_nodes; i++) {
                    StagReal weight = stag::gaussian_kernel(a, data.at(i), dps.at(j));
                    if (weight > thresh) {
                      danger_mutex.lock();
                      graph_edges.emplace_back(i, n_nodes + j, weight);
                      graph_edges.emplace_back(n_nodes + j, i, weight);
                      danger_mutex.unlock();
                    }
                  }

                  // Add edges for new data
                  for (auto i = 0; i < dps.size(); i++) {
                    if (i != j) {
                      StagReal weight = stag::gaussian_kernel(a, dps.at(i), dps.at(j));
                      if (weight > thresh) {
                        danger_mutex.lock();
                        graph_edges.emplace_back(n_nodes + i, n_nodes + j, weight);
                        graph_edges.emplace_back(n_nodes + j, n_nodes + i, weight);
                        danger_mutex.unlock();
                      }
                    }
                  }
                }
              }
          )
      );
    }

  }

  // Update the data vectors
  data.insert(data.end(), dps.begin(), dps.end());
  n_nodes += dps.size();
}

stag::Graph cluster::DynamicFCSimGraph::get_graph() {
  SprsMat adj_mat(n_nodes, n_nodes);
  adj_mat.setFromTriplets(graph_edges.begin(), graph_edges.end());
  return stag::Graph(adj_mat);
}

std::string cluster::DynamicFCSimGraph::algorithm_name() {
  return "DynamicFCSimGraph";
}
