#include <random>

#include <random.h>
#include <cluster.h>
#include <kde.h>
#include <iostream>

#include "kde.h"
#include "cluster.h"

cluster::NaiveDynamicCPS::NaiveDynamicCPS(StagReal a_input, StagInt d_input)
  : cps(stag::complete_graph(10))
{
  a = a_input;
  d = d_input;
  n_nodes = 0;
}

void cluster::NaiveDynamicCPS::add_data(const std::vector<stag::DataPoint> &dps) {
  // Update the data vectors
  data.conservativeResize(n_nodes + dps.size(), n_nodes + dps.size());
  for (auto i = 0; i < dps.size(); i++) {
    assert(dps.at(i).dimension == d);
    for (auto j = 0; j < d; j++) {
      data.coeffRef(n_nodes + i, j) = dps.at(i).coordinates[j];
    }
  }

  cps = stag::approximate_similarity_graph(&data, a);

  n_nodes += dps.size();
}

stag::Graph cluster::NaiveDynamicCPS::get_graph() {
  return cps;
}

std::string cluster::NaiveDynamicCPS::algorithm_name() {
  return "NaiveDynamicCPS";
}
