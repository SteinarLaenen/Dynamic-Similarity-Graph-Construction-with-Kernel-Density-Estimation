#ifndef DYNAMIC_KDE_CLUSTERING_CLUSTER_H
#define DYNAMIC_KDE_CLUSTERING_CLUSTER_H

#include <random>
#include <set>
#include <graph.h>
#include <data.h>

namespace cluster {

  /**
   * An abstract dynamic similarity graph.
   *
   * Provides methods for adding data points to the data set, and a method
   * to retrieve a pointer to the maintained similarity graph.
   */
  class AbstractDynamicSimilarityGraph {
  public:
    /**
     * Add a set of data points to the data set, and update the underlying
     * graph.
     *
     * @param dps
     */
    virtual void add_data(const std::vector<stag::DataPoint>& dps) = 0;

    /*
     * Get a pointer to the current similarity graph.
     */
    virtual stag::Graph get_graph() = 0;

    /**
     * Return the name of this algorithm.
     */
    virtual std::string algorithm_name() = 0;
  };

  class DynamicKDETreeEntry {
  public:
    DynamicKDETreeEntry(StagReal a_input, StagInt d_input, StagInt this_depth,
                        StagInt max_depth);

    void possibly_add_query_point(const stag::DataPoint& q, StagInt global_id);

    void possibly_add_query_points(
        const std::vector<stag::DataPoint>& qs,
        const std::vector<StagInt>& global_id);

    std::unordered_set<StagInt> add_data(const std::vector<stag::DataPoint>& data,
                                         StagInt base_global_id);

    void remove_q_ids(const std::vector<StagInt>& q_ids);

    StagReal estimate_weight(const stag::DataPoint& q, StagInt q_id);

    std::vector<StagReal> estimate_weights(const std::vector<stag::DataPoint>& qs,
                                           const std::vector<StagInt>& q_ids);

    std::vector<EdgeTriplet> sample_neighbors(
        const std::vector<stag::DataPoint>& qs,
        const std::vector<StagInt>& q_ids,
        const std::vector<StagInt>& nums_to_sample);

    ~DynamicKDETreeEntry();

  private:
    // Key info about the kernel function
    StagReal a;
    StagInt d;
    StagInt this_depth;
    StagInt max_depth;

    // Whether the number of data points at this node is currently below the cutoff
    // for using the exact estimator.
    //
    // Once we are no longer below the cutoff, we add a left child followed by
    // a right child.
    bool below_cutoff;
    kde::DynamicKDEWithResizing this_estimator;
    kde::DynamicExactKDE exact_kde;

    // The total number of data points currently at this node.
    StagInt n_node;
    StagInt n_left;
    StagInt n_right;

    // Children, if they exist
    DynamicKDETreeEntry* left_child;
    DynamicKDETreeEntry* right_child;
    std::uniform_real_distribution<double> sampling_dist;

    // Keep track of the edges which are sampled 'through' this node.
    // The global ids of the nodes whose edges are sampled here
    std::unordered_set<StagInt> edge_source_nodes;

    // The first ever estimate for each query point at this node.
    // Once our estimate changes enough, we will force a re-sample.
    std::unordered_map<StagInt, StagReal> last_estimates;

    // Store a map of query_ids from the estimator to global query ids
    std::unordered_map<StagInt, StagInt> local_to_global_q_id;
    std::unordered_map<StagInt, StagInt> global_to_local_q_id;

    // Store a map of local to global data ids
    std::unordered_map<StagInt, StagInt> local_to_global_d_id;
  };


  /**
 * A dynamic approximate similarity graph data structure as described in our
 * paper.
 */
  class DynamicCPS : public AbstractDynamicSimilarityGraph {
  public:
    /**
     * Constructor: take the Gaussian kernel parameter.
     * @param a
     */
    DynamicCPS(StagReal a, StagInt d_input, StagInt max_depth);

    void add_data(const std::vector<stag::DataPoint>& dps);

    stag::Graph get_graph();

    std::string algorithm_name();

  private:
    void sample_new_edges(const std::vector<StagInt>& dp_ids);

    // Keep track of the basic features of the similarity graph
    StagReal a;
    StagInt d;

    // Keep track of the data currently stored in the similarity graph
    StagInt n_nodes;
    std::vector<stag::DataPoint> data;

    // Track the edges added for each node
    std::unordered_map<StagInt, std::vector<EdgeTriplet>> graph_edges;

    // Keep track of the number of edges per node to sample
    StagInt n_nodes_at_last_edges_update;
    StagInt edges_per_node;

    DynamicKDETreeEntry tree_root;
  };

  /**
   * A dynamic fully connected similarity graph.
   */
  class DynamicFCSimGraph : public AbstractDynamicSimilarityGraph {
  public:
    /**
     * Constructor: take the Gaussian kernel parameter.
     */
    DynamicFCSimGraph(StagReal a, StagReal trunc);

    void add_data(const std::vector<stag::DataPoint>& dps);

    stag::Graph get_graph();

    std::string algorithm_name();

  private:
    // Keep track of the data currently stored in the similarity graph
    StagReal a;
    StagInt n_nodes;
    std::vector<stag::DataPoint> data;
    StagReal thresh;

    // Track the edges added for each node
    std::vector<EdgeTriplet> graph_edges;
  };

  /**
   * A naive dynamic CPS similarity graph.
   */
  class NaiveDynamicCPS : public AbstractDynamicSimilarityGraph {
  public:
    /**
     * Constructor: take the Gaussian kernel parameter.
     */
    NaiveDynamicCPS(StagReal a, StagInt d);

    void add_data(const std::vector<stag::DataPoint>& dps);

    stag::Graph get_graph();

    std::string algorithm_name();

  private:
    // Keep track of the data currently stored in the similarity graph
    StagReal a;
    StagInt d;
    StagInt n_nodes;

    DenseMat data;

    // Store the CPS
    stag::Graph cps;
  };

  /**
   * A dynamic k-NN similarity graph.
   */
  class DynamickNNSimGraph : public AbstractDynamicSimilarityGraph {
  public:
    /**
     * Constructor: take the Gaussian kernel parameter.
     */
    DynamickNNSimGraph(StagInt k);

    void add_data(const std::vector<stag::DataPoint>& dps);

    stag::Graph get_graph();

    std::string algorithm_name();

  private:
    void possibly_replace_nn(StagInt this_node, StagInt other_node,
                             std::mutex& danger_mutex);

    // Keep track of the data currently stored in the similarity graph
    StagInt k;
    StagInt n_nodes;
    std::vector<stag::DataPoint> data;

    // Keep track of the distances to all nearest neighbors
    std::unordered_map<StagInt, std::multiset<StagReal>> nearest_distances;
    std::unordered_map<StagInt, std::unordered_set<StagInt>> nearest_neighbors;
  };

}

#endif //DYNAMIC_KDE_CLUSTERING_CLUSTER_H
