/**
 * @file kde.h
 * \brief Methods for computing approximate kernel density estimation.
 *
 * Given some *kernel function*
 * \f$k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}\f$,
 * a set of *data points* \f$x_1, \ldots, x_n \in \mathbb{R}^d\f$,
 * and a *full_query point* \f$q \in \mathbb{R}^d\f$,
 * the *kernel density* of \f$q\f$ is given by
 *
 * \f[
 *    K(q) = \frac{1}{n} \sum_{i = 1}^n k(q, x_i).
 * \f]
 *
 * A common kernel function is the Gaussian kernel function
 * \f[
 *    k(u, v) = \exp\left(- a \|u - v\|_2^2\right),
 * \f]
 * where \f$a \geq 0\f$ is a parameter controlling the 'bandwidth' of the
 * kernel.
 *
 * Computing the kernel density for a full_query point exactly requires computing the
 * distance from the full_query point to every data point, requiring
 * \f$\Omega(n d)\f$ time.
 * This motivates the study of *kernel density estimation*, in which the goal
 * is to estimate the kernel density within some error tolerance, in faster
 * time than computing it exactly.
 * Specifically, given some error parameter \f$\epsilon\f$, a kernel density
 * estimation algorithm will return \f$\mathrm{KDE}(q)\f$ for some full_query point
 * \f$q\f$ such that
 * \f[
 *  (1 - \epsilon) K(q) \leq \mathrm{KDE}(q) \leq (1 + \epsilon) K(q).
 * \f]
 *
 * This module provides the stag::CKNSGaussianKDE data structure which takes
 * \f$O(\epsilon^{-1} n^{1.25})\f$ time for initialisation, and can then provide
 * KDE estimates in time \f$O(\epsilon^{-2} n^{0.25})\f$ for each full_query.
 */
#ifndef KDE_H
#define KDE_H

#include <mutex>

#include <definitions.h>

#include "lsh.h"

namespace kde {

  StagReal unnormalised_kde(StagReal a,
                            const std::vector<stag::DataPoint>& data,
                            const stag::DataPoint& query);

  /**
   * Compute the median value of a vector.
   */
  StagReal median(std::vector<StagReal> &v);

  /**
   * Compute the squared distance between the given two data points.
   *
   * @param u the first data point
   * @param v the second data point
   * @return the squared distance between \f$u\f$ and \f$v\f$.
   */
  StagReal squared_distance(const stag::DataPoint& u, const stag::DataPoint& v);

  /**
   * Compute the squared distance between the given two data points, if it is at
   * most max_dist. Otherwise, return -1.
   *
   * @param u the first data point
   * @param v the second data point
   * @param max_dist the maximum dist allowed.
   * @return the squared distance between \f$u\f$ and \f$v\f$.
   */
  StagReal squared_distance_at_most(const stag::DataPoint& u,
                                    const stag::DataPoint& v,
                                    StagReal max_dist);

  /**
   * Compute the sampling probability for the CKNS algorithm, given a value of
   * j and log2(n * mu).
   *
   * @param j
   * @param log_nmu the value of log2(n * mu)
   * @return
   */
  StagReal ckns_p_sampling(StagInt j, StagInt log_nmu, StagInt n,
                           StagInt sampling_offset);

  StagReal ckns_gaussian_rj_squared(StagInt j, StagReal a);

  /**
   * Create the LSH parameters for a particular application of E2LSH as part of
   * the CKNS algorithm with the Gaussian kernel.
   *
   * @param J the total number of sampling levels
   * @param j the current sampling level
   * @param n the total number of data points
   * @param d the dimension of the data
   * @param a the scaling parameter of the Gaussian kernel
   * @return the K and L parameters for this E2LSH table.
   */
  std::vector<StagUInt> ckns_gaussian_create_lsh_params(
      StagInt J, StagInt j, StagReal a, StagReal K2_constant);

  /**
   * Compute the value of J for the CKNS algorithm, given
   *   - n, the number of data points
   *   - log2(n * mu), as an integer (at most log2(n)).
   *
   * @param n the number of data points
   * @param log_nmu the value of log2(n * mu)
   * @return
   */
  StagInt ckns_J(StagInt n, StagInt log_nmu);

  /**
   * Create the abstract base class representing a dynamic KDE algorithm.
   */
   class AbstractDynamicKDE {
   public:
     /**
      * Add query data points and return their IDs.
      */
     virtual std::vector<StagInt> add_query(
         const std::vector<stag::DataPoint>& q) = 0;
     virtual StagInt add_query(const stag::DataPoint& q);

     /**
      * Add data points and return the IDs of updated query points.
      * @param dps
      * @return
      */
     virtual std::vector<StagInt> add_data(const std::vector<stag::DataPoint>& dps) = 0;

     /**
      * Return the current estimate of a given query point id.
      */
     virtual StagReal get_estimate(StagInt qp_id) = 0;

     /**
      * Return the name of this algorithm.
      */
      virtual std::string algorithm_name() = 0;
   };


  /**
   * \cond
   * A helper class for the CKNSGaussianKDE data structure. Undocumented.
   */

  // The hash unit will store a pointer to the Dynamic KDE.
  class DynamicKDE;

  class DynamicKDEHashUnit {
  public:
    DynamicKDEHashUnit(StagReal a, StagInt log_nmu,
                       StagInt j, StagReal K2_constant, StagInt prob_offset,
                       StagInt n_input, StagInt d, StagInt log_nmu_iter, StagInt iter,
                       DynamicKDE* parent);

    /**
     * Return the IDs of queries needing updated.
     */
    std::unordered_set<StagInt> add_data(const std::vector<stag::DataPoint>& data,
                                         std::vector<std::mutex>& hum);

    /**
     * Return the IDs of queries needing updated.
     */
    std::unordered_set<StagInt> add_data(const stag::DataPoint &dp,
                                         std::vector<std::mutex>& hum);

    StagReal query(const stag::DataPoint &q, StagInt q_id);

  private:
    StagReal query_neighbors(const stag::DataPoint &q,
                             const std::vector<stag::DataPoint> &neighbors) const;

    std::unordered_set<StagInt> add_sampled_dp(const stag::DataPoint &dp,
                                               std::vector<std::mutex>& hum);

    /**
     * Update the estimate of the q_id point based on the addition of some new
     * point to this hash unit. Return a boolean indicating whether the estimate
     * has been updated.
     */
    bool possibly_update_estimate(StagInt q_id, StagReal dist_to_new_point,
                                  std::vector<std::mutex>& hum);

    bool below_cutoff;
    bool final_shell;
    lsh::LSHWithFullHash LSH_buckets;
    StagReal p_sampling;
    StagInt j;
    StagInt J;
    StagInt log_nmu;
    StagInt log_nmu_iter;
    StagInt iter;
    StagReal a;
    StagInt sampling_offset;
    StagInt n;
    StagInt num_sampled_points;

    DynamicKDE* parentKDE;

    // Store the distances for the shell stored controlled by this hash unit
    StagReal shell_min_dist;
    StagReal shell_max_dist;

    // Used only if the number of data points is below the cutoff.
    std::vector<stag::DataPoint> all_data;
    std::unordered_set<StagInt> all_q_ids;
  };
  /**
   * \endcond
   */

  /**
   * \brief A dynamic Gaussisn KDE data structure.
   *
   * This data structure implements the new dynamic KDE data structure described
   * in the paper.
   */
  class DynamicKDE : public AbstractDynamicKDE{
  public:
    /**
     * The default constructor.
     */
    DynamicKDE() {}

    /**
     * The most basic constructor: initialise an empty data structure with the
     * specified Gaussian kernel parameter.
     *
     * @param a the Gaussian kernel bandwidth parameter.
     * @param n the maximum number of data points which can be stored in this
     *          data structure. Once this number is exceeded, the calling code
     *          should reconstruct this data structure with maximum points 2*n.
     * @param d the number of dimensions in the dataset.
     */
    DynamicKDE(StagReal a, StagInt n, StagInt d);

    /**
     * Initialise a new KDE data structure with eps and mu parameters.
     */
    DynamicKDE(StagReal a, StagReal eps, StagInt n, StagInt d);

    /**
     * Initialise a new KDE data structure with full parameter control.
     */
    DynamicKDE(StagReal a, StagInt K1,
               StagReal K2_constant, StagInt sampling_offset, StagInt n,
               StagInt d);

    /**
     * Add the data points in the given matrix.
     *
     * Returns a vector with the full_query points whose full_query estimate has changed.
     */
    std::vector<StagInt> add_data(const std::vector<stag::DataPoint>& dps);

    /**
     * Add several full_query points.
     *
     * Each full_query point is given an ID which is stored in the order in which
     * full_query points are added.
     *
     * Returns a vector containing the IDs of the added points.
     */
    std::vector<StagInt> add_query(
        const std::vector<stag::DataPoint>& q);

    /**
     * Get the KDE estimate for a full_query point.
     *
     * Running time \f$O(1)\f$. Query point is specified by its id.
     *
     * @param qp_id the ID of the full_query point to retrieve.
     * @return the KDE estimate.
     */
    StagReal get_estimate(StagInt qp_id);

    std::string algorithm_name();

    std::vector<stag::DataPoint>* get_query_dps();
    std::vector<StagInt>* get_qp_mus();
    std::vector<std::vector<StagReal>>* get_iter_estimates();

    StagInt num_data_points;
    StagInt num_query_points;

  private:
    void initialize(StagReal a, StagInt K1,
                    StagReal K2_constant, StagInt prob_offset, StagInt n,
                    StagInt d);

    StagInt add_hash_unit(StagInt log_nmu_iter,
                          StagInt log_nmu,
                          StagInt iter,
                          StagInt j,
                          std::mutex &units_mutex);

    /**
     * Perform a full update of the estimate of the provided query points.
     * (i.e. re-query from scratch).
     * @param q_ids
     */
    void update_estimates(const std::vector<StagInt>& q_ids);

    /**
     * Performing a full query for a query point resets all of the stored
     * information about the query point.
     */
    void full_query(const std::vector<stag::DataPoint>& query, StagInt base_q_id);
    void full_query(const std::vector<stag::DataPoint>& query,
                    const std::vector<StagInt>& q_ids);
    void full_query(const stag::DataPoint &query, StagInt q_id);
    void chunk_query(const std::vector<stag::DataPoint>& query,
                     StagInt chunk_start,
                     StagInt chunk_end,
                     const std::vector<StagInt>& q_ids,
                     std::vector<std::mutex>& hum);

    std::vector<std::vector<std::vector<DynamicKDEHashUnit>>> hash_units;
    StagInt max_log_nmu;
    StagInt min_log_nmu;
    StagInt num_log_nmu_iterations;
    StagInt sampling_offset;
    StagInt max_n;
    StagInt d;
    StagReal a;
    StagInt k1;
    StagReal k2_constant;

    //-------------
    // Store the KDE estimates of all query points added to the data structure.
    //-------------
    // Which log_nmu_iteration the current estimate for each query point comes from.
    std::vector<StagInt> qp_mus;

    // The estimates from each iteration of the data structure for each qp.
    std::vector<std::vector<StagReal>> all_iteration_estimates;

    // The actual estimates for each query point.
    std::vector<StagReal> estimates;

    // Keep track of the value of the estimates at the last time there was
    // a full update
    std::vector<StagReal> full_update_estimates;

    // The query points themselves.
    std::vector<stag::DataPoint> query_points;
  };

  class DynamicKDEWithResizing : public AbstractDynamicKDE {
  public:
    DynamicKDEWithResizing(StagReal a, StagInt d);
    std::vector<StagInt> add_data(const std::vector<stag::DataPoint>& dps);
    std::vector<StagInt> add_query(
        const std::vector<stag::DataPoint>& q);
    StagReal get_estimate(StagInt qp_id);
    std::string algorithm_name();

    std::vector<stag::DataPoint> data_points;

  private:
    StagReal a;
    StagInt d;
    StagInt current_max_n;
    StagInt num_data_points;

    DynamicKDE* internal_kde;

    std::vector<stag::DataPoint> query_points;
  };

  class NaiveDynamicKDEHashUnit {
  public:
    NaiveDynamicKDEHashUnit(StagReal a, StagInt log_nmu,
                            StagInt j, StagReal K2_constant, StagInt prob_offset,
                            StagInt n_input, StagInt d, StagInt log_nmu_iter,
                            StagInt iter);

    /**
     * Add data to the hash unit.
     */
    void add_data(const std::vector<stag::DataPoint>& data);
    void add_data(const stag::DataPoint &dp);

    StagReal query(const stag::DataPoint &q);

  private:
    StagReal query_neighbors(const stag::DataPoint &q,
                             const std::vector<stag::DataPoint> &neighbors) const;

    void add_sampled_dp(const stag::DataPoint &dp);

    bool below_cutoff;
    bool final_shell;
    lsh::LSHDynamic LSH_buckets;
    StagReal p_sampling;
    StagInt j;
    StagInt J;
    StagInt log_nmu;
    StagInt log_nmu_iter;
    StagInt iter;
    StagReal a;
    StagInt sampling_offset;
    StagInt n;
    StagInt num_sampled_points;

    // Store the distances for the shell stored controlled by this hash unit
    StagReal shell_min_dist;
    StagReal shell_max_dist;

    // Used only if the number of data points is below the cutoff.
    std::vector<stag::DataPoint> all_data;
  };

  /**
   * \brief A naive dynamic Gaussisn KDE data structure.
   *
   * Allow adding of new data points. Querying is still naive - we recompute
   * every time.
   */
  class NaiveDynamicCKNS : public AbstractDynamicKDE {
  public:
    /**
     * The default constructor.
     */
    NaiveDynamicCKNS() {}

    /**
     * The most basic constructor: initialise an empty data structure with the
     * specified Gaussian kernel parameter.
     *
     * @param a the Gaussian kernel bandwidth parameter.
     * @param n the maximum number of data points which can be stored in this
     *          data structure. Once this number is exceeded, the calling code
     *          should reconstruct this data structure with maximum points 2*n.
     * @param d the number of dimensions in the dataset.
     */
    NaiveDynamicCKNS(StagReal a, StagInt n, StagInt d);

    /**
     * Initialise a new KDE data structure with eps and mu parameters.
     */
    NaiveDynamicCKNS(StagReal a, StagReal eps, StagInt n, StagInt d);

    /**
     * Initialise a new KDE data structure with full parameter control.
     */
    NaiveDynamicCKNS(StagReal a, StagInt K1,
                     StagReal K2_constant, StagInt sampling_offset, StagInt n,
                     StagInt d);

    /**
     * Add a vector of new data points to the data structure.
     */
    std::vector<StagInt> add_data(const std::vector<stag::DataPoint>& dps);

    /**
     * Add several full_query points.
     *
     * Each full_query point is given an ID which is stored in the order in which
     * full_query points are added.
     *
     * Returns a vector containing the IDs of the added points.
     */
    std::vector<StagInt> add_query(
        const std::vector<stag::DataPoint>& q);

    /**
     * Get the KDE estimate for a query point.
     *
     * @param qp_id the ID of the full_query point to retrieve.
     * @return the KDE estimate.
     */
    StagReal get_estimate(StagInt qp_id);

    std::string algorithm_name();

    StagInt num_data_points;
    StagInt num_query_points;

  private:
    void initialize(StagReal a, StagInt K1,
                    StagReal K2_constant, StagInt prob_offset, StagInt n,
                    StagInt d);

    StagInt add_hash_unit(StagInt log_nmu_iter,
                          StagInt log_nmu,
                          StagInt iter,
                          StagInt j,
                          std::mutex &units_mutex);

    /**
     * Performing a full query for a query point resets all of the stored
     * information about the query point.
     */
    void full_query(const std::vector<stag::DataPoint>& query, StagInt base_q_id);
    void full_query(const std::vector<stag::DataPoint>& query,
                    const std::vector<StagInt>& q_ids);
    void full_query(const stag::DataPoint &query, StagInt q_id);
    void chunk_query(const std::vector<stag::DataPoint>& query,
                     StagInt chunk_start,
                     StagInt chunk_end,
                     const std::vector<StagInt>& q_ids,
                     std::vector<std::mutex>& hum);

    void update_estimates();

    std::vector<std::vector<std::vector<NaiveDynamicKDEHashUnit>>> hash_units;
    StagInt max_log_nmu;
    StagInt min_log_nmu;
    StagInt num_log_nmu_iterations;
    StagInt sampling_offset;
    StagInt max_n;
    StagInt d;
    StagReal a;
    StagInt k1;
    StagReal k2_constant;

    //-------------
    // Store the KDE estimates of all query points added to the data structure.
    //-------------
    // The actual estimates for each query point.
    std::vector<StagReal> estimates;

    // The query points themselves.
    std::vector<StagInt> q_ids;
    std::vector<stag::DataPoint> query_points;
  };

  class NaiveDynamicCKNSWithResizing: public AbstractDynamicKDE {
  public:
    NaiveDynamicCKNSWithResizing(StagReal a, StagInt d);
    std::vector<StagInt> add_data(const std::vector<stag::DataPoint> &dps);
    std::vector<StagInt> add_query(
        const std::vector<stag::DataPoint> &q);
    StagReal get_estimate(StagInt qp_id);
    std::string algorithm_name();

  private:
    StagReal a;
    StagInt d;
    StagInt current_max_n;

    NaiveDynamicCKNS* internal_kde;

    std::vector<stag::DataPoint> data_points;
    std::vector<stag::DataPoint> query_points;
  };

  class DynamicExactKDE : public AbstractDynamicKDE {
  public:
    /**
     * The default constructor.
     */
    DynamicExactKDE() {}

    /**
     * The most basic constructor: initialise an empty data structure with the
     * specified Gaussian kernel parameter.
     *
     * @param a the Gaussian kernel bandwidth parameter.
     */
    DynamicExactKDE(StagReal a);

    /**
     * Add a vector of new data points to the data structure.
     */
    std::vector<StagInt> add_data(const std::vector<stag::DataPoint>& dps);

    /**
     * Add several query points.
     *
     * Each query point is given an ID which is stored in the order in which
     * query points are added.
     *
     * Returns a vector containing the IDs of the added points.
     */
    std::vector<StagInt> add_query(
        const std::vector<stag::DataPoint>& q);

    /**
     * Get the KDE estimate for a query point.
     *
     * @param qp_id the ID of the query point to retrieve.
     * @return the KDE estimate.
     */
    StagReal get_estimate(StagInt qp_id);

    /**
     * Sample a neighbor as part of an approximate similarity graph construction.
     * Returns indices of the data points stored at this node.
     */
     std::vector<StagInt> sample_neighbors(StagInt q_id, std::vector<StagReal> rs);

    std::string algorithm_name();

    // The data points which have been hashed
    StagInt num_data_points;
    std::vector<stag::DataPoint> data_points;

  private:
    StagReal a;


    // The actual estimates for each query point.
    std::vector<StagReal> estimates;

    // The query points themselves.
    std::vector<StagInt> q_ids;
    std::vector<stag::DataPoint> query_points;
  };

  class DynamicRandomSamplingKDE : public AbstractDynamicKDE {
  public:
    /**
     * The default constructor.
     */
    DynamicRandomSamplingKDE() {}

    /**
     * The most basic constructor: initialise an empty data structure with the
     * specified Gaussian kernel parameter.
     *
     * @param a the Gaussian kernel bandwidth parameter.
     * @param p the random sampling probability
     */
    DynamicRandomSamplingKDE(StagReal a, StagReal p);

    /**
     * Add a vector of new data points to the data structure.
     */
    std::vector<StagInt> add_data(const std::vector<stag::DataPoint>& dps);

    /**
     * Add several query points.
     *
     * Each query point is given an ID which is stored in the order in which
     * query points are added.
     *
     * Returns a vector containing the IDs of the added points.
     */
    std::vector<StagInt> add_query(
        const std::vector<stag::DataPoint>& q);

    /**
     * Get the KDE estimate for a query point.
     *
     * @param qp_id the ID of the query point to retrieve.
     * @return the KDE estimate.
     */
    StagReal get_estimate(StagInt qp_id);

    std::string algorithm_name();

  private:
    StagReal a;
    StagReal p;

    // The data points which have been sampled
    StagInt num_sampled_points;
    std::vector<stag::DataPoint> data_points;

    // The actual estimates for each query point.
    std::vector<StagReal> estimates;

    // The query points themselves.
    std::vector<StagInt> q_ids;
    std::vector<stag::DataPoint> query_points;
  };
}

#endif //KDE_H
