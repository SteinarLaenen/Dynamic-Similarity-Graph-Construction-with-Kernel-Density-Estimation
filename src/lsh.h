#ifndef LSH_H
#define LSH_H

#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <definitions.h>
#include <data.h>

// The value for algorithm parameter W.
#define LSH_PARAMETER_W 4.0

namespace lsh {

  /**
   * A Euclidean locality-sensitive hash function.
   */
  class LSHFunction {
  public:
    /**
     * Initialise a random LSH function with the given dimension.
     *
     * @param dimension the dimensionality of the data
     */
    explicit LSHFunction(StagUInt dimension);

    /**
     * Apply this hash function to the given data point.
     *
     * @param point a data point to be hashed
     * @return an integer indicating which hash bucket this data point
     *         was hashed to
     */
    StagInt apply(const stag::DataPoint& point);

    /**
     * For two points at a given distance \f$c\f$, compute the probability that
     * they will collide in a random Euclidean LSH function.
     *
     * @param distance the distance \f$c\f$.
     * @return the collision probability of two points at distance \f$c\f$.
     */
    static StagReal collision_probability(StagReal distance);

  private:
    std::vector<StagReal> a;
    StagReal b;
    StagUInt dim;
  };

  /**
   * \cond
   */
  class MultiLSHFunction {
  public:
    MultiLSHFunction(StagInt dimension, StagInt num_functions);
    StagInt apply(const stag::DataPoint& point);
  private:
    StagInt L;
    DenseMat rand_proj;
    Eigen::VectorXd rand_offset;
    Eigen::Matrix<StagInt, Eigen::Dynamic, 1> uhash_vector;
  };
  /**
   * \endcond
   */

  /**
   * \brief A Euclidean locality sensitive hash table.
   *
   * The E2LSH hash table is constructed with some set of data points, which are
   * hashed with several copies of the stag::LSHFunction.
   *
   * Then, for any full_query point, the data structure returns the points in the
   * original dataset which are close to the full_query point.
   * The probability that a given point \f$x\f$ in the data set is returned for
   * full_query \f$q\f$ is dependent on the distance between \f$q\f$ and \f$x\f$.
   *
   * The E2LSH hash table takes two parameters, K and L, which control the
   * probability that two points will collide in the hash table.
   * For full_query point \f$q\f$, a data point \f$x\f$ at distance \f$c \in \mathbb{R}\f$
   * from \f$q\f$ is returned with probability
   * \f[
   *    1 - (1 - p(c)^K)^L,
   * \f]
   * where \f$p(c)\f$ is the probability that a single stag::LSHFunction will
   * hash \f$q\f$ and \f$x\f$ to the same value.
   * This probability can be computed with the stag::E2LSH::collision_probability
   * method.
   *
   * Larger values of K and L will increase both the construction and full_query time
   * of the hash table.
   */
  class LSHWithFullHash {
  public:
    LSHWithFullHash() {};

    /**
     * Initialise the E2LSH hash table.
     *
     * @param K parameter K of the hash table
     * @param L parameter L of the hash table
     * @param dataSet a pointer to the dataSet to be hashed into the hash
     *                table. The actual data should be stored and controlled by
     *                the calling code, and this vector of data point pointers
     *                will be used by the LSH table.
     */
    LSHWithFullHash(StagUInt K,
                    StagUInt L,
                    StagInt d);

    /**
     * Add a single data point to the hash table.
     *
     * Return the indices of any full_query points collided with (and removed) from
     * the full hash.
     */
    std::unordered_set<StagInt> add_data(const stag::DataPoint& dp);

    /**
     * Query the LSH table to find the near neighbors of a given full_query point.
     *
     * Each point in the dataset will be returned with some probability
     * dependent on the distance to the full_query point and the parameters K and L.
     *
     * @param query the data point to be queried.
     * @param q_id the index of the full_query point to be stored in the full hash
     * @return the near neighbors of the full_query point
     */
    std::vector<stag::DataPoint> get_near_neighbors(const stag::DataPoint& query,
                                                    StagInt q_id);

  private:
    void initialise_hash_functions();

    StagInt compute_lsh(StagUInt gNumber, const stag::DataPoint& point);

    StagUInt dimension; // dimension of points.
    StagUInt parameterK; // parameter K of the algorithm.
    StagUInt parameterL; // parameter L of the algorithm.

    std::vector<StagInt> rnd_vec; // used for hashing vectors

    // The array of pointers to the points that are contained in the
    // structure. Some types of this structure (of UHashStructureT,
    // actually) use indices in this array to refer to points (as
    // opposed to using pointers).
    std::vector<stag::DataPoint> points;

    // This table stores the LSH functions. There are <nHFTuples> rows
    // of <hfTuplesLength> LSH functions.
    std::vector<MultiLSHFunction> lshFunctions;

    // The set of non-empty buckets
    std::vector<std::unordered_map<StagInt,std::vector<StagUInt>>> hashTables;

    // The 'FullHash' table which stores the query points
    std::vector<std::unordered_map<StagInt,std::unordered_set<StagInt>>> fullHash;
  };

  /**
   * A simple extension to the LSH implementation to allow adding of new
   * data points.
   */
  class LSHDynamic {
  public:
    LSHDynamic() {};

    /**
     * Initialise the E2LSH hash table.
     *
     * @param K parameter K of the hash table
     * @param L parameter L of the hash table
     * @param dataSet a pointer to the dataSet to be hashed into the hash
     *                table. The actual data should be stored and controlled by
     *                the calling code, and this vector of data point pointers
     *                will be used by the LSH table.
     */
    LSHDynamic(StagUInt K,
               StagUInt L,
               StagInt d);

    /**
     * Add a single data point to the hash table.
     *
     * Return the indices of any full_query points collided with (and removed) from
     * the full hash.
     */
    void add_data(const stag::DataPoint& dp);

    /**
     * Query the LSH table to find the near neighbors of a given query point.
     *
     * @param query the data point to be queried.
     * @return the near neighbors of the full_query point
     */
    std::vector<stag::DataPoint> get_near_neighbors(const stag::DataPoint& query);

  private:
    void initialise_hash_functions();

    StagInt compute_lsh(StagUInt gNumber, const stag::DataPoint& point);

    StagUInt dimension; // dimension of points.
    StagUInt parameterK; // parameter K of the algorithm.
    StagUInt parameterL; // parameter L of the algorithm.

    std::vector<StagInt> rnd_vec; // used for hashing vectors

    // The array of pointers to the points that are contained in the
    // structure. Some types of this structure (of UHashStructureT,
    // actually) use indices in this array to refer to points (as
    // opposed to using pointers).
    std::vector<stag::DataPoint> points;

    // This table stores the LSH functions. There are <nHFTuples> rows
    // of <hfTuplesLength> LSH functions.
    std::vector<MultiLSHFunction> lshFunctions;

    // The set of non-empty buckets
    std::vector<std::unordered_map<StagInt,std::vector<StagUInt>>> hashTables;
  };
}

#endif //LSH_H
