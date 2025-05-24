#include <algorithm>
#include <unordered_set>

#include <definitions.h>
#include <random.h>

#include "lsh.h"

#define TWO_ROOT_TWOPI 5.0132565
#define TWO_ROOT_TWO 2.828427124

#define MAX_HASH_RND 536870912U

// Generate a random real distributed uniformly in [rangeStart,
// rangeEnd]. Input must satisfy: rangeStart <= rangeEnd. The
// granularity of generated random reals is given by RAND_MAX.
StagReal genUniformRandom(StagReal rangeStart, StagReal rangeEnd){
  assert(rangeStart <= rangeEnd);

  std::uniform_real_distribution<StagReal> dist(rangeStart, rangeEnd);
  StagReal r = dist(*stag::get_global_rng());

  assert(r >= rangeStart && r <= rangeEnd);
  return r;
}

// Generate a random real from normal distribution N(0,1).
StagReal genGaussianRandom(){
  std::normal_distribution<StagReal> dist(0, 1);
  StagReal z = dist(*stag::get_global_rng());
  return z;
}

// Generate a random 32-bits unsigned (Uns32T) in the range
// [rangeStart, rangeEnd]. Inputs must satisfy: rangeStart <=
// rangeEnd.
StagInt genRandomInt(StagInt rangeStart, StagInt rangeEnd){
  assert(rangeStart <= rangeEnd);

  std::uniform_int_distribution<StagInt> dist(rangeStart, rangeEnd);
  StagInt r = dist(*stag::get_global_rng());

  assert(r >= rangeStart && r <= rangeEnd);
  return r;
}

//------------------------------------------------------------------------------
// Implementation of the LSHFunction class.
//------------------------------------------------------------------------------
lsh::LSHFunction::LSHFunction(StagUInt dimension) {
  dim = dimension;

  // The variable is a random Gaussian vector.
  a.reserve(dim);
  for(StagUInt d = 0; d < dim; d++){
    a.emplace_back(genGaussianRandom() / LSH_PARAMETER_W);
  }

  // The variable b is a random offset on the random vector.
  b = genUniformRandom(0, 1);
}

StagInt lsh::LSHFunction::apply(const stag::DataPoint& point) {
  assert(point.dimension == dim);

  StagReal value = 0;
  for(StagUInt d = 0; d < dim; d++){
    value += point.coordinates[d] * a[d];
  }

  return (StagInt) floor(value + b);
}

StagReal lsh::LSHFunction::collision_probability(StagReal c) {
  StagReal eight_over_c_squared = 8 / SQR(c);
  return - ((1 / TWO_ROOT_TWOPI)
            * c
            * exp(-eight_over_c_squared)
            * (exp(eight_over_c_squared) - 1))
         + erf(TWO_ROOT_TWO / c);
}

//------------------------------------------------------------------------------
// Implementation of the multi-LSH function class.
//------------------------------------------------------------------------------
lsh::MultiLSHFunction::MultiLSHFunction(StagInt dimension, StagInt num_functions) {
    L = num_functions;
    rand_proj.conservativeResize(num_functions, dimension);
    rand_offset.conservativeResize(num_functions);
    uhash_vector.conservativeResize(num_functions);

    std::uniform_real_distribution<StagReal> real_dist(0, 1);
    std::uniform_int_distribution<StagInt> int_dist(1, MAX_HASH_RND);
    std::normal_distribution<StagReal> normal_dist(0, 1);

    for (StagInt g = 0; g < num_functions; g++) {
      rand_offset.coeffRef(g) = real_dist(*stag::get_global_rng());
      uhash_vector.coeffRef(g) = int_dist(*stag::get_global_rng());
      for(StagInt i = 0; i < dimension; i++){
        rand_proj.coeffRef(g, i) = normal_dist(*stag::get_global_rng()) / LSH_PARAMETER_W;
      }
    }
  }

StagInt lsh::MultiLSHFunction::apply(const stag::DataPoint& point) {
  assert((StagInt) point.dimension == rand_proj.cols());
  Eigen::Map<Eigen::VectorXd> pointMap(point.coordinates, (StagInt) point.dimension);
  Eigen::Matrix<StagReal, Eigen::Dynamic, 1> projection = rand_proj * pointMap;
  StagInt h = 0;
  for (auto i = 0; i < L; i++) {
    h += uhash_vector(i) * (StagInt) floor(projection(i) + rand_offset(i));
  }
  return h;
}

//------------------------------------------------------------------------------
// Implementation of the LSH with FullHash class.
//------------------------------------------------------------------------------
lsh::LSHWithFullHash::LSHWithFullHash(StagUInt K,
                                      StagUInt L,
                                      StagInt d) {
  dimension = d;
  parameterK = K;
  parameterL = L;

  initialise_hash_functions();

  // Initialise the empty hash tables
  hashTables.resize(parameterL);
  fullHash.resize(parameterL);
}

std::unordered_set<StagInt> lsh::LSHWithFullHash::add_data(const stag::DataPoint& dp) {
  auto this_index = (StagInt) points.size();
  points.push_back(dp);

  std::unordered_set<StagInt> modified_qids;

  for (StagUInt l = 0; l < parameterL; l++) {
    StagInt this_lsh = compute_lsh(l, dp);

    if (!hashTables[l].contains(this_lsh)) {
      hashTables[l][this_lsh] = std::vector<StagUInt>();
    }
    hashTables[l][this_lsh].push_back(this_index);

    if (fullHash[l].contains(this_lsh)) {
      for (auto q_id : fullHash[l][this_lsh]) {
        modified_qids.insert(q_id);
      }

      for (auto q_id : modified_qids) {
        fullHash[l][this_lsh].erase(q_id);
      }
    }
  }

  return modified_qids;
}

void lsh::LSHWithFullHash::initialise_hash_functions() {
  rnd_vec.resize(parameterK);
  for(StagUInt i = 0; i < parameterK; i++){
    rnd_vec[i] = genRandomInt(1, MAX_HASH_RND);
  }

  lshFunctions.reserve(parameterL);
  for(StagUInt i = 0; i < parameterL; i++){
    lshFunctions.emplace_back(dimension, parameterK);
  }
}

StagInt lsh::LSHWithFullHash::compute_lsh(StagUInt gNumber, const stag::DataPoint& point) {
  return lshFunctions[gNumber].apply(point);
}

std::vector<stag::DataPoint> lsh::LSHWithFullHash::get_near_neighbors(
    const stag::DataPoint& query, StagInt q_id) {
  std::vector<stag::DataPoint> near_points;
  std::unordered_set<StagUInt> near_indices;

  for(StagUInt l = 0; l < parameterL; l++){
    StagInt this_lsh = compute_lsh(l, query);

    if (hashTables[l].contains(this_lsh)) {
      for (StagUInt candidatePIndex : hashTables[l][this_lsh]) {
        if (near_indices.find(candidatePIndex) == near_indices.end()) {
          stag::DataPoint& candidatePoint = points[candidatePIndex];
          near_points.push_back(candidatePoint);
          near_indices.insert(candidatePIndex);
        }
      }
    }

    // Add this query point to the full hash
    if (!fullHash[l].contains(this_lsh)) {
      fullHash[l][this_lsh] = std::unordered_set<StagInt>();
    }
    fullHash[l][this_lsh].insert(q_id);
  }

  return near_points;
}

//------------------------------------------------------------------------------
// Implementation of the Dynamic LSH class.
//------------------------------------------------------------------------------
lsh::LSHDynamic::LSHDynamic(StagUInt K,
                            StagUInt L,
                            StagInt d) {
  dimension = d;
  parameterK = K;
  parameterL = L;

  initialise_hash_functions();

  // Initialise the empty hash tables
  hashTables.resize(parameterL);
}

void lsh::LSHDynamic::add_data(const stag::DataPoint& dp) {
  auto this_index = (StagInt) points.size();
  points.push_back(dp);

  for (StagUInt l = 0; l < parameterL; l++) {
    StagInt this_lsh = compute_lsh(l, dp);

    if (!hashTables[l].contains(this_lsh)) {
      hashTables[l][this_lsh] = std::vector<StagUInt>();
    }
    hashTables[l][this_lsh].push_back(this_index);
  }
}

void lsh::LSHDynamic::initialise_hash_functions() {
  rnd_vec.resize(parameterK);
  for(StagUInt i = 0; i < parameterK; i++){
    rnd_vec[i] = genRandomInt(1, MAX_HASH_RND);
  }

  lshFunctions.reserve(parameterL);
  for(StagUInt i = 0; i < parameterL; i++){
    lshFunctions.emplace_back(dimension, parameterK);
  }
}

StagInt lsh::LSHDynamic::compute_lsh(StagUInt gNumber, const stag::DataPoint& point) {
  return lshFunctions[gNumber].apply(point);
}

std::vector<stag::DataPoint> lsh::LSHDynamic::get_near_neighbors(
    const stag::DataPoint& query) {
  std::vector<stag::DataPoint> near_points;
  std::unordered_set<StagUInt> near_indices;

  for(StagUInt l = 0; l < parameterL; l++){
    StagInt this_lsh = compute_lsh(l, query);

    if (hashTables[l].contains(this_lsh)) {
      for (StagUInt candidatePIndex : hashTables[l][this_lsh]) {
        if (near_indices.find(candidatePIndex) == near_indices.end()) {
          stag::DataPoint& candidatePoint = points[candidatePIndex];
          near_points.push_back(candidatePoint);
          near_indices.insert(candidatePIndex);
        }
      }
    }
  }

  return near_points;
}
