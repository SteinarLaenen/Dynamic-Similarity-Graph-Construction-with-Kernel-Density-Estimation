#include "timing.h"

void SimpleTimer::start() {
  t1 = std::chrono::high_resolution_clock::now();
};

void SimpleTimer::stop() {
  t2 = std::chrono::high_resolution_clock::now();
};

StagInt SimpleTimer::elapsed_ms() {
  auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  return time_ms.count();
}
