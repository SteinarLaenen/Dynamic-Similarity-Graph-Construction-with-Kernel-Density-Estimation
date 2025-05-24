#ifndef DYNAMIC_KDE_CLUSTERING_TIMING_H
#define DYNAMIC_KDE_CLUSTERING_TIMING_H
#include <chrono>

#include <definitions.h>

class SimpleTimer {
public:
  SimpleTimer() = default;

  void start();

  void stop();

  StagInt elapsed_ms();

private:
  std::chrono::time_point<std::chrono::system_clock> t1;
  std::chrono::time_point<std::chrono::system_clock> t2;
};

#endif //DYNAMIC_KDE_CLUSTERING_TIMING_H
