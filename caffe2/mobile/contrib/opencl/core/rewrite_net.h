
#pragma once
#include "caffe2/mobile/contrib/opencl/core/net_cl.h"
#include <unordered_set>

namespace caffe2 {
bool tryConvertToOpenCL(const NetDef& predictNet,
                        NetDef* glPredictNet,
                        bool runFusion,
                        std::unordered_set<std::string> cpuOps);

// Exposed for testing
NetDef rewritePredictNetForOpenCL(const NetDef& predictNet,
                                  bool runFusion,
                                  std::unordered_set<std::string> cpuOps);
void dumpDefForOpenCL(const NetDef& net);
} // namespace caffe2
