#ifndef CAFFE2_OPT_MOBILE_H_
#define CAFFE2_OPT_MOBILE_H_

#include "caffe2/core/common.h"
#include "nomnigraph/Representations/NeuralNet.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {
namespace opt {

CAFFE2_API void addNNPACK(nom::repr::NNModule* nn, bool low_memory = false);
CAFFE2_API void fuseNNPACKConvRelu(nom::repr::NNModule* nn);
//caffe2::NetDef tryConvertToACLOpenCL(caffe2::NetDef net, bool runFusion, std::unordered_set<std::string> cpuOps);

} // namespace opt
} // namespace caffe2

#endif // CAFFE2_OPT_MOBILE_H_
