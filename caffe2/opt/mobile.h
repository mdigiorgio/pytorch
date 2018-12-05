#ifndef CAFFE2_OPT_MOBILE_H_
#define CAFFE2_OPT_MOBILE_H_

#include "nomnigraph/Representations/NeuralNet.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {
namespace opt {

void addNNPACK(nom::repr::NNModule* nn, bool low_memory = false);
void fuseNNPACKConvRelu(nom::repr::NNModule* nn);
//caffe2::NetDef tryConvertToACLOpenCL(caffe2::NetDef net, bool runFusion, std::unordered_set<std::string> cpuOps);

} // namespace opt
} // namespace caffe2

#endif // CAFFE2_OPT_MOBILE_H_
