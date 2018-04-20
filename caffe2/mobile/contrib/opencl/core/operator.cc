#include "operator.h"

namespace caffe2 {

CAFFE_DEFINE_REGISTRY(CLOperatorRegistry, OperatorBase, const OperatorDef &,
                      Workspace *);
CAFFE_REGISTER_DEVICE_TYPE(DeviceType::OPENCL, CLOperatorRegistry);

} // namespace caffe2
