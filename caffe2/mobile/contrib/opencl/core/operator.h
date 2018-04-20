#ifndef CAFFE2_OPENCL_OPERATOR_H_
#define CAFFE2_OPENCL_OPERATOR_H_

#include "caffe2/core/operator.h"
#include "caffe2/core/registry.h"

namespace caffe2 {

CAFFE_DECLARE_REGISTRY(CLOperatorRegistry, OperatorBase, const OperatorDef &,
                       Workspace *);
#define REGISTER_CL_OPERATOR_CREATOR(key, ...)                                 \
  CAFFE_REGISTER_CREATOR(CLOperatorRegistry, key, __VA_ARGS__)
#define REGISTER_CL_OPERATOR(name, ...)                                        \
  extern void CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();                  \
  static void CAFFE2_UNUSED CAFFE_ANONYMOUS_VARIABLE_CL##name() {              \
    CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();                            \
  }                                                                            \
  CAFFE_REGISTER_CLASS(CLOperatorRegistry, name, __VA_ARGS__)
#define REGISTER_CL_OPERATOR_STR(str_name, ...)                                \
  CAFFE_REGISTER_TYPED_CLASS(CLOperatorRegistry, str_name, __VA_ARGS__)

#define REGISTER_CL_OPERATOR_WITH_ENGINE(name, engine, ...)                    \
  CAFFE_REGISTER_CLASS(CLOperatorRegistry, name##_ENGINE_##engine, __VA_ARGS__)

} // namespace caffe2

#endif // CAFFE2_OPENCL_OPERATOR_H_
