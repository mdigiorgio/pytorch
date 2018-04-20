#ifndef CAFFE2_OPENCL_OPERATORS_ACTIVATION_OPS_H_
#define CAFFE2_OPENCL_OPERATORS_ACTIVATION_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T>
class CLSigmoidOp final : public Operator<CLContext> {
public:
  CLSigmoidOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<CLContext>(operator_def, ws) {}
  USE_OPERATOR_FUNCTIONS(CLContext);
  bool RunOnDevice() override;
private:
  arm_compute::CLActivationLayer sigmoid_layer_;
  bool first_run_ = true, second_run_ = true;
  CLContext::deleted_unique_ptr<const OpenCLTensor<T>> X_;
};

template <typename T> class CLReluOp final : public Operator<CLContext> {
public:
  CLReluOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<CLContext>(operator_def, ws) {}
  virtual ~CLReluOp() noexcept {}
  USE_OPERATOR_FUNCTIONS(CLContext);
  bool RunOnDevice() override;
private:
  arm_compute::CLActivationLayer relu_layer_;
  bool first_run_ = true, second_run_ = true;
  CLContext::deleted_unique_ptr<const OpenCLTensor<T>> X_;

};

} // namespace caffe2

#endif // CAFFE2_OPENCL_OPERATORS_ACTIVATION_OPS_H_
