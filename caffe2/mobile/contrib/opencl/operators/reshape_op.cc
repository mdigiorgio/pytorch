#include "caffe2/mobile/contrib/opencl/core/context.h"
#include "caffe2/mobile/contrib/opencl/core/operator.h"
#include "caffe2/operators/reshape_op.h"

namespace caffe2 {

template <typename T> class CLReshapeOp final : public Operator<CLContext> {
public:
  CLReshapeOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<CLContext>(operator_def, ws) {}
  virtual ~CLReshapeOp() noexcept {}
  USE_OPERATOR_FUNCTIONS(CLContext);
  bool RunOnDevice() override;
};

template <typename T>
bool CLReshapeOp<T>::RunOnDevice() {
  auto *Xblob = OperatorBase::Inputs()[0];
  auto X = CLContext::getCLTensor<T>(Xblob);
  auto arg = OperatorBase::GetRepeatedArgument<int>("shape");
  for (int i = 0; i < arg.size(); ++i) {
    LOG(INFO) << "[C2DEBUG] shape: " << arg[i];
  }
  return true;
}

REGISTER_CL_OPERATOR(Reshape, CLReshapeOp<DataType>);

} // namespace caffe2
