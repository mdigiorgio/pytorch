#include "caffe2/mobile/contrib/opencl/core/context.h"
#include "caffe2/mobile/contrib/opencl/core/operator.h"

#include "caffe2/operators/softmax_op.h"

namespace caffe2 {

template <typename T> class CLSoftmaxOp final : public Operator<CLContext> {
public:
  CLSoftmaxOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<CLContext>(operator_def, ws) {}
  virtual ~CLSoftmaxOp() noexcept {}
  USE_OPERATOR_FUNCTIONS(CLContext);
  bool RunOnDevice() override;
private:
  arm_compute::CLSoftmaxLayer softmax_layer_;
  bool first_run_ = true, second_run_ = true;
  CLContext::deleted_unique_ptr<const OpenCLTensor<T>> X_;
};

template <typename T>
bool CLSoftmaxOp<T>::RunOnDevice() {

  auto *Xblob = OperatorBase::Inputs()[0];
  X_ = CLContext::getCLTensor<T>(Xblob, X_.release());

  OpenCLTensor<T> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<OpenCLTensor<T>>();
  if (first_run_) {
    first_run_ = false;
    Y->ResizeLike(*X_);
    softmax_layer_.configure(X_->get_underlying(), Y->get_underlying());
  } else if (second_run_) {
    X_->lazy_allocate(Xblob, second_run_, true);
    second_run_ = false;
    Y->ResizeLike(*X_);
    Y->allocate();
    softmax_layer_.run();
  } else {
    // Configure
    softmax_layer_.configure(X_->get_underlying(), Y->get_underlying());
    // Allocate
    X_->lazy_allocate(Xblob, second_run_, true);
    bool need_allocation = Y->ResizeLike(*X_);
    if (need_allocation) {
      Y->allocate();
    }
    // Run
    softmax_layer_.run();
  }

  return true;
}

REGISTER_CL_OPERATOR(Softmax, CLSoftmaxOp<DataType>);

} // namespace caffe2
