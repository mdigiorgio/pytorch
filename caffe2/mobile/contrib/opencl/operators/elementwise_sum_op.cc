#include "caffe2/mobile/contrib/opencl/core/context.h"
#include "caffe2/mobile/contrib/opencl/core/operator.h"
#include "caffe2/operators/utility_ops.h"

namespace caffe2 {

template <typename T> class CLSumOp final : public Operator<CLContext> {
public:
  CLSumOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<CLContext>(operator_def, ws) {}
  virtual ~CLSumOp() noexcept {}
  USE_OPERATOR_FUNCTIONS(CLContext);
  bool RunOnDevice() override;
private:
  arm_compute::CLArithmeticAddition add_layer_;
  bool first_run_ = true, second_run_ = true;
  CLContext::deleted_unique_ptr<const OpenCLTensor<T>> A_, B_;
};


template <typename T>
bool CLSumOp<T>::RunOnDevice() {

  auto *Ablob = OperatorBase::Inputs()[0];
  auto *Bblob = OperatorBase::Inputs()[1];

  A_ = CLContext::getCLTensor<T>(Ablob, A_.release());
  B_ = CLContext::getCLTensor<T>(Bblob, B_.release());

  OpenCLTensor<T> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<OpenCLTensor<T>>();
  if (first_run_) {
    first_run_ = false;
    Y->ResizeLike(*A_);
    add_layer_.configure(A_->get_underlying(), B_->get_underlying(), Y->get_underlying(), arm_compute::ConvertPolicy::SATURATE);
  } else if (second_run_) {
    A_->lazy_allocate(Ablob, second_run_, true);
    B_->lazy_allocate(Bblob, second_run_, true);
    second_run_ = false;
    Y->allocate();
    add_layer_.run();
  } else {
    bool need_allocation = Y->ResizeLike(*A_);
    // Configure
    add_layer_.configure(A_->get_underlying(), B_->get_underlying(), Y->get_underlying(), arm_compute::ConvertPolicy::SATURATE);
    // Allocate
    A_->lazy_allocate(Ablob, second_run_, true);
    B_->lazy_allocate(Bblob, second_run_, true);
    if (need_allocation) {
      Y->allocate();
    }
    // Run
    add_layer_.run();
  }

  return true;
}

REGISTER_CL_OPERATOR(Sum, CLSumOp<DataType>);
REGISTER_CL_OPERATOR(Add, CLSumOp<DataType>);

} // namespace caffe2
