#include "caffe2/mobile/contrib/opencl/core/context.h"
#include "caffe2/mobile/contrib/opencl/core/operator.h"

#include "caffe2/operators/fully_connected_op.h"

namespace caffe2 {

template <typename T> class CLFullyConnectedOp final : public Operator<CLContext> {
public:
  CLFullyConnectedOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<CLContext>(operator_def, ws) {}
  virtual ~CLFullyConnectedOp() noexcept {}
  USE_OPERATOR_FUNCTIONS(CLContext);
  bool RunOnDevice() override;
private:
  arm_compute::CLFullyConnectedLayer fc_layer_;
  bool first_run_ = true, second_run_ = true;
  CLContext::deleted_unique_ptr<const OpenCLTensor<T>> X_, W_, B_;
};

template <typename T>
bool CLFullyConnectedOp<T>::RunOnDevice() {

  auto Xblob = OperatorBase::Inputs()[0];
  auto *Wblob = OperatorBase::Inputs()[1];
  auto *Bblob = OperatorBase::Inputs()[2];

  X_ = CLContext::getCLTensor<T>(Xblob, X_.release());
  if (first_run_) {
    W_ = CLContext::getCLTensor<T>(Wblob);
    B_ = CLContext::getCLTensor<T>(Bblob);
  }

  auto M = X_->dim32(0);
  auto CIn = X_->dim32(1);
  auto Height = X_->dim32(2);
  auto Width = X_->dim32(3);
  auto N = W_->dim32(0);

  CAFFE_ENFORCE_EQ(1, B_->ndim());
  CAFFE_ENFORCE_EQ(N, B_->dim32(0));

  vector<int64_t> output_dims = {M, N};
  OpenCLTensor<T> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<OpenCLTensor<T>>();
  if (first_run_) {
    first_run_ = false;
    Y->Resize(output_dims);

    fc_layer_.configure(X_->get_underlying(), W_->get_underlying(),
                     B_->get_underlying(), Y->get_underlying(), arm_compute::FullyConnectedLayerInfo());
  } else if (second_run_) {
    X_->lazy_allocate(Xblob, second_run_, true);
    W_->lazy_allocate(Wblob, second_run_, second_run_);
    B_->lazy_allocate(Bblob, second_run_, second_run_);
    second_run_ = false;
    Y->Resize(output_dims);
    Y->allocate();
    fc_layer_.run();
  } else {
    bool need_allocation = Y->Resize(output_dims);
    // Configure
    arm_compute::FullyConnectedLayerInfo fc_info = arm_compute::FullyConnectedLayerInfo();
    fc_info.retain_internal_weights = true;
    fc_layer_.configure(X_->get_underlying(), W_->get_underlying(),
                     B_->get_underlying(), Y->get_underlying(), fc_info);
    // Allocate
    X_->lazy_allocate(Xblob, second_run_, true);
    if (need_allocation) {
      Y->allocate();
    }
    // Run
    fc_layer_.run();
  }

  return true;
}

REGISTER_CL_OPERATOR(FC, CLFullyConnectedOp<DataType>);

} // namespace caffe2
