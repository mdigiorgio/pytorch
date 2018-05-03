#include "caffe2/mobile/contrib/opencl/core/context.h"
#include "caffe2/mobile/contrib/opencl/core/operator.h"

#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

template <typename T>
class CLChannelShuffleOp final : public ConvPoolOpBase<CLContext> {
public:
  USE_OPERATOR_FUNCTIONS(CLContext);
  CLChannelShuffleOp(const OperatorDef &operator_def, Workspace *ws)
      : ConvPoolOpBase<CLContext>(operator_def, ws) {}
  virtual ~CLChannelShuffleOp() noexcept {}
  bool RunOnDevice() override;
private:
  arm_compute::CLChannelShuffleLayer cs_layer_;
  bool first_run_ = true, second_run_ = true;
  CLContext::deleted_unique_ptr<const OpenCLTensor<T>> X_;
};

template <typename T>
bool CLChannelShuffleOp<T>::RunOnDevice() {

  auto *Xblob = OperatorBase::Inputs()[0];
  X_ = CLContext::getCLTensor<T>(Xblob, X_.release());

  LOG(ERROR) << "[C2DEBUG] channel shuffle X_: " << X_->dim32(0) << " " << X_->dim32(1)
             << " " << X_->dim32(2) << " " << X_->dim32(3);
  OpenCLTensor<T> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<OpenCLTensor<T>>();

  if (first_run_) {
    LOG(ERROR) << "[C2DEBUG] first_run";
    first_run_ = false;
    if (Y->get_underlying() != X_->get_underlying())
    {
      Y->ResizeLike(*X_);
    }
    cs_layer_.configure(X_->get_underlying(), Y->get_underlying(), group_);
  } else if (second_run_) {
    LOG(ERROR) << "[C2DEBUG] second_run";
    X_->lazy_allocate(Xblob, second_run_, true);
    second_run_ = false;
    // in place operation, do not need to allocate new memory
    if (Y->get_underlying() != X_->get_underlying()) {
      Y->allocate();
    }
    cs_layer_.run();
  } else {
    LOG(ERROR) << "[C2DEBUG] third_run+";
    LOG(ERROR) << "[C2DEBUG] X->lazy_allocate";
    X_->lazy_allocate(Xblob, second_run_, true);
    bool need_allocation = false;
    if (Y->get_underlying() != X_->get_underlying()) {
      LOG(ERROR) << "[C2DEBUG] channel shuffle: Y->ResizeLike(X);";
      need_allocation = Y->ResizeLike(*X_, true);
    }
    LOG(ERROR) << "[C2DEBUG] need allocation" << need_allocation;
    cs_layer_.configure(X_->get_underlying(), Y->get_underlying(), group_);
    if (need_allocation) {
      Y->allocate();
    }
    cs_layer_.run();
  }

  return true;
}

REGISTER_CL_OPERATOR(ChannelShuffle, CLChannelShuffleOp<DataType>);

} // namespace caffe2
