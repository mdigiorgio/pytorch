// #include "caffe2/mobile/contrib/opencl/core/context.h"
// #include "caffe2/mobile/contrib/opencl/core/operator.h"

// #include "caffe2/operators/conv_pool_op_base.h"

// namespace caffe2 {

// template <typename T>
// class CLChannelShuffleOp final : public ConvPoolOpBase<CLContext> {
// public:
//   USE_OPERATOR_FUNCTIONS(CLContext);
//   CLChannelShuffleOp(const OperatorDef &operator_def, Workspace *ws)
//       : ConvPoolOpBase<CLContext>(operator_def, ws) {}
//   virtual ~CLChannelShuffleOp() noexcept {}
//   bool RunOnDevice() override;
// private:
//   arm_compute::CLChannelShuffleLayer cs_layer_;
//   bool first_run_ = true, second_run_ = true;
//   CLContext::deleted_unique_ptr<const OpenCLTensor<T>> X_;
// };

// template <typename T>
// bool CLChannelShuffleOp<T>::RunOnDevice() {

//   auto *Xblob = OperatorBase::Inputs()[0];
//   X_ = CLContext::getCLTensor<T>(Xblob, X_.release());

//   OpenCLTensor<T> *Y =
//       OperatorBase::Outputs()[0]->template GetMutable<OpenCLTensor<T>>();

//   if (first_run_) {
//     first_run_ = false;
//     if (Y->get_underlying() != X_->get_underlying())
//     {
//       Y->ResizeLike(*X_);
//     }
//     cs_layer_.configure(X_->get_underlying(), Y->get_underlying(), group_);
//   } else if (second_run_) {
//     X_->lazy_allocate(Xblob, second_run_, true);
//     second_run_ = false;
//     // in place operation, do not need to allocate new memory
//     if (Y->get_underlying() != X_->get_underlying()) {
//       Y->ResizeLike(*X_);
//       Y->allocate();
//     }
//     cs_layer_.run();
//   } else {
//     X_->lazy_allocate(Xblob, second_run_, true);
//     bool need_allocation = false;
//     if (Y->get_underlying() != X_->get_underlying()) {
//       need_allocation = Y->ResizeLike(*X_, true);
//     }
//     cs_layer_.configure(X_->get_underlying(), Y->get_underlying(), group_);
//     if (need_allocation) {
//       Y->allocate();
//     }
//     cs_layer_.run();
//   }

//   return true;
// }

// REGISTER_CL_OPERATOR(Relu, CLChannelShuffleOp<DataType>);

// } // namespace caffe2
