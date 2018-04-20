// #include "caffe2/mobile/contrib/opencl/core/context.h"
// #include "caffe2/mobile/contrib/opencl/core/operator.h"

// namespace caffe2 {

// template <typename T>
// class CLNormalizePlanarYUVOp final : public Operator<CLContext> {
// public:
//   CLNormalizePlanarYUVOp(const OperatorDef &operator_def, Workspace *ws)
//       : Operator<CLContext>(operator_def, ws) {}
//   virtual ~CLNormalizePlanarYUVOp() noexcept {}
//   USE_OPERATOR_FUNCTIONS(CLContext);
//   bool RunOnDevice() override;
// private:
//   arm_compute::GCNormalizePlanarYUVLayer norm_layer_;
//   bool first_run_ = true, second_run_ = true;
//   CLContext::deleted_unique_ptr<const OpenCLTensor<T>> X_, mean_, sd_;
// };

// template <typename T> bool CLNormalizePlanarYUVOp<T>::RunOnDevice() {

//   auto Xblob = OperatorBase::Inputs()[0];
//   auto *meanblob = OperatorBase::Inputs()[1];
//   auto *sdblob = OperatorBase::Inputs()[2];

//   X_ = CLContext::getCLTensor<T>(Xblob, X_.release());
//   if (first_run_) {
//     mean_ = CLContext::getCLTensor<T>(meanblob);
//     sd_ = CLContext::getCLTensor<T>(sdblob);
//   }

//   CAFFE_ENFORCE_EQ(X_->ndim(), 4);
//   auto N = X_->dim32(0);
//   auto C = X_->dim32(1);
//   auto H = X_->dim32(2);
//   auto W = X_->dim32(3);

//   CAFFE_ENFORCE_EQ(C, mean_->dim32(1));
//   CAFFE_ENFORCE_EQ(C, sd_->dim32(1));

//   OpenCLTensor<T> *Y =
//       OperatorBase::Outputs()[0]->template GetMutable<OpenCLTensor<T>>();
//   if (first_run_) {
//     first_run_ = false;
//     Y->ResizeLike(*X_);
//     norm_layer_.configure(X_->get_underlying(), Y->get_underlying(), mean_->get_underlying(), sd_->get_underlying());
//   } else if (second_run_) {
//     X_->lazy_allocate(Xblob, second_run_, true);
//     mean_->lazy_allocate(meanblob, second_run_, second_run_);
//     sd_->lazy_allocate(sdblob, second_run_, second_run_);
//     second_run_ = false;
//     Y->ResizeLike(*X_);
//     Y->allocate();
//     norm_layer_.run();
//   } else {
//     X_->lazy_allocate(Xblob, second_run_, true);
//     bool need_allocation = Y->ResizeLike(*X_);
//     norm_layer_.configure(X_->get_underlying(), Y->get_underlying(), mean_->get_underlying(), sd_->get_underlying());
//     if (need_allocation) {
//       Y->allocate();
//     }
//     norm_layer_.run();
//   }

//   return true;
// }

// REGISTER_CL_OPERATOR(NormalizePlanarYUV, CLNormalizePlanarYUVOp<DataType>);

// } // namespace caffe2
