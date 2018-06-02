#include "caffe2/mobile/contrib/opencl/core/context.h"
 #include "caffe2/mobile/contrib/opencl/core/operator.h"

#include "caffe2/operators/spatial_batch_norm_op.h"

namespace caffe2 {

template <typename T> class CLSpatialBNOp final : public Operator<CLContext> {
public:
  CLSpatialBNOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<CLContext>(operator_def, ws),
        is_test_(OperatorBase::GetSingleArgument<int>(OpSchema::Arg_IsTest, 0)),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5)),
        momentum_(OperatorBase::GetSingleArgument<float>("momentum", 0.9)),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))) { }
  virtual ~CLSpatialBNOp() noexcept {}
  USE_OPERATOR_FUNCTIONS(CLContext);
  bool RunOnDevice() override;
 protected:
  bool is_test_;
  double epsilon_;
  double momentum_;
  StorageOrder order_;
  INPUT_TAGS(INPUT, SCALE, BIAS, EST_MEAN, EST_VAR);
  OUTPUT_TAGS(OUTPUT, RUNNING_MEAN, RUNNING_VAR, SAVED_MEAN, SAVED_INV_VAR);
private:
  arm_compute::CLBatchNormalizationLayer bn_layer_;
  bool first_run_ = true, second_run_ = true;
  CLContext::deleted_unique_ptr<const OpenCLTensor<T>> X_, mean_, var_, bias_, scale_;
};

template <typename T>
bool CLSpatialBNOp<T>::RunOnDevice() {
  auto *XBlob = OperatorBase::Inputs()[0];
  auto *scaleBlob = OperatorBase::Inputs()[SCALE];
  auto *biasBlob = OperatorBase::Inputs()[BIAS];
  auto *meanBlob = OperatorBase::Inputs()[EST_MEAN];
  auto *varBlob = OperatorBase::Inputs()[EST_VAR];

  X_ = CLContext::getCLTensor<T>(XBlob, X_.release());
  if (first_run_) {
    scale_ = CLContext::getCLTensor<T>(scaleBlob);
    bias_ = CLContext::getCLTensor<T>(biasBlob);
    mean_ = CLContext::getCLTensor<T>(meanBlob);
    var_ = CLContext::getCLTensor<T>(varBlob);
  }

  auto C = X_->dim32(1);
  CAFFE_ENFORCE_EQ(scale_->ndim(), 1);
  CAFFE_ENFORCE_EQ(bias_->ndim(), 1);
  CAFFE_ENFORCE_EQ(mean_->ndim(), 1);
  CAFFE_ENFORCE_EQ(var_->ndim(), 1);

  CAFFE_ENFORCE_EQ(scale_->dim32(0), C);
  CAFFE_ENFORCE_EQ(bias_->dim32(0), C);
  CAFFE_ENFORCE_EQ(mean_->dim32(0), C);
  CAFFE_ENFORCE_EQ(var_->dim32(0), C);

  OpenCLTensor<T> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<OpenCLTensor<T>>();
  if (first_run_) {
    first_run_ = false;
    Y->ResizeLike(*X_);
    bn_layer_.configure(X_->get_underlying(), Y->get_underlying(),
                     mean_->get_underlying(), var_->get_underlying(),
                     bias_->get_underlying(), scale_->get_underlying(), epsilon_);
  } else if (second_run_) {
    X_->lazy_allocate(XBlob, second_run_, true);
    scale_->lazy_allocate(scaleBlob, second_run_, second_run_);
    bias_->lazy_allocate(biasBlob, second_run_, second_run_);
    mean_->lazy_allocate(meanBlob, second_run_, second_run_);
    var_->lazy_allocate(varBlob, second_run_, second_run_);
    second_run_ = false;
    Y->ResizeLike(*X_);
    Y->allocate();
    bn_layer_.run();
  } else {
    // Configure
    bn_layer_.configure(X_->get_underlying(), Y->get_underlying(),
                     mean_->get_underlying(), var_->get_underlying(),
                     bias_->get_underlying(), scale_->get_underlying(), epsilon_);
    // Allocate
    X_->lazy_allocate(XBlob, second_run_, true);
    bool need_allocation = Y->ResizeLike(*X_);
    if (need_allocation) {
      Y->allocate();
    }
    // Run
    bn_layer_.run();
  }
  return true;
}

REGISTER_CL_OPERATOR(SpatialBN, CLSpatialBNOp<DataType>);

} // namespace caffe2
