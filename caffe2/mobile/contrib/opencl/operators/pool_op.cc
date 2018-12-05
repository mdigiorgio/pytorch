#include "caffe2/operators/pool_op.h"
#include "caffe2/mobile/contrib/opencl/core/context.h"
#include "caffe2/mobile/contrib/opencl/core/operator.h"

namespace caffe2 {

template <typename T>
class CLAveragePoolOp final : public ConvPoolOpBase<CLContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CLContext);
  CLAveragePoolOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CLContext>(operator_def, ws) {
  }
  ~CLAveragePoolOp() {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;
private:
  arm_compute::CLPoolingLayer pooling_layer_;
  bool first_run_ = true, second_run_ = true;
  CLContext::deleted_unique_ptr<const OpenCLTensor<T>> X_;
};

template<typename T>
class CLMaxPoolOp final : public ConvPoolOpBase<CLContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CLContext);
  CLMaxPoolOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CLContext>(operator_def, ws) {
  }
  ~CLMaxPoolOp() {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;
private:
  arm_compute::CLPoolingLayer pooling_layer_;
  bool first_run_ = true, second_run_ = true;
  CLContext::deleted_unique_ptr<const OpenCLTensor<T>> X_;
};

template <>
bool CLAveragePoolOp<DataType>::RunOnDeviceWithOrderNCHW() {

  auto *Xblob = OperatorBase::Inputs()[0];
  if (first_run_) {
    X_ = CLContext::getCLTensor<DataType>(Xblob);
  } else {
    X_ = CLContext::getCLTensor<DataType>(Xblob, X_.release());
  }

  int N = X_->dim32(0);
  int channels = X_->dim32(1);
  int height = X_->dim32(2);
  int width = X_->dim32(3);

  vector<int64_t> output_dims = {N, channels, 1, 1};
  if (!global_pooling_) {
    output_dims[2] = (height + pad_t() + pad_b() - kernel_h()) / stride_h() + 1;
    output_dims[3] = (width + pad_l() + pad_r() - kernel_w()) / stride_w() + 1;
  }

  OpenCLTensor<DataType> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<OpenCLTensor<DataType>>();
  if (first_run_) {
    first_run_ = false;
    CAFFE_ENFORCE_EQ(kernel_.size(), 2, "ARM OpenCL only supports 2D pooling");
    CAFFE_ENFORCE_EQ(kernel_h(), kernel_w(),
                     "ARM OpenCL only supports equal kernel size");
    Y->Resize(output_dims);
    if (global_pooling_) {
      arm_compute::PoolingLayerInfo info(arm_compute::PoolingType::AVG);
      pooling_layer_.configure(X_->get_underlying(), Y->get_underlying(), info);
    } else {
      arm_compute::PadStrideInfo ps_info(stride_w(), stride_h(), pad_l(), pad_r(),
                                         pad_t(), pad_b(),
                                         arm_compute::DimensionRoundingType::FLOOR);
      arm_compute::PoolingLayerInfo info(arm_compute::PoolingType::AVG, kernel_h(),
                                         ps_info);
      pooling_layer_.configure(X_->get_underlying(), Y->get_underlying(), info);
    }
  } else if (second_run_) {
    X_->lazy_allocate(Xblob, second_run_, true);
    second_run_ = false;
    Y->Resize(output_dims);
    Y->allocate();
    pooling_layer_.run();
  } else {
    bool need_allocation =Y->Resize(output_dims);
    // Configure
    if (global_pooling_) {
      arm_compute::PoolingLayerInfo info(arm_compute::PoolingType::AVG);
      pooling_layer_.configure(X_->get_underlying(), Y->get_underlying(), info);
    } else {
      arm_compute::PadStrideInfo ps_info(stride_w(), stride_h(), pad_l(), pad_r(),
                                         pad_t(), pad_b(),
                                         arm_compute::DimensionRoundingType::FLOOR);
      arm_compute::PoolingLayerInfo info(arm_compute::PoolingType::AVG, kernel_h(),
                                         ps_info);
      pooling_layer_.configure(X_->get_underlying(), Y->get_underlying(), info);
    }
    // Allocate
    X_->lazy_allocate(Xblob, second_run_, true);
    if (need_allocation) {
      Y->allocate();
    }
    // Run
    pooling_layer_.run();
  }
  return true;
}

template <> bool CLMaxPoolOp<DataType>::RunOnDeviceWithOrderNCHW() {

  auto *Xblob = OperatorBase::Inputs()[0];
  X_ = CLContext::getCLTensor<DataType>(Xblob, X_.release());

  int N = X_->dim32(0);
  int channels = X_->dim32(1);
  int height = X_->dim32(2);
  int width = X_->dim32(3);

  vector<int64_t> output_dims = {N, channels, 1, 1};
  if (!global_pooling_) {
    output_dims[2] = (height + pad_t() + pad_b() - kernel_h()) / stride_h() + 1;
    output_dims[3] = (width + pad_l() + pad_r() - kernel_w()) / stride_w() + 1;
  }
  OpenCLTensor<DataType> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<OpenCLTensor<DataType>>();

  if (first_run_) {
    first_run_ = false;
    CAFFE_ENFORCE_EQ(kernel_.size(), 2, "ARM OpenCL only supports 2D pooling");
    CAFFE_ENFORCE_EQ(kernel_h(), kernel_w(),
                     "ARM OpenCL only supports equal kernel size");
    Y->Resize(output_dims);
    if (global_pooling_) {
      arm_compute::PoolingLayerInfo info(arm_compute::PoolingType::MAX);
      pooling_layer_.configure(X_->get_underlying(), Y->get_underlying(), info);
    } else {
      arm_compute::PadStrideInfo ps_info(stride_w(), stride_h(), pad_l(), pad_r(),
                                         pad_t(), pad_b(),
                                         arm_compute::DimensionRoundingType::FLOOR);
      arm_compute::PoolingLayerInfo info(arm_compute::PoolingType::MAX, kernel_h(),
                                         ps_info);
      pooling_layer_.configure(X_->get_underlying(), Y->get_underlying(), info);
    }
  } else if (second_run_) {
    X_->lazy_allocate(Xblob, second_run_, true);
    second_run_ = false;
    Y->Resize(output_dims);
    Y->allocate();
    pooling_layer_.run();
  } else {
    bool need_allocation = Y->Resize(output_dims);
    // Configure
    if (global_pooling_) {
      arm_compute::PoolingLayerInfo info(arm_compute::PoolingType::MAX);
      pooling_layer_.configure(X_->get_underlying(), Y->get_underlying(), info);
    } else {
      arm_compute::PadStrideInfo ps_info(stride_w(), stride_h(), pad_l(), pad_r(),
                                         pad_t(), pad_b(),
                                         arm_compute::DimensionRoundingType::FLOOR);
      arm_compute::PoolingLayerInfo info(arm_compute::PoolingType::MAX, kernel_h(),
                                         ps_info);
      pooling_layer_.configure(X_->get_underlying(), Y->get_underlying(), info);
    }
    // Allocate
    X_->lazy_allocate(Xblob, second_run_, true);
    if (need_allocation) {
      Y->allocate();
    }
    // Run
    pooling_layer_.run();
  }

  return true;
}

template <>
bool CLAveragePoolOp<DataType>::RunOnDeviceWithOrderNHWC() {
  return false;
}

template <>
bool CLMaxPoolOp<DataType>::RunOnDeviceWithOrderNHWC() {
  return false;
}

REGISTER_CL_OPERATOR(AveragePool, CLAveragePoolOp<DataType>);
REGISTER_CL_OPERATOR(MaxPool, CLMaxPoolOp<DataType>);

} // namespace caffe2
