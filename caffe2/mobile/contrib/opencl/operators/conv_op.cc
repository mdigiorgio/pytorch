#include "caffe2/mobile/contrib/opencl/core/context.h"
#include "caffe2/mobile/contrib/opencl/core/operator.h"

#include "arm_compute/runtime/CL/CLSubTensor.h"
#include "caffe2/operators/conv_op.h"

namespace caffe2 {

struct EmptyActivation {
  static string act() {
    return "";
  }
};

struct ReluActivation {
  static string act() {
    return "relu";
  }
};

struct SigmoidActivation {
  static string act() {
    return "sigmoid";
  }
};
template <typename T, typename Activation>
class CLConvOp final : public ConvPoolOpBase<CLContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CLContext);
  CLConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CLContext>(operator_def, ws) {
    // Since this is the default convolution implementation, we will
    // use CAFFE_ENFORCE instead of OPERATOR_NEEDS_FEATURE.
    CAFFE_ENFORCE(
        group_ == 1 || order_ == StorageOrder::NCHW,
        "Group convolution only supports NCHW order right now.");
  }
  ~CLConvOp() {}

  bool RunOnDevice() override;
private:
  void gconv_configure(arm_compute::CLConvolutionLayer* conv, arm_compute::ICLTensor *input, arm_compute::ICLTensor *weights, arm_compute::ICLTensor *biases, arm_compute::ICLTensor *output, const arm_compute::PadStrideInfo &conv_info, const arm_compute::WeightsInfo &weights_info, int groups, int idx) {
    const auto& input_shape = compute_output_shape(input->info()->tensor_shape(), groups, 2, idx);
    const auto& weights_shape = compute_output_shape(weights->info()->tensor_shape(), groups, 3, idx);
    const auto& biases_shape = compute_output_shape(biases->info()->tensor_shape(), groups, 0, idx);
    const auto& output_shape = compute_output_shape(output->info()->tensor_shape(), groups, 2, idx);
    LOG(ERROR) << "[C2DEBUG] input shape: " << input_shape.first[3] << " " << input_shape.first[2] << " " << input_shape.first[1] << " " << input_shape.first[0];
    LOG(ERROR) << "[C2DEBUG] weights shape: " << weights_shape.first[3] << " " << weights_shape.first[2] << " " << weights_shape.first[1] << " " << weights_shape.first[0];
    LOG(ERROR) << "[C2DEBUG] bias shape: " << biases_shape.first[0];
    auto input_ = std::unique_ptr<arm_compute::CLSubTensor>(new arm_compute::CLSubTensor(input, input_shape.first, input_shape.second, true));
    auto output_ = std::unique_ptr<arm_compute::CLSubTensor>(new arm_compute::CLSubTensor(output, output_shape.first, output_shape.second, true));
    auto weights_ = std::unique_ptr<arm_compute::CLSubTensor>(new arm_compute::CLSubTensor(weights, weights_shape.first, weights_shape.second, true));
    auto biases_ = std::unique_ptr<arm_compute::CLSubTensor>(new arm_compute::CLSubTensor(biases, biases_shape.first, biases_shape.second, true));
    conv->configure(input_.get(), weights_.get(), biases_.get(), output_.get(), conv_info, weights_info);
  }
  arm_compute::CLConvolutionLayer conv_;
  std::vector<std::unique_ptr<arm_compute::CLConvolutionLayer>> gconv_;
  arm_compute::CLDepthwiseConvolutionLayer3x3 depth_conv_;
  bool first_run_ = true, second_run_ = true;
  CLContext::deleted_unique_ptr<const OpenCLTensor<T>> X_, filter_, bias_;
};

template <typename T, typename Activation>
bool CLConvOp<T, Activation>::RunOnDevice() {
  auto *Xblob = OperatorBase::Inputs()[0];
  auto *filterblob = OperatorBase::Inputs()[1];
  auto *biasblob = OperatorBase::Inputs()[2];
  X_ = CLContext::getCLTensor<T>(Xblob, X_.release());
  filter_ = CLContext::getCLTensor<T>(filterblob, filter_.release());
  bias_ = CLContext::getCLTensor<T>(biasblob, bias_.release());

  OpenCLTensor<T> *Y =
    OperatorBase::Outputs()[0]->template GetMutable<OpenCLTensor<T>>();

  const int N = X_->dim32(0), H = X_->dim32(2), W = X_->dim32(3), C = X_->dim32(1);
  LOG(INFO) << "[C2DEBUG] Conv " << N << " " << H << " " << W << " " << C;
  CAFFE_ENFORCE_EQ(kernel_.size(), 2,
                   "Only 2d convolution is supported with ARM compute backend");

  CAFFE_ENFORCE(X_->ndim() == filter_->ndim());
  const int M = filter_->dim32(0);
  CAFFE_ENFORCE(filter_->dim32(2) == kernel_h());
  CAFFE_ENFORCE(filter_->dim32(3) == kernel_w());
  bool depthwise = group_ == C;
  bool grouped = group_ > 1;
  LOG(ERROR) << "[C2DEBUG] group: " << group_ << " depthwise: " << depthwise << " grouped: " << grouped;
  CAFFE_ENFORCE(C % group_ == 0);
  CAFFE_ENFORCE(M % group_ == 0);
  CAFFE_ENFORCE(filter_->dim32(1) == C / group_);

  // Weights are being resused between runs, hence we need to specifiy to retain them if they were marked as unused
  bool retain_internal_weights = !filter_->get_underlying()->is_used();

  if (first_run_) {
    if (grouped) {
      for(int i = 0; i < group_; ++i) {
        std::unique_ptr<arm_compute::CLConvolutionLayer> gconv(new arm_compute::CLConvolutionLayer());
        gconv_.push_back(std::move(gconv));
      }
    }
    first_run_ = false;

    // resize output accordingly
    TensorCPU fakeX;
    fakeX.Resize(X_->dims());
    TensorCPU fakeY;
    ConvPoolOpBase<CLContext>::SetOutputSize(fakeX, &fakeY, filter_->dim32(0));
    LOG(ERROR) << "[C2DEBUG] fakeY: " << fakeY.dims();
    Y->ResizeLike(fakeY);
    LOG(ERROR) << "[C2DEBUG] dims of X " << X_->dims();
    LOG(ERROR) << "[C2DEBUG] dims of X(gctensor) "
      << X_->get_underlying()->info()->dimension(3) << " "
      << X_->get_underlying()->info()->dimension(2) << " "
      << X_->get_underlying()->info()->dimension(1) << " "
      << X_->get_underlying()->info()->dimension(0) << " "
    ;
    LOG(ERROR) << "[C2DEBUG] dims of Y " << Y->dims();
    LOG(ERROR) << "[C2DEBUG] dims of Y(gctensor) "
      << Y->get_underlying()->info()->dimension(3) << " "
      << Y->get_underlying()->info()->dimension(2) << " "
      << Y->get_underlying()->info()->dimension(1) << " "
      << Y->get_underlying()->info()->dimension(0) << " "
    ;
    LOG(ERROR) << "[C2DEBUG] dims of filter_ " << filter_->dims();
    LOG(ERROR) << "[C2DEBUG] dims of filter_(gctensor) "
      << filter_->get_underlying()->info()->dimension(3) << " "
      << filter_->get_underlying()->info()->dimension(2) << " "
      << filter_->get_underlying()->info()->dimension(1) << " "
      << filter_->get_underlying()->info()->dimension(0) << " "
    ;

    if (depthwise) {
      depth_conv_.configure(X_->get_underlying(), filter_->get_underlying(), bias_->get_underlying(),
                            Y->get_underlying(),
                            arm_compute::PadStrideInfo(stride_[0], stride_[1], pads_[0], pads_[1]));
    } else if (grouped) {
      for (int i = 0; i < group_; ++i) {
        LOG(ERROR) << "[C2DEBUG] configure gconv " << i;
        gconv_configure(gconv_[i].get(), X_->get_underlying(), filter_->get_underlying(), bias_->get_underlying(),
                      Y->get_underlying(),
                        arm_compute::PadStrideInfo(stride_[0], stride_[1], pads_[0], pads_[1]), arm_compute::WeightsInfo(false, 0, 0, 0, retain_internal_weights), group_, i);
        retain_internal_weights = true;
      }
    } else {
      string activation = Activation::act();
      if (activation == "") {
        conv_.configure(X_->get_underlying(), filter_->get_underlying(), bias_->get_underlying(),
                        Y->get_underlying(),
                        arm_compute::PadStrideInfo(stride_[0], stride_[1], pads_[0], pads_[1]));
      } else {
        arm_compute::ActivationLayerInfo act_info;
        if (activation == "relu") {
          act_info = arm_compute::ActivationLayerInfo(arm_compute::ActivationLayerInfo::ActivationFunction::RELU);
        } else if (activation == "sigmoid") {
          act_info = arm_compute::ActivationLayerInfo(arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC);
        }
        conv_.configure(X_->get_underlying(), filter_->get_underlying(), bias_->get_underlying(),
                        Y->get_underlying(),
                        arm_compute::PadStrideInfo(stride_[0], stride_[1], pads_[0], pads_[1]), arm_compute::WeightsInfo(), arm_compute::Size2D(1, 1), act_info);
      }
    }
  } else if (second_run_) {
    LOG(ERROR) << "[C2DEBUG] second";
    // Always attempt to copy the CPU to GPU on input
    X_->lazy_allocate(Xblob, second_run_, true);
    filter_->lazy_allocate(filterblob, second_run_, second_run_);
    bias_->lazy_allocate(biasblob, second_run_, second_run_);
    second_run_ = false;
    LOG(ERROR) << "[C2DEBUG] second before Y allocate";
    LOG(INFO) << "[C2DEBUG] dims of X(gctensor) "
      << X_->get_underlying()->info()->dimension(3) << " "
      << X_->get_underlying()->info()->dimension(2) << " "
      << X_->get_underlying()->info()->dimension(1) << " "
      << X_->get_underlying()->info()->dimension(0) << " "
    ;
    LOG(ERROR) << "[C2DEBUG] dims of Y(gctensor) "
      << Y->get_underlying()->info()->dimension(3) << " "
      << Y->get_underlying()->info()->dimension(2) << " "
      << Y->get_underlying()->info()->dimension(1) << " "
      << Y->get_underlying()->info()->dimension(0) << " "
    ;
    Y->allocate();
    LOG(ERROR) << "[C2DEBUG] After Y allocate";
    if (depthwise) {
      LOG(ERROR) << "[C2DEBUG] depthwise";
      depth_conv_.run();
    } else if (grouped) {
      LOG(ERROR) << "[C2DEBUG] In grouped run";
      for (int i = 0; i < group_; ++i) {
        LOG(ERROR) << "[C2DEBUG] gconv_[" << i << "] run.";
        gconv_[i].get()->run();
      }
    } else {
      LOG(ERROR) << "[C2DEBUG] second before conv_.run()";
      conv_.run();
      LOG(ERROR) << "[C2DEBUG] second after conv_.run()";
    }
  } else {
    LOG(ERROR) << "[C2DEBUG] normal run";
    X_->lazy_allocate(Xblob, second_run_, true);
    filter_->lazy_allocate(filterblob, second_run_, true);
    bias_->lazy_allocate(biasblob, second_run_, true);
    LOG(ERROR) << "[C2DEBUG] after X";
    TensorCPU fakeX;
    fakeX.Resize(X_->dims());
    TensorCPU fakeY;
    ConvPoolOpBase<CLContext>::SetOutputSize(fakeX, &fakeY, filter_->dim32(0));
    LOG(ERROR) << "[C2DEBUG] after SetOutputSize";
    bool need_allocation = Y->ResizeLike(fakeY, true);
    if (need_allocation) {
      LOG(ERROR) << "[C2DEBUG] Y->allocate() in third run.";
      Y->allocate();
    }
    LOG(ERROR) << "[C2DEBUG] after Y->allocate()";
    if (depthwise) {
      LOG(ERROR) << "[C2DEBUG] Running depthwise conv";
      depth_conv_.configure(X_->get_underlying(), filter_->get_underlying(), bias_->get_underlying(),
                            Y->get_underlying(),
                            arm_compute::PadStrideInfo(stride_[0], stride_[1], pads_[0], pads_[1]));
      depth_conv_.run();
    } else if (grouped) {
      for (int i = 0; i < group_; ++i) {
        gconv_configure(gconv_[i].get(), X_->get_underlying(), filter_->get_underlying(), bias_->get_underlying(),
                      Y->get_underlying(),
                        arm_compute::PadStrideInfo(stride_[0], stride_[1], pads_[0], pads_[1]), arm_compute::WeightsInfo(false, 0, 0, 0, retain_internal_weights), group_, i);
        gconv_[i]->run();
        retain_internal_weights = true;
      }
    } else {
      LOG(ERROR) << "[C2DEBUG] before conv_.configure";
      conv_.configure(X_->get_underlying(), filter_->get_underlying(), bias_->get_underlying(),
                      Y->get_underlying(),
                      arm_compute::PadStrideInfo(stride_[0], stride_[1], pads_[0], pads_[1]), arm_compute::WeightsInfo(false, 0, 0, 0, retain_internal_weights));
      LOG(ERROR) << "[C2DEBUG] before conv_.run";
      conv_.run();
      LOG(ERROR) << "[C2DEBUG] after conv_.run";
    }
 }
  LOG(ERROR) << "[C2DEBUG] after before return";
  return true;
}

REGISTER_CL_OPERATOR(Conv, CLConvOp<DataType, EmptyActivation>);
REGISTER_CL_OPERATOR(ConvRelu, CLConvOp<DataType, ReluActivation>);
REGISTER_CL_OPERATOR(ConvSigmoid, CLConvOp<DataType, SigmoidActivation>);

} // namespace caffe2
