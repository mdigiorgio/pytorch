#include "caffe2/mobile/contrib/opencl/core/context.h"
#include "caffe2/mobile/contrib/opencl/core/operator.h"

#include "caffe2/operators/bbox_transform_op.h"

namespace caffe2 {

template <typename T>
class CLBBoxTransformOp final : public Operator<CLContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CLContext);
  CLBBoxTransformOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CLContext>(operator_def, ws),
        weights_(OperatorBase::GetRepeatedArgument<T>(
            "weights",
            vector<T>{1.0f, 1.0f, 1.0f, 1.0f})),
        apply_scale_(
            OperatorBase::GetSingleArgument<bool>("apply_scale", true)),
        correct_transform_coords_(OperatorBase::GetSingleArgument<bool>(
            "correct_transform_coords",
            false)) {
    CAFFE_ENFORCE_EQ(
        weights_.size(),
        4,
        "weights size " + caffe2::to_string(weights_.size()) + "must be 4.");
  }
  ~CLBBoxTransformOp() {}

  bool RunOnDevice() override;
protected:
  // weights [wx, wy, ww, wh] to apply to the regression target
  vector<T> weights_;
  // Transform the boxes to the scaled image space after applying the bbox
  //   deltas.
  // Set to false to match the detectron code, set to true for the keypoint
  //   model and for backward compatibility
  bool apply_scale_{true};
  // Correct bounding box transform coordates, see bbox_transform() in boxes.py
  // Set to true to match the detectron code, set to false for backward
  //   compatibility
  bool correct_transform_coords_{false};
private:
  arm_compute::CLBBoxTransform bbox_transform_;
  bool first_run_ = true, second_run_ = true;
  CLContext::deleted_unique_ptr<const OpenCLTensor<T>> roi_in_, delta_in_, iminfo_in_;
};

template <typename T, typename Activation>
bool CLBBoxTransformOp<T, Activation>::RunOnDevice() {
  auto *RoiInblob = OperatorBase::Inputs()[0];
  auto *DeltaInblob = OperatorBase::Inputs()[1];
  auto *IminfoInblob = OperatorBase::Inputs()[2];
  roi_in_ = CLContext::getCLTensor<T>(RoiInblob, roi_in_.release());
  delta_in_ = CLContext::getCLTensor<T>(DeltaInblob, delta_in_.release());
  iminfo_in_ = CLContext::getCLTensor<T>(IminfoInblob, iminfo_in_.release());

  OpenCLTensor<T> *Y =
    OperatorBase::Outputs()[0]->template GetMutable<OpenCLTensor<T>>();

  const int N = roi_in_->dim32(0);

  CAFFE_ENFORCE_EQ(roi_in_->ndim(), 2);
  CAFFE_ENFORCE(roi_in_->dim32(1) == 4 || roi_in->dim32(1) == 5);

  CAFFE_ENFORCE_EQ(delta_in_.ndim(), 2);
  CAFFE_ENFORCE_EQ(delta_in_.dim32(0), N);
  CAFFE_ENFORCE_EQ(delta_in_.dim32(1) % 4, 0);
  const int num_classes = delta_in_.dim32(1) / 4;

  CAFFE_ENFORCE_EQ(iminfo_in_.ndim(), 2);
  CAFFE_ENFORCE_EQ(iminfo_in_.dim32(1), 3);
  const int batch_size = iminfo_in_.dim32(0);
  CAFFE_ENFORCE_EQ(batch_size, 1);

  DCHECK_EQ(weights_.size(), 4);

  arm_compute::BoundingBoxTransformInfo bbox_info(apply_scale, weights_, utils::BBOX_XFORM_CLIP_DEFAULT);

  if (first_run_) {
    first_run_ = false;

    TensorCPU fakeX;
    fakeX.Resize(roi_in_->dims());
    TensorCPU fakeY;
    ConvPoolOpBase<CLContext>::SetOutputSize(fakeX, &fakeY, delta_in_->dim32(0));
    Y->ResizeLike(fakeY);

    bbox_transform.configure(roi_in_->get_underlying(), Y->get_underlying(), delta_in_->get_underlying(), bbox_info);
  } else if (second_run_) {
    roi_in_->lazy_allocate(RoiInblob, second_run_, true);
    delta_in_->lazy_allocate(DeltaInblob, second_run_, second_run_);
    iminfo_in_->lazy_allocate(IminfoInblob, second_run_, second_run_);
    second_run_ = false;
    Y->allocate();
    // Run
    bbox_transform.run();
  } else {
    TensorCPU fakeX;
    fakeX.Resize(roi_in_->dims());
    TensorCPU fakeY;
    ConvPoolOpBase<CLContext>::SetOutputSize(fakeX, &fakeY, delta_in_->dim32(0));
    bool need_allocation = Y->ResizeLike(fakeY, true);
    // Configure
    bbox_transform.configure(roi_in_->get_underlying(), Y->get_underlying(), delta_in_->get_underlying(), bbox_info);
    // Allocate
    roi_in_->lazy_allocate(RoiInblob, second_run_, true);
    delta_in_->lazy_allocate(DeltaInblob, second_run_, true);
    iminfo_in_->lazy_allocate(IminfoInblob, second_run_, true);
    if (need_allocation) {
      Y->allocate();
    }
    // Run
    bbox_transform.run();
 }
  return true;
}

REGISTER_CL_OPERATOR(Conv, CLBBoxTransformOp<DataType>);

} // namespace caffe2
