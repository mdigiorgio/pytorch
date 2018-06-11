#include "caffe2/mobile/contrib/opencl/core/context.h"
#include "caffe2/mobile/contrib/opencl/core/operator.h"

#include "caffe2/mobile/contrib/opencl/operators/activation_ops.h"
#include "caffe2/operators/relu_op.h"

namespace caffe2 {

template <typename T>
bool CLReluOp<T>::RunOnDevice() {

  auto *Xblob = OperatorBase::Inputs()[0];
  X_ = CLContext::getCLTensor<T>(Xblob, X_.release());

  OpenCLTensor<T> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<OpenCLTensor<T>>();

  if (first_run_) {
    first_run_ = false;
    if (Y->get_underlying() != X_->get_underlying())
    {
      Y->ResizeLike(*X_);
    }
    relu_layer_.configure(
        X_->get_underlying(), Y->get_underlying(),
        arm_compute::ActivationLayerInfo(
          arm_compute::ActivationLayerInfo::ActivationFunction::RELU));

  } else if (second_run_) {
    X_->lazy_allocate(Xblob, second_run_, true);
    second_run_ = false;
    // in place activation, do not need to allocate new memory
    if (Y->get_underlying() != X_->get_underlying()) {
      Y->ResizeLike(*X_);
      Y->allocate();
    }
    relu_layer_.run();
  } else {
    bool need_allocation = false;
    if (Y->get_underlying() != X_->get_underlying()) {
      need_allocation = Y->ResizeLike(*X_, true);
    }
    // Configure
    relu_layer_.configure(
        X_->get_underlying(), Y->get_underlying(),
        arm_compute::ActivationLayerInfo(
          arm_compute::ActivationLayerInfo::ActivationFunction::RELU));
    // Allocate
    X_->lazy_allocate(Xblob, second_run_, true);
    if (need_allocation) {
      Y->allocate();
    }
    // Run
    relu_layer_.run();
  }

  return true;
}

REGISTER_CL_OPERATOR(Relu, CLReluOp<DataType>);

template <typename T>
bool CLSigmoidOp<T>::RunOnDevice() {

  auto *Xblob = OperatorBase::Inputs()[0];
  X_ = CLContext::getCLTensor<T>(Xblob, X_.release());

  OpenCLTensor<T> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<OpenCLTensor<T>>();
  if (first_run_) {
    first_run_ = false;

    if (Y->get_underlying() != X_->get_underlying())
    {
        Y->ResizeLike(*X_);
    }

    sigmoid_layer_.configure(
      X_->get_underlying(), Y->get_underlying(),
      arm_compute::ActivationLayerInfo(
          arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC));
  } else if (second_run_) {
    X_->lazy_allocate(Xblob, second_run_, true);
    second_run_ = false;
    // in place activation, do not need to allocate new memory
    if (Y->get_underlying() != X_->get_underlying()) {
      Y->ResizeLike(*X_);
      Y->allocate();
    }
    sigmoid_layer_.run();
  } else {
    // Configure
    sigmoid_layer_.configure(
      X_->get_underlying(), Y->get_underlying(),
      arm_compute::ActivationLayerInfo(
          arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC));
    // Allocate
    X_->lazy_allocate(Xblob, second_run_, true);
    bool need_allocation = false;
    if (Y->get_underlying() != X_->get_underlying())
    {
      need_allocation = Y->ResizeLike(*X_, true);
    }
    if (need_allocation) {
      Y->allocate();
    }
    // Run
    sigmoid_layer_.run();
  }

  return true;
}

REGISTER_CL_OPERATOR(Sigmoid, CLSigmoidOp<DataType>);

} // namespace caffe2
