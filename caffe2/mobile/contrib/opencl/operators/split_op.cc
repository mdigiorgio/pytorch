#include "caffe2/mobile/contrib/opencl/core/context.h"
#include "caffe2/mobile/contrib/opencl/core/operator.h"

namespace caffe2 {

namespace {
inline int get_acl_axis(int axis, StorageOrder order, int num_dimensions) {
  if (num_dimensions < 4) {
      return axis;
  }
  if (order == StorageOrder::NCHW) {
    switch(axis) {
      case 0:
        return 3;
      case 1:
        return 2;
      case 2:
        return 1;
      case 3:
        return 0;
      default:
        CAFFE_THROW("axis must be between 0 and 3");
    }
  } else {
    // NHWC case
    switch(axis) {
      case 0:
        return 3;
      case 1:
        return 1;
      case 2:
        return 0;
      case 3:
        return 2;
      default:
        CAFFE_THROW("axis must be between 0 and 3");
    }
  }
}
} // namespace

template <typename T>
class CLSplitOp final : public Operator<CLContext> {
public:
  CLSplitOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<CLContext>(operator_def, ws),
        axis_(OperatorBase::GetSingleArgument<int>("axis", 3)),
        order_(StringToStorageOrder(OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
  }
  virtual ~CLSplitOp() noexcept {}
  USE_OPERATOR_FUNCTIONS(CLContext);
  bool RunOnDevice() override;
private:
  arm_compute::CLSplit split_layer_;
  bool first_run_ = true, second_run_ = true;
  CLContext::deleted_unique_ptr<const OpenCLTensor<T>> X_;
  int axis_;
  StorageOrder order_;
};

template <typename T> bool CLSplitOp<T>::RunOnDevice() {

  auto Xblob = OperatorBase::Inputs()[0];

  X_ = CLContext::getCLTensor<T>(Xblob, X_.release());

  CAFFE_ENFORCE_LT(axis_, X_->ndim());

  const int acl_axis = get_acl_axis(axis_, order_, X_->ndim());
  const int num_splits = OperatorBase::Outputs().size();
  CAFFE_ENFORCE_GE(num_splits, 2);

  vector<TIndex> output_dims(X_->dims());
  output_dims[axis_] /= num_splits;

  std::vector<OpenCLTensor<T> *> outputsTensor;
  std::vector<arm_compute::ICLTensor*> cl_outputs;
  std::vector<bool> need_allocation;
  for (int i = 0; i < num_splits; ++i) {
    OpenCLTensor<T> *Y =
        OperatorBase::Outputs()[i]->template GetMutable<OpenCLTensor<T>>();
    need_allocation.push_back(Y->Resize(output_dims));
    outputsTensor.push_back(Y);
    cl_outputs.push_back(Y->get_underlying());
  }

  if (first_run_) {
    first_run_ = false;
    // Configure
    split_layer_.configure(X_->get_underlying(), cl_outputs, acl_axis);
  } else if (second_run_) {
    // Allocate
    X_->lazy_allocate(Xblob, second_run_, true);
    for (int i = 0; i < num_splits; ++i) {
      outputsTensor[i]->allocate();
    }
    second_run_ = false;
    // Run
    split_layer_.run();
  } else {
    // Configure
    split_layer_.configure(X_->get_underlying(), cl_outputs, acl_axis);
    // Allocate
    X_->lazy_allocate(Xblob, second_run_, true);
    for (int i = 0; i < num_splits; ++i) {
      if (need_allocation[i]) {
        outputsTensor[i]->allocate();
      }
    }
    // Run
    split_layer_.run();
  }

  return true;
}

REGISTER_CL_OPERATOR(Split, CLSplitOp<DataType>);

} // namespace caffe2
