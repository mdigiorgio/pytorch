#include "context.h"

#include "caffe2/core/allocator.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

CAFFE_KNOWN_TYPE(OpenCLTensor<DataType>);

bool CLContext::initialized = false;

CLContext::CLContext() {
  CAFFE_ENFORCE(arm_compute::opencl_is_available());
  if(!initialized) {
    arm_compute::CLScheduler::get().default_init();
    initialized = true;
  }
}

void EventCreateOPENCL(const DeviceOption & /* unused */,
                       Event * /* unused */) {}
void EventRecordOPENCL(Event * /* unused */, const void * /* unused */,
                       const char * /* unused */) {}
void EventWaitOPENCLOPENCL(const Event * /* unused */, void * /* unused */) {}
void EventFinishOPENCL(const Event * /* unused */) {}
void EventResetOPENCL(Event * /* unused */) {}

REGISTER_EVENT_CREATE_FUNCTION(OPENCL, EventCreateOPENCL);
REGISTER_EVENT_RECORD_FUNCTION(OPENCL, EventRecordOPENCL);
REGISTER_EVENT_WAIT_FUNCTION(OPENCL, OPENCL, EventWaitOPENCLOPENCL);
REGISTER_EVENT_FINISH_FUNCTION(OPENCL, EventFinishOPENCL);
REGISTER_EVENT_RESET_FUNCTION(OPENCL, EventResetOPENCL);

std::pair<arm_compute::TensorShape, arm_compute::Coordinates> compute_output_shape(arm_compute::TensorShape input_shape, unsigned int num_splits, unsigned int axis, unsigned int idx)
{
  ARM_COMPUTE_ERROR_ON(axis >= input_shape.num_dimensions());
  ARM_COMPUTE_ERROR_ON_MSG(input_shape[axis] % num_splits, "Split should be exact");

  const unsigned int split_size = input_shape[axis] / num_splits;

  arm_compute::TensorShape output_shape = input_shape;
  output_shape.set(axis, split_size);

  arm_compute::Coordinates coords;
  coords.set(axis, idx * split_size);

  return std::make_pair(output_shape, coords);
}

} // namespace caffe2
