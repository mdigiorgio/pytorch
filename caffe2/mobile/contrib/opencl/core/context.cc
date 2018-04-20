#include "context.h"

#include "caffe2/core/allocator.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

CAFFE_KNOWN_TYPE(OpenCLTensor<half>);
CAFFE_KNOWN_TYPE(Tensor<CLContext>);

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

} // namespace caffe2
