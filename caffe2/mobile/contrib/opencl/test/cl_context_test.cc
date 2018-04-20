#include "caffe2/mobile/contrib/opencl/core/context.h"
#include <gtest/gtest.h>

namespace caffe2 {

TEST(OPENCLContextTest, Initialization) {
  auto context = new CLContext();
  delete context;
}

} // namespace caffe2
