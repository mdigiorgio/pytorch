#include "cl_operator_test.h"

namespace caffe2 {

TEST(OPENCLOperatorTest, Softmax) {

  Workspace ws;
  int N = 1;
  int D = 128;
  PopulateCPUBlob(&ws, true, "cpu_X", {N, D}, 1);

  NetDef cpu_net;
  {
    AddOp(&cpu_net, "Softmax", {"cpu_X"}, {"ref_Y"});
  }

  NetDef gpu_net;
  gpu_net.set_type("opencl");
  {
    OperatorDef* def = AddOp(&gpu_net, "Softmax", {"cpu_X"}, {"gpu_Y"});
    MAKE_OPENCL_OPERATOR(def);
  }

  compareNetResult(ws, cpu_net, gpu_net);

}

} // namespace caffe2
