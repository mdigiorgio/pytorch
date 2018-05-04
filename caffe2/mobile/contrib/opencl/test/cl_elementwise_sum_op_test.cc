#include "cl_operator_test.h"

namespace caffe2 {

TEST(OPENCLOperatorTest, Sum) {
  Workspace ws;
  int N = 28;
  int D = 128;
  PopulateCPUBlob(&ws, true, "cpu_X", {N, D}, 1);
  PopulateCPUBlob(&ws, true, "cpu_Y", {N, D}, 1);

  NetDef cpu_net;
  {
    AddOp(&cpu_net, "Sum", {"cpu_X", "cpu_Y"}, {"ref_Y"});
  }

  NetDef gpu_net;
  gpu_net.set_type("opencl");
  {
    OperatorDef* def = AddOp(&gpu_net, "Sum", {"cpu_X", "cpu_Y"}, {"gpu_Y"});
    MAKE_OPENCL_OPERATOR(def);
  }

  compareNetResult(ws, cpu_net, gpu_net);
}

} // namespace caffe2
