#include "cl_operator_test.h"

namespace caffe2 {

TEST(OPENCLOperatorTest, Sigmoid) {
  Workspace ws;

  PopulateCPUBlob(&ws, true, "cpu_X", {1, 4, 4, 4});

  NetDef cpu_net;
  {
    AddOp(&cpu_net, "Sigmoid", {"cpu_X"}, {"ref_Y"});
  }

  NetDef gpu_net;
  gpu_net.set_type("opengl");
  {
    OperatorDef* def = AddOp(&gpu_net, "Sigmoid", {"cpu_X"}, {"gpu_Y"});
    MAKE_OPENCL_OPERATOR(def);
  }
  compareNetResult(ws, cpu_net, gpu_net);

}

TEST(OPENCLOperatorTest, ReLU) {
  Workspace ws;

  PopulateCPUBlob(&ws, true, "cpu_X", {1, 4, 4, 4});

  NetDef cpu_net;
  {
    AddOp(&cpu_net, "Relu", {"cpu_X"}, {"ref_Y"});
  }

  NetDef gpu_net;
  gpu_net.set_type("opengl");
  {
    OperatorDef* def = AddOp(&gpu_net, "Relu", {"cpu_X"}, {"gpu_Y"});
    MAKE_OPENCL_OPERATOR(def);
  }

  compareNetResult(ws, cpu_net, gpu_net);
}

TEST(OPENCLOperatorTest, SigmoidTwice) {
  Workspace ws;

  PopulateCPUBlob(&ws, true, "cpu_X", {1, 4, 4, 4});

  NetDef cpu_net;
  {
    AddOp(&cpu_net, "Sigmoid", {"cpu_X"}, {"ref_Y1"});
    AddOp(&cpu_net, "Sigmoid", {"ref_Y1"}, {"ref_Y2"});
  }

  NetDef gpu_net;
  gpu_net.set_type("opengl");
  {
    OperatorDef* def = AddOp(&gpu_net, "Sigmoid", {"cpu_X"}, {"gpu_Y1"});
    MAKE_OPENCL_OPERATOR(def);
  }
  {
    OperatorDef* def = AddOp(&gpu_net, "Sigmoid", {"gpu_Y1"}, {"gpu_Y2"});
    MAKE_OPENCL_OPERATOR(def);
  }

  compareNetResult(ws, cpu_net, gpu_net, "ref_Y2", "gpu_Y2");
}

} // namespace caffe2
