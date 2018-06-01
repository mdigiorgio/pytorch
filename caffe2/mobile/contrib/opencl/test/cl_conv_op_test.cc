#include "cl_operator_test.h"
#include "caffe2/core/timer.h"

namespace caffe2 {

constexpr float tol = 3.0e-2;

TEST(OPENCLOperatorTest, Conv) {

 Workspace ws;
 auto channel_in = 16;
 auto channel_out = 16;
 auto spatial = 16; // --> 2x2 w no padding, all values 9
 auto kern = 3;

 PopulateCPUBlob(&ws, true, "cpu_X", {1, channel_in, spatial, spatial}, 1337);
 PopulateCPUBlob(&ws, true, "W", {channel_out, channel_in, kern, kern}, 1337);
 PopulateCPUBlob(&ws, false, "b", {channel_out}, 0);

#define ADD_CONV_ARGS                                                   \
 {                                                                      \
   ADD_ARG((*def), "kernel", i, kern);                                  \
   ADD_ARG((*def), "stride", i, 1);                                     \
   ADD_ARG((*def), "pad", i, 0);                                        \
   ADD_ARG((*def), "order", s, "NCHW");                                 \
 }

  NetDef cpu_net;
  {
    OperatorDef* def = AddOp(&cpu_net, "Conv", {"cpu_X", "W", "b"}, {"ref_Y"});
    def->set_name("cpu_conv");
    ADD_CONV_ARGS;
  }

  NetDef gpu_net;
  gpu_net.set_type("opencl");
  {

    OperatorDef* def = AddOp(&gpu_net, "Conv", {"cpu_X", "W", "b"}, {"gpu_Y"});
    MAKE_OPENCL_OPERATOR(def);
    ADD_CONV_ARGS;
  }

#undef ADD_CONV_ARGS

  compareNetResult4D(ws, cpu_net, gpu_net, "ref_Y", "gpu_Y", tol);

}

TEST(OPENCLOperatorTest, ConvReluConv) {

  Workspace ws;
  auto channel_in = 16;
  auto channel_out = 16;
  auto spatial = 32;
  auto kern = 3;

  PopulateCPUBlob(&ws, true, "cpu_X", {1, channel_in, spatial, spatial}, 1337);
  PopulateCPUBlob(&ws, true, "W", {channel_out, channel_in, kern, kern}, 1337);
  PopulateCPUBlob(&ws, false, "b", {channel_out}, 0);
  PopulateCPUBlob(&ws, true, "W2", {channel_out, channel_in, kern, kern});
  PopulateCPUBlob(&ws, true, "b2", {channel_out});

#define ADD_CONV_ARGS                                                   \
  {                                                                     \
    ADD_ARG((*def), "kernel", i, kern);                                 \
    ADD_ARG((*def), "stride", i, 1);                                    \
    ADD_ARG((*def), "pad", i, 1);                                       \
    ADD_ARG((*def), "order", s, "NCHW");                                \
  }

  NetDef cpu_net;
  {
    OperatorDef* def = AddOp(&cpu_net, "Conv", {"cpu_X", "W", "b"}, {"ref_Y"});
    def->set_name("cpu_conv");
    ADD_CONV_ARGS;
  }
  {
    OperatorDef* def = AddOp(&cpu_net, "Relu", {"ref_Y"}, {"ref_relu"});
  }
  {
    OperatorDef* def = AddOp(&cpu_net, "Conv", {"ref_relu", "W2", "b2"}, {"ref_Y2"});
    ADD_CONV_ARGS;
  }

  NetDef gpu_net;
  gpu_net.set_type("opencl");
  {
    OperatorDef* def = AddOp(&gpu_net, "Conv", {"cpu_X", "W", "b"}, {"gpu_Y"});
    MAKE_OPENCL_OPERATOR(def);
    ADD_CONV_ARGS;
  }
  {
    OperatorDef* def = AddOp(&gpu_net, "Relu", {"gpu_Y"}, {"gpu_relu"});
    MAKE_OPENCL_OPERATOR(def);
  }
  {
    OperatorDef* def = AddOp(&gpu_net, "Conv", {"gpu_relu", "W2", "b2"}, {"gpu_Y2"});
    MAKE_OPENCL_OPERATOR(def);
    ADD_CONV_ARGS;
  }

#undef ADD_CONV_ARGS

  compareNetResult4D(ws, cpu_net, gpu_net, "ref_Y2", "gpu_Y2", 4.0e-02);

}

TEST(OPENCLOperatorTest, ConvBenchmark) {

 Workspace ws;
 auto channel_in = 4;
 auto channel_out = 4;
 auto spatial = 10;
 auto kern = 3;
 long long iters = 2;

 PopulateCPUBlob(&ws, false, "cpu_X", {1, channel_in, spatial, spatial}, 1, 0, 0.1);

#define ADD_CONV_ARGS(_def)                                                        \
 {                                                                                 \
    ADD_ARG((*_def), "kernel", i, kern);                                           \
    ADD_ARG((*_def), "stride", i, 1);                                              \
    ADD_ARG((*_def), "pad", i, 0);                                                 \
    ADD_ARG((*_def), "order", s, "NCHW");                                          \
  }

  NetDef gpu_net;
  NetDef cpu_net;
  gpu_net.set_type("opencl");

  std::string prev_out = "cpu_X";
  for (auto i = 0; i < iters; ++i) {
    std::string weightName = "W" + to_string(i);
    std::string biasName = "b" + to_string(i);
    std::string output = "conv" + to_string(i);
    PopulateCPUBlob(&ws, false, weightName, {channel_out, channel_in, kern, kern}, 1);
    PopulateCPUBlob(&ws, false, biasName, {channel_out}, 0);
    OperatorDef* def = AddOp(&gpu_net, "Conv", {prev_out, weightName, biasName}, {output});
    if (i == 0) {
      OperatorDef* def2 = AddOp(&cpu_net, "Conv", {prev_out, weightName, biasName}, {"cpu" + output});
    ADD_CONV_ARGS(def2);
    } else {
      OperatorDef* def2 = AddOp(&cpu_net, "Conv", {"cpu" + prev_out, weightName, biasName}, {"cpu" + output});
    ADD_CONV_ARGS(def2);
    }
    prev_out = output;
    MAKE_OPENCL_OPERATOR(def);
    ADD_CONV_ARGS(def);
  }

#undef ADD_CONV_ARGS

  compareNetResult4D(ws, cpu_net, gpu_net, "cpu" + prev_out, prev_out, tol);

}

TEST(OPENCLOperatorTest, GroupedConv) {

 for (auto channel_in: std::vector<int>({16, 24, 32, 48})) {
   for (auto channel_out: std::vector<int>({16, 24, 32, 48})) {
     for (auto groups: std::vector<int>({4, 8})) {
       Workspace ws;

       LOG(ERROR) << "[C2DEBUG] c2: " << channel_in << " " << channel_out << " " << groups;
       auto spatial = 16; // --> 2x2 w no padding, all values 9
       auto kern = 3;
       PopulateCPUBlob(&ws, true, "cpu_X", {1, channel_in, spatial, spatial}, 1337);
       PopulateCPUBlob(&ws, true, "W", {channel_out, channel_in / groups, kern, kern}, 1337);
       PopulateCPUBlob(&ws, false, "b", {channel_out}, 0);

#define ADD_CONV_ARGS                           \
       {                                        \
         ADD_ARG((*def), "kernel", i, kern);    \
         ADD_ARG((*def), "stride", i, 1);       \
         ADD_ARG((*def), "group", i, groups);   \
         ADD_ARG((*def), "pad", i, 0);          \
         ADD_ARG((*def), "order", s, "NCHW");   \
       }

       NetDef cpu_net;
       {
         OperatorDef* def = AddOp(&cpu_net, "Conv", {"cpu_X", "W", "b"}, {"ref_Y"});
         def->set_name("cpu_conv");
         ADD_CONV_ARGS;
       }

       NetDef gpu_net;
       gpu_net.set_type("opencl");
       {

         OperatorDef* def = AddOp(&gpu_net, "Conv", {"cpu_X", "W", "b"}, {"gpu_Y"});
         MAKE_OPENCL_OPERATOR(def);
         ADD_CONV_ARGS;
       }

#undef ADD_CONV_ARGS

       compareNetResult4D(ws, cpu_net, gpu_net, "ref_Y", "gpu_Y", tol);
     }
   }
 }

}


TEST(OPENCLOperatorTest, DepthwiseConv) {

 for (auto channel_in: std::vector<int>({3, 9, 16, 48, 124})) {
   Workspace ws;
   auto channel_out = channel_in;
   auto groups = channel_in;
   auto spatial = 16; // --> 2x2 w no padding, all values 9
   auto kern = 3;

   PopulateCPUBlob(&ws, true, "cpu_X", {1, channel_in, spatial, spatial}, 1337);
   PopulateCPUBlob(&ws, true, "W", {channel_out, 1, kern, kern}, 1337);
   PopulateCPUBlob(&ws, false, "b", {channel_out}, 0);

#define ADD_CONV_ARGS                           \
   {                                            \
     ADD_ARG((*def), "kernel", i, kern);        \
     ADD_ARG((*def), "stride", i, 1);           \
     ADD_ARG((*def), "group", i, groups);       \
     ADD_ARG((*def), "pad", i, 0);              \
     ADD_ARG((*def), "order", s, "NCHW");       \
   }

   NetDef cpu_net;
   {
     OperatorDef* def = AddOp(&cpu_net, "Conv", {"cpu_X", "W", "b"}, {"ref_Y"});
     def->set_name("cpu_conv");
     ADD_CONV_ARGS;
   }

   NetDef gpu_net;
   gpu_net.set_type("opencl");
   {

     OperatorDef* def = AddOp(&gpu_net, "Conv", {"cpu_X", "W", "b"}, {"gpu_Y"});
     MAKE_OPENCL_OPERATOR(def);
     ADD_CONV_ARGS;
   }

#undef ADD_CONV_ARGS

   compareNetResult4D(ws, cpu_net, gpu_net, "ref_Y", "gpu_Y", tol);
 }

}

} // namespace caffe2

