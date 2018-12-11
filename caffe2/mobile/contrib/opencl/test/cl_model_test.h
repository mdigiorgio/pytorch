#include "caffe2/mobile/contrib/opencl/core/context.h"
#include "caffe2/mobile/contrib/opencl/test/cl_operator_test.h"
#include "caffe2/mobile/contrib/opencl/core/rewrite_net.h"
#include <gtest/gtest.h>

#include "caffe2/core/operator.h"
#include "caffe2/core/workspace.h"
#include "caffe2/opt/mobile.h"
#include <unordered_set>

C10_DEFINE_int(warmup, 3, "The number of iterations to warm up.");
C10_DEFINE_int(iter, 10, "The number of iterations to run.");
C10_DEFINE_bool(
    run_individual,
    true,
    "Whether to benchmark individual operators.");


constexpr float tol = 0.03;
namespace caffe2 {
  void benchmarkModel(std::string init_net_pb, std::string predict_net_pb, std::string input_name, std::vector<int64_t> input_dims, std::string net_name="benchmark_net", std::unordered_set<std::string> cpu_ops = std::unordered_set<std::string>({})) {
    unique_ptr<Workspace> ws(new Workspace());
    NetDef init_net_def;
    CAFFE_ENFORCE(ReadProtoFromFile(init_net_pb, &init_net_def));
    CAFFE_ENFORCE(ws->RunNetOnce(init_net_def));
    NetDef predict_net_def, predict_net_def_gpu, tmp_net_def;
    CAFFE_ENFORCE(ReadProtoFromFile(predict_net_pb, &tmp_net_def));
    PopulateCPUBlob(ws.get(), true, input_name, input_dims);
    LOG(ERROR) << "[C2DEBUG] rewriting OpenCL net";
    predict_net_def.CopyFrom(tmp_net_def);
    predict_net_def.clear_op();
    predict_net_def.clear_external_output();
    for (auto i = 0; i < tmp_net_def.op_size(); ++i) {
      auto op = predict_net_def.add_op();
      op->CopyFrom(tmp_net_def.op(i));
    }
    auto* output_ = predict_net_def.add_external_output();
    *output_ = predict_net_def.op()[predict_net_def.op().size() - 1].output()[0];
    //predict_net_def_gpu = opt::tryConvertToACLOpenCL(predict_net_def, false, cpu_ops);
    tryConvertToOpenCL(predict_net_def, &predict_net_def_gpu, false, cpu_ops);
    LOG(ERROR) << "[C2DEBUG] predict_net_def_gpu.size(): " << predict_net_def_gpu.op().size();
    // change the name of last op
    auto index = predict_net_def_gpu.op().size() - 1;
    dumpDefForOpenCL(predict_net_def_gpu);
    auto last_blob = predict_net_def_gpu.op()[index].output()[0];
    auto op = predict_net_def_gpu.mutable_op(index);
    auto output = op->mutable_output(0);
    *output = last_blob + "_gpu";
    for (auto i = 0; i < predict_net_def_gpu.external_output_size(); ++i) {
      auto out = predict_net_def_gpu.mutable_external_output(i);
      if (*out == last_blob) {
        *out = last_blob + "_gpu";
      }
    }

  compareNetResult4D(*ws, predict_net_def, predict_net_def_gpu, last_blob, last_blob + "_gpu");
  LOG(ERROR) << "[C2DEBUG] after compareNetResult4D";
  NetBase* net = ws->CreateNet(predict_net_def_gpu);
  LOG(ERROR) << "[C2DEBUG] Benchmarking OpenCL Net";
  net->TEST_Benchmark(FLAGS_warmup, FLAGS_iter, FLAGS_run_individual);
  // Test CPU
  for (auto i = 0; i < predict_net_def.op().size(); ++i) {
    auto op = predict_net_def.mutable_op(i);
    if (std::find(cpu_ops.begin(), cpu_ops.end(), op->type()) == cpu_ops.end()) {
      op->mutable_device_option()->set_device_type(PROTO_CPU);
    }
  }
  predict_net_def.set_type("simple");
  predict_net_def.set_name("cpu_net");
  net = ws->CreateNet(predict_net_def);
  LOG(INFO) << "[C2DEBUG] Benchmarking CPU Net";
  net->TEST_Benchmark(FLAGS_warmup, FLAGS_iter, FLAGS_run_individual);

  }
} // namespace caffe2
