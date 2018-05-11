#include "caffe2/opt/mobile.h"
#include "caffe2/core/logging.h"
#include "caffe2/opt/converter.h"
#include "caffe2/opt/fusion.h"

namespace caffe2 {
namespace opt {

using namespace nom;

void addNNPACK(repr::NNModule* nn, bool low_memory) {
  for (auto node : nn->dataFlow.getMutableNodes()) {
    auto* nodeData = node->data().get(); // Let graph retain ownership.

    // Skip blobs.
    if (!isa<nom::repr::NeuralNetOperator>(nodeData)) {
      continue;
    }

    // Check if it is a convolution.
    auto nnOp = dyn_cast<nom::repr::NeuralNetOperator>(nodeData);
    if (!isa<nom::repr::Conv>(nnOp)) {
      continue;
    }

    // Requires X, W, b for NNPACK
    if (node->getInEdges().size() < 3) {
      continue;
    }

    std::string engine = "NNPACK";

    // Now do some specific checks to see if an NNPACK engine is correct.
    bool validTransformCandidate = true;
    auto conv = dyn_cast<nom::repr::Conv>(nnOp);

    if (conv->getLayout() != nom::repr::Conv::NNLayout::NCHW) {
      continue;
    }

    // NNPACK only supports stride == 1
    for (auto stride : conv->getStrides()) {
      if (stride != 1) {
        validTransformCandidate = false;
        break;
      }
    }
    if (!validTransformCandidate) {
      continue;
    }

    // NNPACK only supports 2DConv.
    const auto& kernelShape = conv->getKernelShape();
    if (kernelShape.size() != 2) {
      continue;
    }

    // Kx1 and 1xK convs are inefficient in NNPACK.
    if (kernelShape[0] != kernelShape[1]) {
      if (kernelShape[0] == 1 || kernelShape[1] == 1) {
        continue;
      }
    }

    // We're good to use our engine.
    auto annotation = conv->getMutableAnnotation();
    if (!annotation || !isa<Caffe2Annotation>(annotation)) {
      continue;
    }
    auto* op = dyn_cast<Caffe2Annotation>(annotation)->getMutableOperatorDef();
    op->set_engine(engine);
    if (!low_memory) {
      auto* precompute_argument = op->add_arg();
      precompute_argument->set_name("convolution_transform_strategy");
      precompute_argument->set_s("PRECOMPUTE");
    }
  }
}

namespace {

inline bool isNNPACKConvReluEfficient(
    const std::string& algo,
    const repr::Conv& conv) {
  if (algo == "AUTO" || algo == "") {
    for (auto stride : conv.getStrides()) {
      if (stride > 1) {
        return false;
      }
    }
    for (auto kernel : conv.getKernelShape()) {
      if (kernel < 2) {
        return false;
      }
    }
  } else if (!(algo == "WINOGRAD" || algo == "WINOGRAD_FP16" ||
               algo == "FT8x8" || algo == "FT16x16")) {
    return false;
  }
  return true;
}

} // namespace

void fuseNNPACKConvRelu(repr::NNModule* nn) {
  auto should_fuse = [](const repr::Conv& conv) {
    const auto annotation = conv.getAnnotation();
    if (!annotation || !isa<Caffe2Annotation>(annotation)) {
      return false;
    }
    const auto* op = dyn_cast<Caffe2Annotation>(annotation)->getOperatorDef();

    // We only want to fuse for fast NNPACK convs
    if (op->engine() != "NNPACK") {
      return false;
    }
    caffe2::string algo = "AUTO";
    for (const auto arg : op->arg()) {
      if (arg.name() == "algo") {
        algo = arg.s();
      }
    }
    if (!isNNPACKConvReluEfficient(algo, conv)) {
      return false;
    }
    return true;
  };

  auto postprocess = [](repr::NNGraph::NodeRef conv_node) {
    auto conv = repr::nn::get<repr::Conv>(conv_node);
    auto annotation = conv->getMutableAnnotation();
    if (!annotation || !isa<Caffe2Annotation>(annotation)) {
      return;
    }
    auto* op = dyn_cast<Caffe2Annotation>(annotation)->getMutableOperatorDef();
    auto* arg = op->add_arg();
    arg->set_name("activation");
    arg->set_s("Relu");
  };

  fuseActivation<repr::Conv, repr::Relu>(nn, should_fuse, postprocess);
}

void runOpenCLFusion(repr::NNModule* nn) {
  fuseConvRelu(nn);
  fuseConvSigmoid(nn);
}

caffe2::NetDef tryConvertToACLOpenCL(caffe2::NetDef net, bool runFusion, std::unordered_set<std::string> cpuOps) {
  auto nn = convertToNNModule(net);
  if (runFusion) {
    runOpenCLFusion(&nn);
  }
  for (auto node: nn.dataFlow.getMutableNodes()) {
    auto* node_data = node->data().get();

    // skip blobs
    if (!isa<nom::repr::NeuralNetOperator>(node_data)) {
      continue;
    }

    auto nn_op = dyn_cast<nom::repr::NeuralNetOperator>(node_data);
    if (cpuOps.count(nn_op->getName()) > 0) {
      // CPU Op
      std::vector<repr::NNGraph::NodeRef> copy_from_cl;
      auto inputs = repr::nn::getInputs(node);
      for (auto i = 0; i < inputs.size(); ++i) {
        // if the blob is produced by a OpenCL Op, we need to insert CopyFromCL Op
        if (repr::nn::hasProducer(inputs[i])) {
          auto producer_node = repr::nn::getProducer(inputs[i]);
          // if it is produced by OpenCL Op
          auto producer_data = producer_node->data().get();
          auto producer_op = dyn_cast<nom::repr::NeuralNetOperator>(producer_data);
          if (cpuOps.count(producer_op->getName()) == 0) {
            copy_from_cl.push_back(inputs[i]);
          }
        }
      }
      if (copy_from_cl.size() > 0) {
        auto copy_node = repr::nn::insertOp<repr::CopyFromCL>(nn.dataFlow, copy_from_cl[0], node);
        for (auto i = 1; i < copy_from_cl.size(); ++i) {
          auto new_node = repr::nn::insertOp<repr::CopyFromCL>(nn.dataFlow, copy_from_cl[i], node);
          nn.dataFlow.replaceNode(new_node, copy_node);
        }
      }
    } else {
      // OpenCL Op
      auto inputs = repr::nn::getInputs(node);
      for (auto i = 0; i < inputs.size(); ++i) {
        auto new_node = nn.dataFlow.createNode(make_unique<repr::Tensor>(repr::nn::get<repr::NeuralNetData>(inputs[i])->getName() + "_"));
        nn.dataFlow.replaceNode(inputs[i], new_node);
      }
      auto outputs = repr::nn::getOutputs(node);
      for (auto i = 0; i < outputs.size(); ++i) {
        if (repr::nn::hasConsumer(outputs[i])) {
          auto new_node = nn.dataFlow.createNode(make_unique<repr::Tensor>(repr::nn::get<repr::NeuralNetData>(outputs[i])->getName() + "_"));
          nn.dataFlow.replaceNode(outputs[i], new_node);
        }
      }
    }
  }
  return convertToCaffe2Proto(nn, net);
}

} // namespace opt
} // namespace caffe2
