#include "caffe2/opt/converter.h"
#include "caffe2/opt/fusion.h"
#include "caffe2/opt/passes.h"

namespace caffe2 {
namespace opt {

using namespace nom;

// template <typename OperationT, typename ActivationT, typename FusedT>
// bool fusionHelper(std::string fusedName, repr::NNGraph* g) {
//   for (auto& node_pair : repr::nn::dataIterator<OperationT>(*g)) {
//     repr::NNGraph::NodeRef node;
//     OperationT* operation;
//     std::tie(operation, node) = node_pair;

//     // Single output check (intrinsic to a operation, but we double check)
//     auto outputs = repr::nn::getOutputs(node);
//     if (outputs.size() != 1) {
//       continue;
//     }
//     auto tensorNode = outputs.front();

//     // Single user check.
//     auto consumers = repr::nn::getConsumers(tensorNode);
//     if (consumers.size() != 1) {
//       continue;
//     }

//     // Followed by Activation check.
//     auto* nextNode = consumers.front();
//     if (!repr::nn::is<ActivationT>(nextNode)) {
//       continue;
//     }

//     // Naming for operationenience
//     auto* operationNode = node;
//     auto* reluNode = nextNode;

//     // Create our Operation + Activation and annotate it by modifying the
//     // original Operation
//     auto* fusedNode = g->createNode(util::make_unique<FusedT>(*operation));
//     auto fused = repr::nn::get<FusedT>(fusedNode);

//     // Modification of the original Fuseable
//     auto oldAnnotation = operation->getMutableAnnotation();
//     if (oldAnnotation) {
//       if (isa<caffe2::Caffe2Annotation>(oldAnnotation)) {
//         fused->setAnnotation(util::make_unique<caffe2::Caffe2Annotation>());
//         auto annotation = dyn_cast<caffe2::Caffe2Annotation>(fused->getMutableAnnotation());
//         auto operationOp = dyn_cast<caffe2::Caffe2Annotation>(oldAnnotation)->getMutableOperatorDef();
//         operationOp->set_type(fusedName);
//         annotation->setOperatorDef(operationOp);
//       } else {
//         assert(0 && "Unsupported annotation.");
//       }
//     }

//     for (const auto input : repr::nn::getInputs(operationNode)) {
//       g->createEdge(input, fusedNode);
//     }
//     for (const auto output : repr::nn::getOutputs(operationNode)) {
//       g->createEdge(fusedNode, output);
//     }

//     g->deleteNode(operationNode);
//     g->deleteNode(tensorNode);
//     g->deleteNode(reluNode);
//     return true;
//   }
//   return false;
// }

// $$ X_{bn} = \frac{s(X - m)}{\sqrt{\sigma + \epsilon}} + b_{bn}$$
// $$ X_{conv} = X * W + b_{conv} $$
// thus, substituting $X$ with $X_{conv}$ in the BN equation we get:
// $$X_{bn} = X * \frac{sW}{\sqrt{\sigma + \epsilon}} + \frac{s(b_{conv} - m)}{\sqrt{\sigma + \epsilon}} + b_{bn}$$
// or
// $$ W' = W\frac{s}{\sqrt{\sigma + \epsilon}}$$
// $$ b' = (b_{conv} - m)\frac{s}{\sqrt{\sigma + \epsilon}} + b_{bn}$$
bool fuseConvBNHelper(repr::NNModule* nn, caffe2::Workspace* ws) {
  for (auto node_pair : repr::nn::dataIterator<repr::Conv>(nn->dataFlow)) {
    repr::NNGraph::NodeRef convNode;
    repr::Conv* conv;
    std::tie(conv, convNode) = node_pair;

    auto output = repr::nn::getOutputs(convNode).front();
    auto consumers = repr::nn::getConsumers(output);
    if (consumers.size() != 1) {
      continue;
    }
    auto consumer = consumers.front();
    if (!repr::nn::is<repr::BatchNormalization>(consumer)) {
      continue;
    }
    auto bnNode = consumer;
    auto bn = repr::nn::get<repr::BatchNormalization>(bnNode);

    auto convInputs = repr::nn::getInputs(convNode);
    if (convInputs.size() < 3) {
      assert(0 && "Invalid convolution input size (TODO: optional bias)");
      continue;
    }

    auto bnInputs = repr::nn::getInputs(bnNode);
    if (bnInputs.size() < 5) {
      assert(0 && "Invalid batch normalization input size");
      continue;
    }

#define EXPOSE_TENSOR_DATA(name, index, inputs)                              \
  auto name = repr::nn::get<repr::Tensor>(inputs[index]);                    \
  assert(ws->HasBlob(name->getName()) && "Blob not in workspace");           \
  auto name##Tensor = ws->GetBlob(name->getName())->GetMutable<TensorCPU>(); \
  auto name##Data = name##Tensor->mutable_data<float>();

    EXPOSE_TENSOR_DATA(filter, 1, convInputs);
    EXPOSE_TENSOR_DATA(biasConv, 2, convInputs);

    EXPOSE_TENSOR_DATA(scale, 1, bnInputs);
    EXPOSE_TENSOR_DATA(biasBN, 2, bnInputs);
    EXPOSE_TENSOR_DATA(mean, 3, bnInputs);
    EXPOSE_TENSOR_DATA(variance, 4, bnInputs);

#undef EXPOSE_TENSOR_DATA

    // Assume M{CHW,HWC}
    auto chwDim = filterTensor->dim32(1) * filterTensor->dim32(2) *
        filterTensor->dim32(3);
    for (auto c = 0; c < filterTensor->dim32(0); ++c) {
      float coeff =
          scaleData[c] / std::sqrt(varianceData[c] + bn->getEpsilon());
      for (auto i = 0; i < chwDim; ++i) {
        filterData[c * chwDim + i] *= coeff;
      }
      auto bias = (biasConvData[c] - meanData[c]) * coeff + biasBNData[c];
      biasConvData[c] = bias;
    }

    nn->dataFlow.deleteNode(bnNode);
    return true;
  }
  return false;
}

// bool fuseConvRelu(nom::repr::NNModule* nn) {
//   while (fusionHelper<repr::Conv, repr::Relu, repr::ConvRelu>(
//       "ConvRelu", &nn->dataFlow)) {
//   }
//   return true;
// }

// bool fuseConvSigmoid(nom::repr::NNModule* nn) {
//   while (fusionHelper<repr::Conv, repr::Sigmoid, repr::ConvSigmoid>(
//       "ConvSigmoid", &nn->dataFlow)) {
//   }
//   return true;
// }

// bool fuseAveragePoolRelu(nom::repr::NNModule* nn) {
//   while (fusionHelper<repr::AveragePool, repr::Relu, repr::AveragePoolRelu>(
//       "AveragePoolRelu", &nn->dataFlow)) {
//   }
//   return true;
// }

// bool fuseMaxPoolRelu(nom::repr::NNModule* nn) {
//   while (fusionHelper<repr::MaxPool, repr::Relu, repr::MaxPoolRelu>(
//       "MaxPoolRelu", &nn->dataFlow)) {
//   }
//   return true;
// }

// bool fuseSumRelu(nom::repr::NNModule* nn) {
//   while (fusionHelper<repr::Sum, repr::Relu, repr::SumRelu>(
//       "SumRelu", &nn->dataFlow)) {
//   }
//   return true;
// }

void fuseConvBN(nom::repr::NNModule* nn, caffe2::Workspace* ws) {
  while (fuseConvBNHelper(nn, ws)) {
  }
}

REGISTER_WS_OPT_PASS_FROM_FUNC(FuseConvBN, fuseConvBN);

} // namespace opt
} // namespace caffe2
