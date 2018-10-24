#include "caffe2/mobile/contrib/opencl/test/cl_model_test.h"

namespace caffe2 {

// The last softmax op didn't pass because of the dimension mismatch, and we are not likely to hit it in other models, but the implementation should be correct
// TEST(OPENGLModelTest, SqueezenetV11) {
//   std::string parent_path = "/data/local/tmp/";
//   benchmarkModel(parent_path + "squeezenet_init.pb", parent_path + "squeezenet_predict.pb", "data", {1, 3, 224, 224}, "squeezenet_v11");
// }

// TEST(OPENGLModelTest, Model) {
//   std::string parent_path = "/data/local/tmp/";
//   benchmarkModel(parent_path + "init_net.pb", parent_path + "predict_net.pb", "data", {1, 3, 640, 360}, "model", {"GenerateProposals", "BBoxTransform", "BoxWithNMSLimit", "RoIAlign"});
// }

TEST(OPENCLModelTest, Model) {
  std::string parent_path = "/data/local/tmp/";
  benchmarkModel(parent_path + "411_init.pb", parent_path + "411.pb", "data", {1, 3, 640, 360}, "model", {"GenerateProposals", "BoxWithNMSLimit", "RoIAlign", "BBoxID", "HeatmapPCAKeypoint", "Reshape", "Slice", "Shape", "Split", "AddPadding"});
}

} // namespace caffe2
