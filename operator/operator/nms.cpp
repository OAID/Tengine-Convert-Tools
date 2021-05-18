#include "operator/nms.hpp"

namespace TEngine {

bool NMS::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    const  TShape& input_shape = ishape[0];
    int dim_num = input_shape.GetSize();
    TShape shape;
    std::vector<int> dim = {param_.max_class, dim_num};
    shape.SetDim(dim);
    shape.SetDataLayout(TENGINE_LAYOUT_NCHW);
    oshape[0] = shape;
    return true;
}

void NMS::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("max_class", 0)
        .SetAttr("iou_threshold", 0.f)
        .SetAttr("score_threshold", 0.f);
}

}    // namespace TEngine