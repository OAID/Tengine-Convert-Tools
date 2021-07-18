#include "operator/gather.hpp"
#include "operator/gather_param.hpp"
#include "static_graph.hpp"
#include <cmath>

namespace TEngine {

bool Gather::InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout)
{
    const TShape& input = ishape[0];
    const TShape& input2 = ishape[1];

    std::vector<int> input_dim = input.GetDim();
    std::vector<int> input_dim2 = input2.GetDim();
    std::vector<int> output_dim;
    printf ("input2_num: %d\n", input_dim2[0]);
    if (param_.axis > ( int )input_dim.size())
    {
        return false;
    }
    int indices_size;
    if (param_.indices_num != 0){

        indices_size = param_.indices_num;

    }
    else {
        indices_size = input_dim2[0];
    }
    
    /*
    printf("gather input dims: ");
    for(int i =0; i<(int)input_dim.size(); i++){
        printf("%d ", input_dim[i]);
    }
    printf("\n");
    */
    if (param_.is_onnx == true)
    {
        if (param_.axis == 0)
        {
            for (int i = 0; i < ( int )input_dim.size() - 1; i++)
            {
                output_dim.push_back(input_dim[i + 1]);
            }
        }
        else
        {
            for (int i = 0; i < ( int )input_dim.size(); i++)
            {
                if (i == param_.axis)
                    output_dim.push_back(indices_size);
                else
                {
                    output_dim.push_back(input_dim[i]);
                }
            }
        }
        oshape[0].SetDim(output_dim);
    }
    else
    {
        input_dim[param_.axis] = indices_size;
        oshape[0].SetDim(input_dim);
    }

    /*
    printf("gather output dims: ");
    for(int i =0; i<(int)output_dim.size(); i++){
        printf("%d ", output_dim[i]);
    }
    printf("\n");
    */

    oshape[0].SetDataLayout(input.GetDataLayout());

    return true;
}

void Gather::SetSchema(void)
{
    Input({"input:float32", "indices:float32"})
        .Output({"output:float32"})
        .SetAttr("axis", 0)
        .SetAttr("indices_size", 1)
        .SetAttr("is_onnx", false)
        .SetDoc(R"DOC(Slice Operator)DOC");
}

}    // namespace TEngine
