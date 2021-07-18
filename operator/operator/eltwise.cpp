/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2018, Open AI Lab
 * Author: chunyinglv@openailab.com
 */
#include "operator/eltwise.hpp"
#include "static_graph.hpp"
#include <cmath>

namespace TEngine {

bool Eltwise::InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout)
{
    if (ishape.size() == 1)
    {
        oshape = ishape;
        return true;
    }

    if (ishape.size() != 2)
    {
        return false;
    }

    TShape input_shape0 = ishape[0];
    TShape input_shape1 = ishape[1];
    auto& dim0 = input_shape0.GetDim();
    auto& dim1 = input_shape1.GetDim();

    int dim_num = dim0.size() >= dim1.size() ? dim0.size():dim1.size();
    std::vector<int> out_dims;
    if (dim0.size() >= dim1.size()){
        for (int i=0; i<dim0.size()-dim1.size();i++){
            out_dims.push_back(dim0[i]);
        }
        for (int i=0; i<dim1.size();i++){
            out_dims.push_back(dim0[dim0.size()-dim1.size()+i] >= dim1[i] ? dim0[dim0.size()-dim1.size()+i] : dim1[i]);
        }
 
    }
    else{
        for (int i=0; i<dim1.size()-dim0.size();i++){
            out_dims.push_back(dim1[i]);
        }
        for (int i=0; i<dim0.size();i++){
            out_dims.push_back(dim1[dim1.size()-dim0.size()+i] >= dim0[i] ? dim1[dim1.size()-dim0.size()+i] : dim0[i]);
        }
    }

/*     if (i0_size >= i1_size)
    {
        oshape[0] = ishape[0];
    }
    else if (i0_size < i1_size)
    {
        oshape[0] = ishape[1];
    } */
    TShape shape;
    shape.SetDim(out_dims);
    shape.SetDataLayout(layout);

    oshape[0] = shape;
    return true;
}

void Eltwise::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("type", 2)
        .SetAttr("caffe_flavor", 1)
        .SetAttr("power", 1)
        .SetAttr("scale", 1)
        .SetAttr("shift", 0)
        .SetDoc(R"DOC(Eltwise Layer)DOC");
}

}    // namespace TEngine
