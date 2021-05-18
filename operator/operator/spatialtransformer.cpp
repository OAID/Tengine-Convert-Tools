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
 * Copyright (c) 2021, Open AI Lab
 * Author: bzhang@openailab.com
 */
#include "operator/spatialtransformer.hpp"

namespace TEngine {

bool SpatialTransformer::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    const TShape& input = ishape[0];

    int size = param_.target_shape.size();
    int out_h = 0;
    int out_w = 0;
    std::vector<int> dim;
    if(size == 2){
        out_h = param_.target_shape[0];
        out_w = param_.target_shape[1];
        dim.push_back(input.GetN());
        dim.push_back(input.GetC());
        dim.push_back(out_h);
        dim.push_back(out_w);
    }

    TShape shape;
    shape.SetDim(dim);
    shape.SetDataLayout(TENGINE_LAYOUT_NCHW);
    oshape[0] = shape;
    return true;
}

void SpatialTransformer::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("sampler_type", -1)
        .SetAttr("transformer_type", -1)
        .SetDoc(R"DOC(SpatialTransformer Layer)DOC");
}

}    // namespace TEngine
