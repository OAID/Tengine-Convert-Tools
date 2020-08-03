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
 * Copyright (c) 2019, Open AI Lab
 * Author: bingzhang@openailab.com
 */
#include "operator/shape.hpp"
#include "static_graph.hpp"
#include <cmath>

namespace TEngine {
bool Shape::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    const TShape& input = ishape[0];

    std::vector<int> out_dim;
    const std::vector<int>& in_dim = input.GetDim();
    TShape shape;

    out_dim.push_back(( int )in_dim.size());
    shape.SetDim(out_dim);
    shape.SetDataLayout(input.GetDataLayout());

    oshape[0] = shape;

    return true;
}
void Shape::SetSchema(void)
{
    Input({"input:float32"}).Output({"output:int32"}).SetDoc(R"DOC(Shape Operator)DOC");
}

}    // namespace TEngine