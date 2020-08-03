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
 * Author: haitao@openailab.com
 */
#include "operator/expand.hpp"
#include <math.h>
#include <algorithm>

namespace TEngine {

bool Expand::InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout)
{
    const TShape input_shape1 = ishape[0];
    const TShape input_shape2 = ishape[1];

    std::vector<int> dims;

    std::vector<int> dim1 = input_shape1.GetDim();
    std::vector<int> dim2 = param_.shape;

    if (dim1.size() == dim2.size())
    {
        for (int i = 0; i < ( int )dim2.size(); i++)
        {
            dims.push_back(dim1[i] >= dim2[i] ? dim1[i] : dim2[i]);
        }
    }
    else
    {
        int diff = fabs(dim1.size() - dim2.size());
        if (dim1.size() > dim2.size())
        {
            for (int i = 0; i < ( int )dim1.size(); i++)
            {
                dims.push_back(dim1[i]);
            }
            for (int i = 0; i < ( int )dim1.size() - diff; i++)
            {
                dims.push_back(dim1[i + diff] > dim2[i] ? dim1[i + diff] : dim2[i]);
            }
        }
        else
        {
            for (int i = 0; i < ( int )dim2.size(); i++)
            {
                dims.push_back(dim2[i]);
            }
            for (int i = 0; i < ( int )dim2.size() - diff; i++)
            {
                dims.push_back(dim2[i + diff] > dim1[i] ? dim2[i + diff] : dim2[i]);
            }
        }
    }

    TShape shape;
    oshape[0].SetDim(dims);

    return true;
}

void Expand::SetSchema(void)
{
    Input({"input:float32"}).Output({"output:float32"}).SetDoc("DOC Expand DOC");
}

}    // namespace TEngine
