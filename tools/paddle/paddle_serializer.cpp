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
 * Author: bzhang@openailab.com
 *         cxl@openailab.com
 */

#include "paddle_serializer.hpp"
#include <set>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include "tengine_c_api.h"
#include "exec_attr.hpp"
#include "type_name.hpp"
#include "data_type.hpp"
#include "tengine_errno.hpp"
#include "static_graph.hpp"
#include "operator_manager.hpp"


namespace TEngine {

using op_load_t = std::function<bool(StaticGraph* graph, StaticNode* node, const PaddleNode& paddle_node)>;

bool PaddleSerializer::LoadModel(const std::vector<std::string>& file_list, StaticGraph* graph)
{
    return true;
}

bool PaddleSerializerRegisterOpLoader(void)
{
    SerializerPtr serializer;

    if (!SerializerManager::SafeGet("paddle", serializer))
        return false;

    // PaddleSerializer* p_paddle = dynamic_cast<PaddleSerializer*>(serializer.get());

    return true;
}


}