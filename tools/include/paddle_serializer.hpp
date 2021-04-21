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
#ifndef __PADDLE_SERIALIZER_HPP__
#define __PADDLE_SERIALIZER_HPP__

#include <iostream>
#include <fstream>
#include <sstream>
#include <functional>
#include <map>
#include <vector>

#include "serializer.hpp"
#include "static_graph_interface.hpp"
#include "logger.hpp"

#include "framework.pb.h"

namespace TEngine {

struct PaddleNode
{

};

class PaddleSerializer : public Serializer
{
public:
    PaddleSerializer()
    {
        name_ = "paddle_loader";
        version_ = "0.1";
        format_name_ = "paddle";
    }
    virtual ~PaddleSerializer(){}

    unsigned int GetFileNum(void) override
    {
        return 2;
    }

    bool LoadModel(const std::vector<std::string>& file_list, StaticGraph* graph) override;

    bool LoadConstTensor(const std::string& fname, StaticTensor* const_tensor) override
    {
        return false;
    }
    bool LoadConstTensor(int fd, StaticTensor* const_tensor) override
    {
        return false;
    }
protected:
    bool LoadBinaryFile(const char* fname);
    bool LoadTextFile(const char* fname, paddle::framework::proto::ProgramDesc& model_net);

    bool LoadGraph(paddle::framework::proto::ProgramDesc& model, StaticGraph* graph);
    bool LoadConstTensor(StaticGraph* graph, const paddle::framework::proto::BlockDesc& paddle_graph);
    void CreateInputNode(StaticGraph* graph, const paddle::framework::proto::BlockDesc& paddle_graph);
    bool LoadNode(StaticGraph* graph, StaticNode*, const paddle::framework::proto::OpDesc&);
    void LoadConstNode(const paddle::framework::proto::BlockDesc& paddle_graph, StaticGraph* graph);

    std::vector<std::string> paddle_node_outputs;
    
};

}

#endif