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
        return 1;
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
// protected:
    // bool LoadGraph(te_caffe::NetParameter& test_net, te_caffe::NetParameter& train_net, StaticGraph* graph);
};

}

#endif