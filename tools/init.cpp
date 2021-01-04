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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */

#include "config.hpp"

#include <iostream>
#include <functional>

#include "serializer.hpp"

#include "caffe_serializer.hpp"
#include "onnx_serializer.hpp"
#include "oneflow_serializer.hpp"
#include "mxnet_serializer.hpp"
#include "tf_serializer.hpp"
#include "tf_lite_serializer.hpp"
#include "tm_serializer.hpp"
#include "src_tm_serializer.hpp"
#include "ncnn_serializer.hpp"
#include "darknet_serializer.hpp"

#ifdef BUILD_MEGENGINE_SERIALIZER
#include "megengine_serializer.hpp"
#endif

#include "logger.hpp"

namespace TEngine {

extern bool OnnxSerializerRegisterOpLoader();
extern bool OneFlowSerializerRegisterOpLoader();
extern bool CaffeSerializerRegisterOpLoader();
extern bool MxnetSerializerRegisterOpLoader();
extern bool TFSerializerRegisterOpLoader();
extern bool NcnnSerializerRegisterOpLoader();
extern bool DarkNetSerializerRegisterOpLoader();
extern bool TFLiteSerializerRegisterOpLoader();

#ifdef BUILD_MEGENGINE_SERIALIZER
extern bool MegengineSerializerRegisterOpLoader();
#endif

bool TmSerializerInit(void);

}    // namespace TEngine

using namespace TEngine;

int serializer_plugin_init(void)
{
    // Register into factory
    auto factory = SerializerFactory::GetFactory();

    // CAFFE
    factory->RegisterInterface<CaffeSingle>("caffe_single");
    factory->RegisterInterface<CaffeBuddy>("caffe_buddy");

    auto caffe_single = factory->Create("caffe_single");
    auto caffe_buddy = factory->Create("caffe_buddy");

    SerializerManager::SafeAdd("caffe_single", SerializerPtr(caffe_single));
    SerializerManager::SafeAdd("caffe", SerializerPtr(caffe_buddy));

    CaffeSerializerRegisterOpLoader();

    // ONNX
    factory->RegisterInterface<OnnxSerializer>("onnx");
    auto onnx_serializer = factory->Create("onnx");

    SerializerManager::SafeAdd("onnx", SerializerPtr(onnx_serializer));
    OnnxSerializerRegisterOpLoader();

    // ONNX
    factory->RegisterInterface<OneFlowSerializer>("oneflow");
    auto oneflow_serializer = factory->Create("oneflow");

    SerializerManager::SafeAdd("oneflow", SerializerPtr(oneflow_serializer));
    OneFlowSerializerRegisterOpLoader();

    // MXNET
    factory->RegisterInterface<MxnetSerializer>("mxnet");
    auto mxnet_serializer = factory->Create("mxnet");

    SerializerManager::SafeAdd("mxnet", SerializerPtr(mxnet_serializer));

    MxnetSerializerRegisterOpLoader();

    // DARKNET
    factory->RegisterInterface<DarkNetSerializer>("darknet");
    auto darknet_serializer = factory->Create("darknet");

    SerializerManager::SafeAdd("darknet", SerializerPtr(darknet_serializer));

    DarkNetSerializerRegisterOpLoader();

    // TF
    factory->RegisterInterface<TFSerializer>("tensorflow");
    auto tf_serializer = factory->Create("tensorflow");

    SerializerManager::SafeAdd("tensorflow", SerializerPtr(tf_serializer));

    TFSerializerRegisterOpLoader();

    // TFLITE
    factory->RegisterInterface<TFLiteSerializer>("tflite");
    auto tf_lite_serializer = factory->Create("tflite");

    SerializerManager::SafeAdd("tflite", SerializerPtr(tf_lite_serializer));

    TFLiteSerializerRegisterOpLoader();

    // NCNN
    factory->RegisterInterface<NcnnSerializer>("ncnn");
    auto ncnn_serializer = factory->Create("ncnn");

    SerializerManager::SafeAdd("ncnn", SerializerPtr(ncnn_serializer));

    NcnnSerializerRegisterOpLoader();

#ifdef BUILD_MEGENGINE_SERIALIZER
    // MegEngine
    factory->RegisterInterface<MegengineSerializer>("megengine");
    auto megengine_serializer = factory->Create("megengine");

    SerializerManager::SafeAdd("megengine", SerializerPtr(megengine_serializer));

    MegengineSerializerRegisterOpLoader();
#endif

    TmSerializerInit();

    return 0;
}
