/*******************************************************************************
 * Copyright (c) 2021-2022 Glass Imaging Inc.
 * Author: Fabio Riccardi <fabio@glass-imaging.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include <map>

#ifndef GLS_CL_HPP
#define GLS_CL_HPP

#define CL_HPP_ENABLE_EXCEPTIONS

#ifdef __APPLE__
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_USE_CL_IMAGE2D_FROM_BUFFER_KHR true

#include <OpenCL/cl_ext.h>

#include "CL/opencl.hpp"

#elif __ANDROID__

#define CL_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200

// Include cl_icd_wrapper.h before <CL/*>
#include "gls_icd_wrapper.h"

#include <CL/cl_ext.h>
#include <CL/opencl.hpp>

#endif

namespace gls {

class OpenCLContext {
    cl::Context _clContext;
    const std::string _shadersRootPath;
    std::map<std::string, cl::Program> _program_cache;
#if defined(__ANDROID__) && defined(USE_ASSET_MANAGER)
    std::map<std::string, std::string> cl_shaders;
    std::map<std::string, std::vector<unsigned char>> cl_bytecode;
#endif

   public:
    OpenCLContext(const std::string& shadersRootPath = "");

    cl::Context clContext() { return _clContext; }
    const std::string& shadersRootPath() { return _shadersRootPath; }

#if defined(__ANDROID__) && defined(USE_ASSET_MANAGER)
    std::map<std::string, std::string>* getShadersMap() { return &cl_shaders; }
    std::map<std::string, std::vector<unsigned char>>* getBytecodeMap() { return &cl_bytecode; }
#endif

    std::string OpenCLSource(const std::string& shaderName);
    std::vector<unsigned char> OpenCLBinary(const std::string& shaderName);

    cl::Program loadProgram(const std::string& programName, const std::string& shadersRootPath = "");

    static int saveBinaryFile(const std::string& path, const std::vector<unsigned char>& binary);

    static int buildProgram(cl::Program& program);

    static void handleProgramException(const cl::BuildError& e);

    static cl::NDRange computeWorkGroupSizes(size_t width, size_t height);

    inline static cl::EnqueueArgs buildEnqueueArgs(size_t width, size_t height) {
        cl::NDRange global_workgroup_size = cl::NDRange(width, height);
        cl::NDRange local_workgroup_size = computeWorkGroupSizes(width, height);
        return cl::EnqueueArgs(global_workgroup_size, local_workgroup_size);
    }
};

std::string clStatusToString(cl_int status);

}  // namespace gls
#endif /* GLS_CL_HPP */
