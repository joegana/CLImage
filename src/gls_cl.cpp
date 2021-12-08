/*******************************************************************************
 * Copyright (c) 2021 Glass Imaging Inc.
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

#include <fstream>
#include <iostream>
#include <map>

#include "gls_cl.hpp"
#include "gls_logging.h"

namespace gls {

static const char* TAG = "CLImage";

#ifdef __APPLE__

cl::Context getContext() {
    cl::Context context = cl::Context::getDefault();

    static bool initialized = false;
    if (!initialized) {
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        // Macs have multiple GPUs, select the one with most compute units
        int max_compute_units = 0;
        cl::Device best_device;
        for (const auto& d : devices) {
            int device_compute_units = d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
            if (device_compute_units > max_compute_units) {
                max_compute_units = device_compute_units;
                best_device = d;
            }
        }
        cl::Device::setDefault(best_device);

        cl::Device d = cl::Device::getDefault();
        LOG_INFO(TAG) << "OpenCL Default Device: " << d.getInfo<CL_DEVICE_NAME>() << std::endl;
        LOG_INFO(TAG) << "- Device Version: " << d.getInfo<CL_DEVICE_VERSION>() << std::endl;
        LOG_INFO(TAG) << "- Driver Version: " << d.getInfo<CL_DRIVER_VERSION>() << std::endl;
        LOG_INFO(TAG) << "- OpenCL C Version: " << d.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl;
        LOG_INFO(TAG) << "- Compute Units: " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
        LOG_INFO(TAG) << "- CL_DEVICE_MAX_WORK_GROUP_SIZE: " << d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
        LOG_INFO(TAG) << "- CL_DEVICE_EXTENSIONS: " << d.getInfo<CL_DEVICE_EXTENSIONS>() << std::endl;

        initialized = true;
    }
    return context;
}

#elif __ANDROID__

cl::Context getContext() {
    static bool initialized = false;

    if (!initialized) {
        CL_WRAPPER_NS::bindOpenCLLibrary();

        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform platform;
        for (auto& p : platforms) {
            std::string version = p.getInfo<CL_PLATFORM_VERSION>();
            if (version.find("OpenCL 2.") != std::string::npos) {
                platform = p;
            }
        }
        if (platform() == nullptr) {
            throw cl::Error(-1, "No OpenCL 2.0 platform found.");
        }

        cl::Platform defaultPlatform = cl::Platform::setDefault(platform);
        if (defaultPlatform != platform) {
            throw cl::Error(-1, "Error setting default platform.");
        }

        cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0};
        cl::Context context(CL_DEVICE_TYPE_ALL, properties);

        cl::Device d = cl::Device::getDefault();
        LOG_INFO(TAG) << "- Device: " << d.getInfo<CL_DEVICE_NAME>() << std::endl;
        LOG_INFO(TAG) << "- Device Version: " << d.getInfo<CL_DEVICE_VERSION>() << std::endl;
        LOG_INFO(TAG) << "- Driver Version: " << d.getInfo<CL_DRIVER_VERSION>() << std::endl;
        LOG_INFO(TAG) << "- OpenCL C Version: " << d.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl;
        LOG_INFO(TAG) << "- Compute Units: " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
        LOG_INFO(TAG) << "- CL_DEVICE_MAX_WORK_GROUP_SIZE: " << d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
        LOG_INFO(TAG) << "- CL_DEVICE_EXTENSIONS: " << d.getInfo<CL_DEVICE_EXTENSIONS>() << std::endl;

        cl::Context::setDefault(context);

        initialized = true;
    }

    return cl::Context::getDefault();
}
#endif

#if defined(__ANDROID__) && !defined(COMMAND_LINE_TOOL)
static std::map<std::string, std::string> cl_shaders;
static std::map<std::string, std::vector<unsigned char>> cl_bytecode;

std::map<std::string, std::string>* getShadersMap() { return &cl_shaders; }

std::map<std::string, std::vector<unsigned char>>* getBytecodeMap() { return &cl_bytecode; }
#endif

std::string OpenCLSource(std::string shaderName) {
#if defined(__ANDROID__) && !defined(COMMAND_LINE_TOOL)
    return cl_shaders[shaderName];
#else
    std::ifstream file("OpenCL/" + shaderName, std::ios::in | std::ios::ate);
    if (file.is_open()) {
        std::streampos size = file.tellg();
        std::vector<char> memblock((int)size);
        file.seekg(0, std::ios::beg);
        file.read(memblock.data(), size);
        file.close();
        return std::string(memblock.data(), memblock.data() + size);
    }
    return "";
#endif
}

std::vector<unsigned char> OpenCLBinary(std::string shaderName) {
#if defined(__ANDROID__) && !defined(COMMAND_LINE_TOOL)
    return cl_bytecode[shaderName];
#else
    std::ifstream file("OpenCLBinaries/" + shaderName, std::ios::in | std::ios::binary | std::ios::ate);
    if (file.is_open()) {
        std::streampos size = file.tellg();
        std::vector<unsigned char> memblock((int)size);
        file.seekg(0, std::ios::beg);
        file.read((char*)memblock.data(), size);
        file.close();
        return memblock;
    }
    return std::vector<unsigned char>();
#endif
}

int SaveBinaryFile(std::string path, const std::vector<unsigned char>& binary) {
    std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (file.is_open()) {
        file.write((char*)binary.data(), binary.size());
        file.close();
        LOG_INFO(TAG) << "Wrote " << binary.size() << " bytes to " << path << std::endl;
        return 0;
    }
    LOG_ERROR(TAG) << "Couldn't open file " << path << std::endl;
    return -1;
}

int SaveOpenCLBinary(std::string shaderName, const std::vector<unsigned char>& binary) {
    return SaveBinaryFile("OpenCL/" + shaderName, binary);
}

void handleProgramException(const cl::BuildError& e) {
    LOG_ERROR(TAG) << "OpenCL Build Error - " << e.what() << ": " << clStatusToString(e.err()) << std::endl;
    // Print build info for all devices
    for (auto& pair : e.getBuildLog()) {
        LOG_ERROR(TAG) << pair.second << std::endl;
    }
}

#ifdef __APPLE__
static const char* cl_options = "-cl-std=CL1.2 -Werror -cl-fast-relaxed-math";
#else
static const char* cl_options = "-cl-std=CL2.0 -Werror -cl-fast-relaxed-math";
#endif

static std::map<std::string, cl::Program*> cl_programs;

cl::Program* loadOpenCLProgram(const std::string& programName) {
    cl::Program* program = cl_programs[programName];
    if (program) {
        return program;
    }

    try {
        cl::Context context = getContext();
        cl::Device device = cl::Device::getDefault();

#if (defined(__ANDROID__) && defined(NDEBUG)) || (defined(__APPLE__) && !defined(TARGET_CPU_ARM64) && !defined(DEBUG))
        std::vector<unsigned char> binary = OpenCLBinary(programName + ".o");

        if (!binary.empty()) {
            program = new cl::Program(context, {device}, {binary});
        } else
#endif
        {
            program = new cl::Program(OpenCLSource(programName + ".cl"));
        }
        program->build(device, cl_options);
        cl_programs[programName] = program;
        return program;
    } catch (const cl::BuildError& e) {
        handleProgramException(e);
        return nullptr;
    }
}

int buildProgram(cl::Program& program) {
    try {
        program.build(cl_options);
        for (auto& pair : program.getBuildInfo<CL_PROGRAM_BUILD_LOG>()) {
            if (!pair.second.empty()) {
                LOG_INFO(TAG) << "OpenCL Build: " << pair.second << std::endl;
            }
        }
    } catch (const cl::BuildError& e) {
        handleProgramException(e);
        return -1;
    }
    return 0;
}

// Compute a list of divisors in the range [1..32]
std::vector<int> computeDivisors(const size_t val) {
    std::vector<int> divisors;
    int divisor = 32;
    while (divisor >= 1) {
        if (val % divisor == 0) {
            divisors.push_back(divisor);
        }
        divisor /= 2;
    }
    return divisors;
}

// Compute the squarest workgroup of size <= max_workgroup_size
cl::NDRange computeWorkGroupSizes(size_t width, size_t height) {
    std::vector<int> width_divisors = computeDivisors(width);
    std::vector<int> height_divisors = computeDivisors(height);

    cl::Device d = cl::Device::getDefault();
    const size_t max_workgroup_size = d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

    int width_divisor = 1;
    int height_divisor = 1;
    while (width_divisor * height_divisor <= max_workgroup_size &&
           (!width_divisors.empty() || !height_divisors.empty())) {
        if (!width_divisors.empty()) {
            int new_width_divisor = width_divisors.back();
            width_divisors.pop_back();
            if (new_width_divisor * height_divisor > max_workgroup_size) {
                break;
            } else {
                width_divisor = new_width_divisor;
            }
        }
        if (!height_divisors.empty()) {
            int new_height_divisor = height_divisors.back();
            height_divisors.pop_back();
            if (new_height_divisor * width_divisor > max_workgroup_size) {
                break;
            } else {
                height_divisor = new_height_divisor;
            }
        }
    }
    return cl::NDRange(width_divisor, height_divisor);
}

}  // namespace gls
