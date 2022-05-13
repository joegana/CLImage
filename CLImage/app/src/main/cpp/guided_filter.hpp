// Copyright (c) 2021-2022 Glass Imaging Inc.
// Author: Fabio Riccardi <fabio@glass-imaging.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef guided_filter_h
#define guided_filter_h

#include "gls_linalg.hpp"

#include "gls_cl_image.hpp"

template <typename PixelType>
struct BoxBlur {
    typename gls::cl_image_2d<PixelType>::unique_ptr tmp_image;

    BoxBlur(gls::OpenCLContext* glsContext, int width, int height);

    void blur(gls::OpenCLContext* glsContext,
              const gls::cl_image_2d<PixelType>& inputImage,
              int filterSize,
              gls::cl_image_2d<PixelType>* outputImage);
};

struct GuidedFilter {
    BoxBlur<gls::pixel_float4> boxBlurFloat4;
    BoxBlur<gls::pixel_fp32_4> boxBlur4;
    BoxBlur<gls::pixel_fp32_2> boxBlur2;

    gls::cl_image_2d<gls::pixel_float4>::unique_ptr mean_I;
    
    gls::cl_image_2d<gls::pixel_fp32_4>::unique_ptr invSigma1;
    gls::cl_image_2d<gls::pixel_fp32_2>::unique_ptr invSigma2;

    gls::cl_image_2d<gls::pixel_fp32_4>::unique_ptr cov_Ip_r;
    gls::cl_image_2d<gls::pixel_fp32_4>::unique_ptr cov_Ip_g;
    gls::cl_image_2d<gls::pixel_fp32_4>::unique_ptr cov_Ip_b;

    GuidedFilter(gls::OpenCLContext* glsContext, int width, int height);

    void filter(gls::OpenCLContext* glsContext,
                const gls::cl_image_2d<gls::pixel_float4>& inputImage,
                int filterSize, const gls::Vector<3>& eps,
                gls::cl_image_2d<gls::pixel_float4>* outputImage);
};

#endif /* guided_filter_h */
