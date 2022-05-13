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

#include <stdio.h>

#include "guided_filter.hpp"

template <typename PixelType>
BoxBlur<PixelType>::BoxBlur(gls::OpenCLContext* glsContext, int width, int height) {
    tmp_image = std::make_unique<gls::cl_image_2d<PixelType>>(glsContext->clContext(), height, width);
}

template <typename PixelType>
void BoxBlur<PixelType>::blur(gls::OpenCLContext* glsContext,
                              const gls::cl_image_2d<PixelType>& inputImage,
                              int filterSize,
                              gls::cl_image_2d<PixelType>* outputImage) {
    assert(inputImage.width == outputImage->width && inputImage.height == outputImage->height &&
           inputImage.width == tmp_image->height && inputImage.height == tmp_image->width);

    // Load the shader source
    const auto program = glsContext->loadProgram("guided_filter");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,    // inputImage
                                     int,           // filterSize
                                     cl::Image2D    // outputImage
                                     >(program, "boxBlurX");

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(inputImage.width, inputImage.height),
           inputImage.getImage2D(), filterSize, tmp_image->getImage2D());

    kernel(gls::OpenCLContext::buildEnqueueArgs(tmp_image->width, tmp_image->height),
           tmp_image->getImage2D(), filterSize, outputImage->getImage2D());
}

template struct BoxBlur<gls::pixel_float4>;
// template struct BoxBlur<gls::pixel_fp32_4>;
template struct BoxBlur<gls::pixel_fp32_2>;

GuidedFilter::GuidedFilter(gls::OpenCLContext* glsContext, int width, int height) :
        boxBlurFloat4(glsContext, width, height),
        boxBlur4(glsContext, width, height),
        boxBlur2(glsContext, width, height) {
    auto cl_context = glsContext->clContext();

    mean_I = std::make_unique<gls::cl_image_2d<gls::pixel_float4>>(cl_context, width, height);
    invSigma1 = std::make_unique<gls::cl_image_2d<gls::pixel_fp32_4>>(cl_context, width, height);
    invSigma2 = std::make_unique<gls::cl_image_2d<gls::pixel_fp32_2>>(cl_context, width, height);
    cov_Ip_r = std::make_unique<gls::cl_image_2d<gls::pixel_fp32_4>>(cl_context, width, height);
    cov_Ip_g = std::make_unique<gls::cl_image_2d<gls::pixel_fp32_4>>(cl_context, width, height);
    cov_Ip_b = std::make_unique<gls::cl_image_2d<gls::pixel_fp32_4>>(cl_context, width, height);
}

void covMatrixProducts(gls::OpenCLContext* glsContext,
                       const gls::cl_image_2d<gls::pixel_float4>& inputImage,
                       gls::cl_image_2d<gls::pixel_fp32_4>* products1Image,
                       gls::cl_image_2d<gls::pixel_fp32_2>* products2Image) {
    // Load the shader source
    const auto program = glsContext->loadProgram("guided_filter");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,    // inputImage
                                    cl::Image2D,    // products1Image
                                    cl::Image2D     // products2Image
                                     >(program, "covMatrixProducts");

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(inputImage.width, inputImage.height),
           inputImage.getImage2D(), products1Image->getImage2D(), products2Image->getImage2D());
}

void invSigma(gls::OpenCLContext* glsContext,
              const gls::cl_image_2d<gls::pixel_float4>& meanImage,
              gls::cl_image_2d<gls::pixel_fp32_4>* invSigma1,
              gls::cl_image_2d<gls::pixel_fp32_2>* invSigma2,
              const gls::Vector<3>& eps) {
    // Load the shader source
    const auto program = glsContext->loadProgram("guided_filter");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,    // meanImage
                                    cl::Image2D,    // products1Image
                                    cl::Image2D,    // products2Image
                                    cl_float3,      // eps
                                    cl::Image2D,    // invSigma1
                                    cl::Image2D     // invSigma2
                                    >(program, "invSigma");

    cl_float3 cl_eps = { eps[0], eps[1], eps[2] };

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(meanImage.width, meanImage.height),
           meanImage.getImage2D(), invSigma1->getImage2D(), invSigma2->getImage2D(), cl_eps,
           invSigma1->getImage2D(), invSigma2->getImage2D());
}

void meanIpProducts(gls::OpenCLContext* glsContext,
                    const gls::cl_image_2d<gls::pixel_float4>& inputImage,
                    gls::cl_image_2d<gls::pixel_fp32_4>* mean_Ip_r,
                    gls::cl_image_2d<gls::pixel_fp32_4>* mean_Ip_g,
                    gls::cl_image_2d<gls::pixel_fp32_4>* mean_Ip_b) {
    // Load the shader source
    const auto program = glsContext->loadProgram("guided_filter");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,    // inputImage
                                    cl::Image2D,    // mean_Ip_r
                                    cl::Image2D,    // mean_Ip_g
                                    cl::Image2D     // mean_Ip_b
                                    >(program, "meanIpProducts");

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(inputImage.width, inputImage.height),
           inputImage.getImage2D(), mean_Ip_r->getImage2D(), mean_Ip_g->getImage2D(), mean_Ip_b->getImage2D());
}

void computeAb(gls::OpenCLContext* glsContext,
               const gls::cl_image_2d<gls::pixel_float4>& mean_pImage,
               const gls::cl_image_2d<gls::pixel_fp32_4>& mean_Ip_rImage,
               const gls::cl_image_2d<gls::pixel_fp32_4>& mean_Ip_gImage,
               const gls::cl_image_2d<gls::pixel_fp32_4>& mean_Ip_bImage,
               const gls::cl_image_2d<gls::pixel_fp32_4>& invSigma1Image,
               const gls::cl_image_2d<gls::pixel_fp32_2>& invSigma2Image,
               gls::cl_image_2d<gls::pixel_fp32_4>* ab_rImage,
               gls::cl_image_2d<gls::pixel_fp32_4>* ab_gImage,
               gls::cl_image_2d<gls::pixel_fp32_4>* ab_bImage) {
    // Load the shader source
    const auto program = glsContext->loadProgram("guided_filter");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,    // mean_pImage
                                    cl::Image2D,    // mean_Ip_rImage
                                    cl::Image2D,    // mean_Ip_gImage
                                    cl::Image2D,    // mean_Ip_bImage
                                    cl::Image2D,    // invSigma1Image
                                    cl::Image2D,    // invSigma2Image
                                    cl::Image2D,    // ab_rImage
                                    cl::Image2D,    // ab_gImage
                                    cl::Image2D     // ab_bImage
                                    >(program, "computeAb");

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(mean_pImage.width, mean_pImage.height),
           mean_pImage.getImage2D(),
           mean_Ip_rImage.getImage2D(),
           mean_Ip_gImage.getImage2D(),
           mean_Ip_bImage.getImage2D(),
           invSigma1Image.getImage2D(),
           invSigma2Image.getImage2D(),
           ab_rImage->getImage2D(),
           ab_gImage->getImage2D(),
           ab_bImage->getImage2D());
}

void computeResult(gls::OpenCLContext* glsContext,
                   const gls::cl_image_2d<gls::pixel_float4>& inputImage,
                   const gls::cl_image_2d<gls::pixel_fp32_4>& ab_rImage,
                   const gls::cl_image_2d<gls::pixel_fp32_4>& ab_gImage,
                   const gls::cl_image_2d<gls::pixel_fp32_4>& ab_bImage,
                   gls::cl_image_2d<gls::pixel_float4>* resultImage) {
    // Load the shader source
    const auto program = glsContext->loadProgram("guided_filter");

    // Bind the kernel parameters
    auto kernel = cl::KernelFunctor<cl::Image2D,    // inputImage
                                    cl::Image2D,    // ab_rImage
                                    cl::Image2D,    // ab_gImage
                                    cl::Image2D,    // ab_bImage
                                    cl::Image2D     // resultImage
                                    >(program, "computeResult");

    // Schedule the kernel on the GPU
    kernel(gls::OpenCLContext::buildEnqueueArgs(inputImage.width, inputImage.height),
           inputImage.getImage2D(),
           ab_rImage.getImage2D(),
           ab_gImage.getImage2D(),
           ab_bImage.getImage2D(),
           resultImage->getImage2D());
}

void GuidedFilter::filter(gls::OpenCLContext* glsContext,
                          const gls::cl_image_2d<gls::pixel_float4>& inputImage,
                          int filterSize, const gls::Vector<3>& eps,
                          gls::cl_image_2d<gls::pixel_float4>* outputImage) {
    boxBlurFloat4.blur(glsContext, inputImage, filterSize, mean_I.get());

    covMatrixProducts(glsContext, inputImage, invSigma1.get(), invSigma2.get());

    boxBlur4.blur(glsContext, *invSigma1, filterSize, invSigma1.get());
    boxBlur2.blur(glsContext, *invSigma2, filterSize, invSigma2.get());

    invSigma(glsContext, *mean_I, invSigma1.get(), invSigma2.get(), eps);

    meanIpProducts(glsContext, inputImage, cov_Ip_r.get(), cov_Ip_g.get(), cov_Ip_b.get());

    boxBlur4.blur(glsContext, *cov_Ip_r, filterSize, cov_Ip_r.get());
    boxBlur4.blur(glsContext, *cov_Ip_g, filterSize, cov_Ip_g.get());
    boxBlur4.blur(glsContext, *cov_Ip_b, filterSize, cov_Ip_b.get());

    auto ab_r = cov_Ip_r.get();
    auto ab_g = cov_Ip_g.get();
    auto ab_b = cov_Ip_b.get();
    computeAb(glsContext, *mean_I, *cov_Ip_r, *cov_Ip_g, *cov_Ip_b, *invSigma1, *invSigma2,
              ab_r, ab_g, ab_b);

    boxBlur4.blur(glsContext, *ab_r, filterSize, ab_r);
    boxBlur4.blur(glsContext, *ab_g, filterSize, ab_g);
    boxBlur4.blur(glsContext, *ab_b, filterSize, ab_b);

    computeResult(glsContext, inputImage, *ab_r, *ab_g, *ab_b, outputImage);
}
