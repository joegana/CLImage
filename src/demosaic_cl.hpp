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

#ifndef demosaic_cl_hpp
#define demosaic_cl_hpp

#include "demosaic.hpp"

#include "gls_cl_image.hpp"

template <typename T>
void applyKernel(gls::OpenCLContext* glsContext, const std::string& kernelName,
                 const gls::cl_image_2d<T>& inputImage,
                 gls::cl_image_2d<T>* outputImage);

void scaleRawData(gls::OpenCLContext* glsContext,
                  const gls::cl_image_2d<gls::luma_pixel_16>& rawImage,
                  gls::cl_image_2d<gls::luma_pixel_float>* scaledRawImage,
                  BayerPattern bayerPattern, gls::Vector<4> scaleMul, float blackLevel);

void interpolateGreen(gls::OpenCLContext* glsContext,
                      const gls::cl_image_2d<gls::luma_pixel_float>& rawImage,
                      gls::cl_image_2d<gls::luma_pixel_float>* greenImage,
                      BayerPattern bayerPattern, float lumaVariance);

void interpolateRedBlue(gls::OpenCLContext* glsContext,
                        const gls::cl_image_2d<gls::luma_pixel_float>& rawImage,
                        const gls::cl_image_2d<gls::luma_pixel_float>& greenImage,
                        gls::cl_image_2d<gls::rgba_pixel_float>* rgbImage,
                        BayerPattern bayerPattern, float chromaVariance, bool rotate_180);

void fasteDebayer(gls::OpenCLContext* glsContext,
                  const gls::cl_image_2d<gls::luma_pixel_float>& rawImage,
                  gls::cl_image_2d<gls::rgba_pixel_float>* rgbImage,
                  BayerPattern bayerPattern);

template <typename T>
void resampleImage(gls::OpenCLContext* glsContext, const std::string& kernelName,
                   const gls::cl_image_2d<T>& inputImage, gls::cl_image_2d<T>* outputImage);

template <typename T>
void reassembleImage(gls::OpenCLContext* glsContext, const gls::cl_image_2d<T>& inputImageDenoised0,
                     const gls::cl_image_2d<T>& inputImage1, const gls::cl_image_2d<T>& inputImageDenoised1,
                     float sharpening, float lumaSigma, gls::cl_image_2d<T>* outputImage);

void transformImage(gls::OpenCLContext* glsContext,
                    const gls::cl_image_2d<gls::rgba_pixel_float>& linearImage,
                    gls::cl_image_2d<gls::rgba_pixel_float>* rgbImage,
                    const gls::Matrix<3, 3>& transform);

void convertTosRGB(gls::OpenCLContext* glsContext,
                   const gls::cl_image_2d<gls::rgba_pixel_float>& linearImage,
                   gls::cl_image_2d<gls::rgba_pixel>* rgbImage,
                   const gls::Matrix<3, 3>& transform, const DemosaicParameters& demosaicParameters);

void denoiseImage(gls::OpenCLContext* glsContext,
                  const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                  const DenoiseParameters& denoiseParameters, bool tight,
                  gls::cl_image_2d<gls::rgba_pixel_float>* outputImage);

#endif /* demosaic_cl_hpp */