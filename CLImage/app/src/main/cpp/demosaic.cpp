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

#include "demosaic.hpp"
#include "demosaic_cl.hpp"

#include "pyramidal_denoise.hpp"

#include "guided_filter.hpp"

#include "gls_logging.h"

static const char* TAG = "CLImage Pipeline";

gls::image<gls::rgb_pixel>::unique_ptr demosaicImage(const gls::image<gls::luma_pixel_16>& rawImage, gls::tiff_metadata* metadata,
                                                     DemosaicParameters* demosaicParameters, bool auto_white_balance,
                                                     const gls::rectangle* gmb_position, bool rotate_180) {
    auto t_start = std::chrono::high_resolution_clock::now();

    gls::OpenCLContext glsContext("");
    auto clContext = glsContext.clContext();

    LOG_INFO(TAG) << "Begin demosaicing image..." << std::endl;

    // --- Image Demosaicing ---

    gls::cl_image_2d<gls::luma_pixel_16> clRawImage(clContext, rawImage);

//    gls::cl_image_2d<gls::rgba_pixel_float> rgbaRawImage(clContext, rawImage.width/2, rawImage.height/2);
//    bayerToRawRGBA(&glsContext, clRawImage, &rgbaRawImage, demosaicParameters->bayerPattern);
//
//    gls::cl_image_2d<gls::rgba_pixel_float> denoisedRgbaRawImage(clContext, rawImage.width/2, rawImage.height/2);
//    despeckleRawRGBAImage(&glsContext,
//                          rgbaRawImage,
//                          &denoisedRgbaRawImage);
//
////    denoiseRawRGBAImage(&glsContext,
////                        denoisedRgbaRawImage,
////                        demosaicParameters->noiseModel.rawNlf,
////                        &rgbaRawImage);
//
//    rawRGBAToBayer(&glsContext, denoisedRgbaRawImage, &clRawImage, demosaicParameters->bayerPattern);

    gls::cl_image_2d<gls::luma_pixel_float> clScaledRawImage(clContext, rawImage.width, rawImage.height);
    scaleRawData(&glsContext, clRawImage, &clScaledRawImage, demosaicParameters->bayerPattern, demosaicParameters->scale_mul,
                 demosaicParameters->black_level / 0xffff);

    NoiseModel* noiseModel = &demosaicParameters->noiseModel;

    const float min_green_variance = 5e-05;
    const float max_green_variance = 5e-03;
    const float nlf_green_variance = std::clamp(noiseModel->rawNlf[1], min_green_variance, max_green_variance);

    const float nlf_alpha = log2(nlf_green_variance / min_green_variance) / log2(max_green_variance / min_green_variance);

    LOG_INFO(TAG) << "nlf_green_variance: " << nlf_green_variance << ", nlf_alpha: " << std::fixed << nlf_alpha << std::endl;

    const float raw_sigma_treshold = nlf_alpha > 0.8 ? 32 : nlf_alpha > 0.6 ? 8 : 4;

    gls::cl_image_2d<gls::luma_pixel_float> clGreenImage(clContext, rawImage.width, rawImage.height);
    interpolateGreen(&glsContext, clScaledRawImage, &clGreenImage, demosaicParameters->bayerPattern, raw_sigma_treshold * sqrt(noiseModel->rawNlf[1]));

    gls::cl_image_2d<gls::rgba_pixel_float> clLinearRGBImage(clContext, rawImage.width, rawImage.height);
    interpolateRedBlue(&glsContext, clScaledRawImage, clGreenImage, &clLinearRGBImage, demosaicParameters->bayerPattern, raw_sigma_treshold * sqrt((noiseModel->rawNlf[0] + noiseModel->rawNlf[2]) / 2), rotate_180);

    // --- Image Denoising ---

    // Convert linear image to YCbCr
    auto cam_to_ycbcr = cam_ycbcr(demosaicParameters->rgb_cam);

    std::cout << "cam_to_ycbcr: " << cam_to_ycbcr.span() << std::endl;

//    auto guidedFilter = GuidedFilter(&glsContext, clLinearRGBImage.width, clLinearRGBImage.height);
//    guidedFilter.filter(&glsContext, clLinearRGBImage, 5, { 0.001, 0.001, 0.001 }, &clLinearRGBImage);

    transformImage(&glsContext, clLinearRGBImage, &clLinearRGBImage, cam_to_ycbcr);

    gls::cl_image_2d<gls::rgba_pixel_float>::unique_ptr despeckledImage = nullptr;
    if (nlf_alpha > 0.6) {
        std::cout << "despeckleYCbCrImage" << std::endl;
        despeckledImage = std::make_unique<gls::cl_image_2d<gls::rgba_pixel_float>>(glsContext.clContext(), clLinearRGBImage.width, clLinearRGBImage.height);
        applyKernel(&glsContext, "despeckleYCbCrImage", clLinearRGBImage, despeckledImage.get());
    }

    PyramidalDenoise<5> pyramidalDenoise(&glsContext, despeckledImage ? *despeckledImage : clLinearRGBImage);
    auto clDenoisedImage = pyramidalDenoise.denoise(&glsContext, &demosaicParameters->denoiseParameters,
                                                    despeckledImage ? despeckledImage.get() : &clLinearRGBImage,
                                                    demosaicParameters->rgb_cam, gmb_position, false,
                                                    &noiseModel->pyramidNlf);

    // denoiseLumaImage(&glsContext, *clDenoisedImage, demosaicParameters->denoiseParameters[0], &clLinearRGBImage);

//    auto boxBlur = BoxBlur<gls::rgba_pixel_float>(&glsContext, clDenoisedImage->width, clDenoisedImage->height);
//    boxBlur.blur(&glsContext, *clDenoisedImage, 5, clDenoisedImage);

//    auto guidedFilter = GuidedFilter(&glsContext, clDenoisedImage->width, clDenoisedImage->height);
//    guidedFilter.filter(&glsContext, *clDenoisedImage, 5, { 0.001, 0.001, 0.001 }, clDenoisedImage);

    // Convert result back to camera RGB
    transformImage(&glsContext, *clDenoisedImage, clDenoisedImage, inverse(cam_to_ycbcr));

    // --- Image Post Processing ---

    gls::cl_image_2d<gls::rgba_pixel> clsRGBImage(clContext, rawImage.width, rawImage.height);
    convertTosRGB(&glsContext, *clDenoisedImage, &clsRGBImage, demosaicParameters->rgb_cam, *demosaicParameters); // TODO: ???

    auto rgbImage = std::make_unique<gls::image<gls::rgb_pixel>>(clsRGBImage.width, clsRGBImage.height);
    auto rgbaImage = clsRGBImage.mapImage();
    for (int y = 0; y < clsRGBImage.height; y++) {
        for (int x = 0; x < clsRGBImage.width; x++) {
            const auto& p = rgbaImage[y][x];
            (*rgbImage)[y][x] = { p.red, p.green, p.blue };
        }
    }
    clsRGBImage.unmapImage(rgbaImage);

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    LOG_INFO(TAG) << "OpenCL Pipeline Execution Time: " << (int) elapsed_time_ms << "ms for image of size: " << rawImage.width << " x " << rawImage.height << std::endl;

    return rgbImage;
}

gls::image<gls::rgba_pixel>::unique_ptr fastDemosaicImage(const gls::image<gls::luma_pixel_16>& rawImage, gls::tiff_metadata* metadata,
                                                          const DemosaicParameters& demosaicParameters, bool auto_white_balance) {
    auto t_start = std::chrono::high_resolution_clock::now();

    gls::OpenCLContext glsContext("");
    auto clContext = glsContext.clContext();

    LOG_INFO(TAG) << "Begin demosaicing image (GPU)..." << std::endl;

    // --- Image Demosaicing ---

    gls::cl_image_2d<gls::luma_pixel_16> clRawImage(clContext, rawImage);
    gls::cl_image_2d<gls::luma_pixel_float> clScaledRawImage(clContext, rawImage.width, rawImage.height);
    scaleRawData(&glsContext, clRawImage, &clScaledRawImage, demosaicParameters.bayerPattern, demosaicParameters.scale_mul,
                 demosaicParameters.black_level / 0xffff);

    gls::cl_image_2d<gls::rgba_pixel_float> clLinearRGBImage(clContext, rawImage.width/2, rawImage.height/2);
    fasteDebayer(&glsContext, clScaledRawImage, &clLinearRGBImage, demosaicParameters.bayerPattern);

    // --- Image Post Processing ---

    gls::cl_image_2d<gls::rgba_pixel> clsRGBImage(clContext, clLinearRGBImage.width, clLinearRGBImage.height);
    convertTosRGB(&glsContext, clLinearRGBImage, &clsRGBImage, demosaicParameters.rgb_cam, demosaicParameters);

    auto rgbaImage = clsRGBImage.toImage();

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    LOG_INFO(TAG) << "OpenCL Pipeline Execution Time: " << (int) elapsed_time_ms << "ms for image of size: " << clLinearRGBImage.width << " x " << clLinearRGBImage.height << std::endl;

    return rgbaImage;
}
