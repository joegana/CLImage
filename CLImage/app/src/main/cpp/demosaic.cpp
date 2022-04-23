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

#include "gls_logging.h"

static const char* TAG = "CLImage Pipeline";

template <size_t levels = 4>
struct PyramidalDenoise {
    typedef gls::cl_image_2d<gls::rgba_pixel_float> imageType;
    std::array<imageType::unique_ptr, levels-1> imagePyramid;
    std::array<imageType::unique_ptr, levels> denoisedImagePyramid;

    PyramidalDenoise(gls::OpenCLContext* glsContext, const imageType& image) {
        for (int i = 0, scale = 2; i < levels-1; i++, scale *= 2) {
            imagePyramid[i] = std::make_unique<imageType>(glsContext->clContext(), image.width/scale, image.height/scale);
        }
        for (int i = 0, scale = 1; i < levels; i++, scale *= 2) {
            denoisedImagePyramid[i] = std::make_unique<imageType>(glsContext->clContext(), image.width/scale, image.height/scale);
        }
    }

    imageType* denoise(gls::OpenCLContext* glsContext, std::array<DenoiseParameters, levels>* denoiseParameters,
                       imageType* image, const gls::Matrix<3, 3>& rgb_cam) {

        gls::Matrix<4, 3> NFL({
            { 0.00243607,   0.00102689,     0.000649588 },
            { 0.00057763,   0.000519531,    0.000315311 },
            { 0.000153949,  0.00017422,     0.000109057 },
            { 0.00122609,   5.68481e-05,    3.30204e-05 }
        });

        // const gls::rectangle gmb_position = {3441, 773, 1531, 991};
        const gls::rectangle gmb_position = {4537, 2351, 1652, 1068};
        auto cpuImage = image->toImage();
        const auto nflParameters = extractNFLFromColoRchecher(cpuImage.get(), gmb_position, 1);
        (*denoiseParameters)[0].lumaVariance *= nflParameters[0];
        (*denoiseParameters)[0].cbVariance *= nflParameters[1];
        (*denoiseParameters)[0].crVariance *= nflParameters[2];

        for (int i = 0, scale = 2; i < levels-1; i++, scale *= 2) {
            resampleImage(glsContext, "downsampleImage", i == 0 ? *image : *imagePyramid[i - 1], imagePyramid[i].get());

            auto cpuLayer = imagePyramid[i]->toImage();
            const auto nflParameters = extractNFLFromColoRchecher(cpuLayer.get(), gmb_position, scale);
            (*denoiseParameters)[i+1].lumaVariance *= nflParameters[0];
            (*denoiseParameters)[i+1].cbVariance *= nflParameters[1];
            (*denoiseParameters)[i+1].crVariance *= nflParameters[2];
        }

//        for (int i = 0; i < 4; i++) {
//            (*denoiseParameters)[i].lumaVariance *= NFL[i][0];
//            (*denoiseParameters)[i].cbVariance *= NFL[i][1];
//            (*denoiseParameters)[i].crVariance *= NFL[i][2];
//        }

        // Denoise the bottom of the image pyramid
        denoiseImage(glsContext, *(imagePyramid[levels-2]),
                     (*denoiseParameters)[levels-1],
                     denoisedImagePyramid[levels-1].get());

        for (int i = levels - 2; i >= 0; i--) {
            // Denoise current layer
            denoiseImage(glsContext, i > 0 ? *(imagePyramid[i - 1]) : *image,
                         (*denoiseParameters)[i],
                         denoisedImagePyramid[i].get());

            // Subtract noise from previous layer
            reassembleImage(glsContext, *(denoisedImagePyramid[i]), *(imagePyramid[i]),
                            *(denoisedImagePyramid[i+1]), (*denoiseParameters)[i].sharpening, denoisedImagePyramid[i].get());
        }

        return denoisedImagePyramid[0].get();
    }
};

gls::image<gls::rgba_pixel>::unique_ptr demosaicImage(const gls::image<gls::luma_pixel_16>& rawImage, gls::tiff_metadata* metadata,
                                                      const DemosaicParameters& demosaicParameters, bool auto_white_balance) {
    auto t_start = std::chrono::high_resolution_clock::now();

    BayerPattern bayerPattern;
    float black_level;
    gls::Vector<4> scale_mul;
    gls::Matrix<3, 3> rgb_cam;

    unpackRawMetadata(rawImage, metadata, &bayerPattern, &black_level, &scale_mul, &rgb_cam, auto_white_balance);

    gls::OpenCLContext glsContext("");
    auto clContext = glsContext.clContext();

    LOG_INFO(TAG) << "Begin demosaicing image (GPU)..." << std::endl;

    // --- Image Demosaicing ---

    gls::cl_image_2d<gls::luma_pixel_16> clRawImage(clContext, rawImage);
    gls::cl_image_2d<gls::luma_pixel_float> clScaledRawImage(clContext, rawImage.width, rawImage.height);
    scaleRawData(&glsContext, clRawImage, &clScaledRawImage, bayerPattern, scale_mul, black_level / 0xffff);

    gls::cl_image_2d<gls::luma_pixel_float> clGreenImage(clContext, rawImage.width, rawImage.height);
    interpolateGreen(&glsContext, clScaledRawImage, &clGreenImage, bayerPattern, 0.003);

    gls::cl_image_2d<gls::rgba_pixel_float> clLinearRGBImage(clContext, rawImage.width, rawImage.height);
    interpolateRedBlue(&glsContext, clScaledRawImage, clGreenImage, &clLinearRGBImage, bayerPattern, 0.001, true);

    // --- Image Denoising ---

    std::array<DenoiseParameters, 4> denoiseParameters = {{
        {
            .lumaVariance = 1.0,
            .cbVariance = 1.0,
            .crVariance = 1.0,
            .sharpening = 1.0
        },
        {
            .lumaVariance = 1.0,
            .cbVariance = 1.0,
            .crVariance = 1.0,
            .sharpening = 1.0
        },
        {
            .lumaVariance = 1.0,
            .cbVariance = 1.0,
            .crVariance = 1.0,
            .sharpening = 1.0
        },
        {
            .lumaVariance = 1.0,
            .cbVariance = 1.0,
            .crVariance = 1.0,
            .sharpening = 1.0
        }
    }};

    // Convert linear image to YCbCr
    const auto cam_to_ycbcr = cam_ycbcr(rgb_cam);
    transformImage(&glsContext, clLinearRGBImage, &clLinearRGBImage, cam_to_ycbcr);

//    gls::cl_image_2d<gls::rgba_pixel_float> despeckledImage(glsContext.clContext(), clLinearRGBImage.width, clLinearRGBImage.height);
//    applyKernel(&glsContext, "despeckleImage", clLinearRGBImage, &despeckledImage);

    auto& despeckledImage = clLinearRGBImage;

    PyramidalDenoise pyramidalDenoise(&glsContext, despeckledImage);
    auto clDenoisedImage = pyramidalDenoise.denoise(&glsContext, &denoiseParameters, &despeckledImage, rgb_cam);

    // Convert result back to camera RGB
    transformImage(&glsContext, *clDenoisedImage, clDenoisedImage, inverse(cam_to_ycbcr));

    // --- Image Post Processing ---

    gls::cl_image_2d<gls::rgba_pixel> clsRGBImage(clContext, rawImage.width, rawImage.height);
    convertTosRGB(&glsContext, *clDenoisedImage, &clsRGBImage, rgb_cam, demosaicParameters);

    auto rgbaImage = clsRGBImage.toImage();

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    LOG_INFO(TAG) << "OpenCL Pipeline Execution Time: " << (int) elapsed_time_ms << "ms for image of size: " << rawImage.width << " x " << rawImage.height << std::endl;

    return rgbaImage;
}

gls::image<gls::rgba_pixel>::unique_ptr fastDemosaicImage(const gls::image<gls::luma_pixel_16>& rawImage, gls::tiff_metadata* metadata,
                                                          const DemosaicParameters& demosaicParameters, bool auto_white_balance) {
    auto t_start = std::chrono::high_resolution_clock::now();

    BayerPattern bayerPattern;
    float black_level;
    gls::Vector<4> scale_mul;
    gls::Matrix<3, 3> rgb_cam;

    unpackRawMetadata(rawImage, metadata, &bayerPattern, &black_level, &scale_mul, &rgb_cam, auto_white_balance);

    gls::OpenCLContext glsContext("");
    auto clContext = glsContext.clContext();

    LOG_INFO(TAG) << "Begin demosaicing image (GPU)..." << std::endl;

    // --- Image Demosaicing ---

    gls::cl_image_2d<gls::luma_pixel_16> clRawImage(clContext, rawImage);
    gls::cl_image_2d<gls::luma_pixel_float> clScaledRawImage(clContext, rawImage.width, rawImage.height);
    scaleRawData(&glsContext, clRawImage, &clScaledRawImage, bayerPattern, scale_mul, black_level / 0xffff);

    gls::cl_image_2d<gls::rgba_pixel_float> clLinearRGBImage(clContext, rawImage.width/2, rawImage.height/2);
    fasteDebayer(&glsContext, clScaledRawImage, &clLinearRGBImage, bayerPattern);

    // --- Image Post Processing ---

    gls::cl_image_2d<gls::rgba_pixel> clsRGBImage(clContext, clLinearRGBImage.width, clLinearRGBImage.height);
    convertTosRGB(&glsContext, clLinearRGBImage, &clsRGBImage, rgb_cam, demosaicParameters);

    auto rgbaImage = clsRGBImage.toImage();

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    LOG_INFO(TAG) << "OpenCL Pipeline Execution Time: " << (int) elapsed_time_ms << "ms for image of size: " << clLinearRGBImage.width << " x " << clLinearRGBImage.height << std::endl;

    return rgbaImage;
}
