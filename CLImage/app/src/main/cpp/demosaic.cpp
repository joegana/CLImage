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

static const float IMX492NFL[6][4][3] = {
    // ISO 100
    {
        { 4.38275e-05, 5.31895e-06, 2.57406e-05 },
        { 1.61605e-05, 5.20781e-06, 2.30347e-05 },
        { 7.54757e-06, 5.11204e-06, 2.02702e-05 },
        { 4.60632e-06, 5.04738e-06, 1.91247e-05 }
    },
    // ISO 200
    {
        { 8.22319e-05, 1.17027e-05, 4.51354e-05 },
        { 2.93849e-05, 1.16521e-05, 3.96089e-05 },
        { 1.34504e-05, 1.14135e-05, 3.48181e-05 },
        { 7.84091e-06, 1.12731e-05, 3.21395e-05 }
    },
    // ISO 400
    {
        { 0.000157257, 7.88058e-06, 4.74268e-05 },
        { 5.33433e-05, 7.64073e-06, 3.63058e-05 },
        { 2.01735e-05, 7.28965e-06, 2.59128e-05 },
        { 9.03684e-06, 7.11232e-06, 2.07884e-05 }
    },
    // ISO 800
    {
        { 0.000298601, 7.07557e-06, 5.32054e-05 },
        { 9.16717e-05, 6.38477e-06, 3.14448e-05 },
        { 2.9975e-05,  5.62945e-06, 1.21154e-05 },
        { 9.22798e-06, 5.40723e-06, 2.80249e-06 }
    },
    // ISO 1600
    {
        { 0.000638579, 1.94856e-05, 0.000115001 },
        { 0.000202505, 1.66096e-05, 6.85067e-05 },
        { 6.08632e-05, 1.47908e-05, 2.62121e-05 },
        { 1.84483e-05, 1.41926e-05, 6.27596e-06 }
    },
    // ISO 32000
    {
        { 0.00151377,  1.19521e-05, 0.0003362   },
        { 0.000480129, 8.78253e-06, 0.000211642 },
        { 0.000147973, 7.96094e-06, 0.000100203 },
        { 4.12846e-05, 7.78549e-06, 4.89834e-05 }
    },
};

std::array<std::array<float, 3>, 4> lerpNFL(const float NFLData0[4][3], const float NFLData1[4][3], float a) {
    std::array<std::array<float, 3>, 4> result;
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 3; i++) {
            result[j][i] = std::lerp(NFLData0[j][i], NFLData1[j][i], a);
        }
    }
    return result;
}

std::array<std::array<float, 3>, 4> nfl(const float NFLData[6][4][3], int iso) {
    if (iso >= 100 && iso < 200) {
        float a = (iso - 100) / 100;
        return lerpNFL(NFLData[0], NFLData[1], a);
    } else if (iso >= 200 && iso < 400) {
        float a = (iso - 200) / 200;
        return lerpNFL(NFLData[1], NFLData[2], a);
    } else if (iso >= 400 && iso < 800) {
        float a = (iso - 400) / 400;
        return lerpNFL(NFLData[2], NFLData[3], a);
    } else if (iso >= 800 && iso < 1600) {
        float a = (iso - 800) / 800;
        return lerpNFL(NFLData[3], NFLData[4], a);
    } else if (iso >= 1600 && iso <= 3200) {
        float a = (iso - 1600) / 1600;
        return lerpNFL(NFLData[4], NFLData[5], a);
    } else {
        // TODO: Maybe throw an error?
        return lerpNFL(NFLData[0], NFLData[0], 0);
    }
}

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
                       imageType* image, const gls::Matrix<3, 3>& rgb_cam, int iso) {
        const bool use_nfl_calibration = true;

        // const gls::rectangle gmb_position = {3441, 773, 1531, 991};
        const gls::rectangle gmb_position = {4537-60, 2351, 1652, 1068};
        if (!use_nfl_calibration) {
            auto cpuImage = image->toImage();
            const auto nflParameters = extractNlfFromColorChecker(cpuImage.get(), gmb_position, 1);
            (*denoiseParameters)[0].lumaSigma *= sqrt(nflParameters[0]);
            (*denoiseParameters)[0].cbSigma *= sqrt(nflParameters[1]);
            (*denoiseParameters)[0].crSigma *= sqrt(nflParameters[2]);
        }

        for (int i = 0, scale = 2; i < levels-1; i++, scale *= 2) {
            resampleImage(glsContext, "downsampleImage", i == 0 ? *image : *imagePyramid[i - 1], imagePyramid[i].get());

            if (!use_nfl_calibration) {
                auto cpuLayer = imagePyramid[i]->toImage();
                const auto nflParameters = extractNlfFromColorChecker(cpuLayer.get(), gmb_position, scale);
                (*denoiseParameters)[i+1].lumaSigma *= sqrt(nflParameters[0]);
                (*denoiseParameters)[i+1].cbSigma *= sqrt(nflParameters[1]);
                (*denoiseParameters)[i+1].crSigma *= sqrt(nflParameters[2]);
            }
        }

        if (use_nfl_calibration) {
            for (int i = 0; i < 4; i++) {
                // TODO: Further parametrize this
                auto nfl_params = nfl(IMX492NFL, iso);
                (*denoiseParameters)[i].lumaSigma *= sqrt(nfl_params[i][0]);
                (*denoiseParameters)[i].cbSigma *= sqrt(nfl_params[i][1]);
                (*denoiseParameters)[i].crSigma *= sqrt(nfl_params[i][2]);
            }
        }

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
                                                      const DemosaicParameters& demosaicParameters, int iso, bool auto_white_balance) {
    auto t_start = std::chrono::high_resolution_clock::now();

    BayerPattern bayerPattern;
    float black_level;
    gls::Vector<4> scale_mul;
    gls::Matrix<3, 3> rgb_cam;

    unpackRawMetadata(rawImage, metadata, &bayerPattern, &black_level, &scale_mul, &rgb_cam, auto_white_balance);

    gls::OpenCLContext glsContext("");
    auto clContext = glsContext.clContext();

    LOG_INFO(TAG) << "Begin demosaicing image (GPU, ISO " << iso << ")..." << std::endl;

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
            .lumaSigma = 0.25,
            .cbSigma = 2.0,
            .crSigma = 2.0,
            .sharpening = 1.0
        },
        {
            .lumaSigma = 0.25,
            .cbSigma = 2.0,
            .crSigma = 2.0,
            .sharpening = 1.0
        },
        {
            .lumaSigma = 0.25,
            .cbSigma = 2.0,
            .crSigma = 2.0,
            .sharpening = 1.0
        },
        {
            .lumaSigma = 0.25,
            .cbSigma = 2.0,
            .crSigma = 2.0,
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
    auto clDenoisedImage = pyramidalDenoise.denoise(&glsContext, &denoiseParameters, &despeckledImage, rgb_cam, iso);

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
