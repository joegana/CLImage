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

//5.4107e-05, 3.5267e-06, 3.2098e-06
//1.9530e-05, 2.7585e-06, 2.6017e-06
//8.6886e-06, 2.1277e-06, 2.0838e-06
//4.2190e-06, 1.8742e-06, 1.8643e-06
//2.2800e-06, 1.7923e-06, 1.7893e-06

static const float NLF_IMX492[6][5][3] = {
    // ISO 100
    {
        { 4.9688e-05, 7.3893e-06, 7.2321e-06 },
        { 1.7964e-05, 6.7149e-06, 6.6420e-06 },
        { 9.6783e-06, 6.1881e-06, 6.1788e-06 },
        { 6.0442e-06, 5.9603e-06, 5.9610e-06 },
        { 4.8379e-06, 5.9137e-06, 5.9116e-06 }
    },
    // ISO 200
    {
        { 9.3366e-05, 1.7681e-05, 1.7271e-05 },
        { 3.2632e-05, 1.5746e-05, 1.5601e-05 },
        { 1.5216e-05, 1.4169e-05, 1.4172e-05 },
        { 8.6612e-06, 1.3534e-05, 1.3546e-05 },
        { 6.5333e-06, 1.3421e-05, 1.3422e-05 }
    },
    // ISO 400
    {
        { 1.7849e-04, 1.3025e-05, 1.2329e-05 },
        { 5.8104e-05, 1.0410e-05, 1.0084e-05 },
        { 2.2722e-05, 8.2682e-06, 8.2175e-06 },
        { 1.0522e-05, 7.4140e-06, 7.4126e-06 },
        { 6.8284e-06, 7.2507e-06, 7.2523e-06 }
    },
    // ISO 800
    {
        { 3.4231e-04, 1.4014e-05, 1.2784e-05 },
        { 1.0282e-04, 9.0075e-06, 8.5997e-06 },
        { 3.2621e-05, 5.0612e-06, 5.0776e-06 },
        { 1.1993e-05, 3.4746e-06, 3.5347e-06 },
        { 9.5276e-06, 3.0545e-06, 3.0695e-06 }
    },
    // ISO 1600
    {
        { 6.9735e-04, 4.0897e-05, 3.7083e-05 },
        { 2.1621e-04, 2.5779e-05, 2.4293e-05 },
        { 6.9542e-05, 1.3638e-05, 1.3477e-05 },
        { 2.4663e-05, 8.5841e-06, 8.5838e-06 },
        { 1.6873e-05, 7.2737e-06, 7.3092e-06 }
    },
    // ISO 3200
    {
        { 1.0793e-03, 4.7093e-05, 4.2599e-05 },
        { 4.1953e-04, 2.9475e-05, 2.7927e-05 },
        { 1.3875e-04, 1.5666e-05, 1.5445e-05 },
        { 4.0298e-05, 9.8978e-06, 9.8024e-06 },
        { 1.5069e-05, 8.1720e-06, 8.0665e-06 }
    },
};

static const float NLF_IMX492V2[6][5][3] = {
    // ISO 100
    {
        { 1.1817e-04, 7.6074e-05, 7.2803e-05 },
        { 5.8687e-05, 5.8594e-05, 5.7425e-05 },
        { 4.2721e-05, 4.8179e-05, 4.7825e-05 },
        { 4.1860e-05, 4.3691e-05, 4.3555e-05 },
        { 3.9072e-05, 4.2436e-05, 4.2401e-05 }
    },
    // ISO 200
    {
        { 1.9630e-04, 1.0408e-04, 9.8109e-05 },
        { 7.9714e-05, 7.0176e-05, 6.7911e-05 },
        { 4.8079e-05, 4.9691e-05, 4.9017e-05 },
        { 4.2638e-05, 4.1427e-05, 4.1235e-05 },
        { 3.8430e-05, 3.8918e-05, 3.8855e-05 }
    },
    // ISO 400
    {
        { 3.3663e-04, 1.6478e-04, 1.5480e-04 },
        { 1.1355e-04, 9.9108e-05, 9.5292e-05 },
        { 5.7377e-05, 5.9418e-05, 5.8118e-05 },
        { 4.5197e-05, 4.3951e-05, 4.3608e-05 },
        { 4.0192e-05, 3.9776e-05, 3.9616e-05 }
    },
    // ISO 800
    {
        { 5.7325e-04, 6.4165e-04, 6.1982e-04 },
        { 1.7906e-04, 5.0318e-04, 4.9554e-04 },
        { 7.7888e-05, 4.1861e-04, 4.1637e-04 },
        { 5.7175e-05, 3.8530e-04, 3.8466e-04 },
        { 4.9265e-05, 3.7448e-04, 3.7418e-04 }
    },
    // ISO 1600
    {
        { 1.0741e-03, 8.7726e-04, 8.4156e-04 },
        { 2.8819e-04, 5.9887e-04, 5.8824e-04 },
        { 1.0201e-04, 4.3271e-04, 4.3060e-04 },
        { 6.2075e-05, 3.7145e-04, 3.7136e-04 },
        { 5.0261e-05, 3.5170e-04, 3.5205e-04 }
    },
    // ISO 3200
    {
        { 2.1776e-03, 1.4413e-03, 1.3562e-03 },
        { 5.5808e-04, 8.5935e-04, 8.3257e-04 },
        { 1.8079e-04, 5.1135e-04, 5.0549e-04 },
        { 8.7985e-05, 3.7854e-04, 3.7755e-04 },
        { 5.6157e-05, 3.4096e-04, 3.4040e-04 }
    }
};

template <int levels=5>
std::array<std::array<float, 3>, levels> lerpNLF(const float NLFData0[levels][3], const float NLFData1[levels][3], float a) {
    std::array<std::array<float, 3>, levels> result;
    for (int j = 0; j < levels; j++) {
        for (int i = 0; i < 3; i++) {
            result[j][i] = std::lerp(NLFData0[j][i], NLFData1[j][i], a);
        }
    }
    return result;
}

template <int levels=5>
std::array<std::array<float, 3>, levels> nlf(const float NLFData[6][levels][3], int iso) {
    iso = std::clamp(iso, 100, 3200);
    if (iso >= 100 && iso < 200) {
        float a = (iso - 100) / 100;
        return lerpNLF(NLFData[0], NLFData[1], a);
    } else if (iso >= 200 && iso < 400) {
        float a = (iso - 200) / 200;
        return lerpNLF(NLFData[1], NLFData[2], a);
    } else if (iso >= 400 && iso < 800) {
        float a = (iso - 400) / 400;
        return lerpNLF(NLFData[2], NLFData[3], a);
    } else if (iso >= 800 && iso < 1600) {
        float a = (iso - 800) / 800;
        return lerpNLF(NLFData[3], NLFData[4], a);
    } else /* if (iso >= 1600 && iso <= 3200) */ {
        float a = (iso - 1600) / 1600;
        return lerpNLF(NLFData[4], NLFData[5], a);
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
                       imageType* image, const gls::Matrix<3, 3>& rgb_cam, int iso, const gls::rectangle* gmb_position, bool rotate_180) {
        const bool calibrate_nlf = gmb_position != nullptr;

        if (calibrate_nlf) {
            auto cpuImage = image->toImage();
            const auto nlfParameters = extractNlfFromColorChecker(cpuImage.get(), *gmb_position, rotate_180, 1);
            (*denoiseParameters)[0].lumaSigma *= sqrt(nlfParameters[0]);
            (*denoiseParameters)[0].cbSigma *= sqrt(nlfParameters[1]);
            (*denoiseParameters)[0].crSigma *= sqrt(nlfParameters[2]);
        }

        for (int i = 0, scale = 2; i < levels-1; i++, scale *= 2) {
            resampleImage(glsContext, "downsampleImage", i == 0 ? *image : *imagePyramid[i - 1], imagePyramid[i].get());

            if (calibrate_nlf) {
                auto cpuLayer = imagePyramid[i]->toImage();
                const auto nlfParameters = extractNlfFromColorChecker(cpuLayer.get(), *gmb_position, rotate_180, scale);
                (*denoiseParameters)[i+1].lumaSigma *= sqrt(nlfParameters[0]);
                (*denoiseParameters)[i+1].cbSigma *= sqrt(nlfParameters[1]);
                (*denoiseParameters)[i+1].crSigma *= sqrt(nlfParameters[2]);
            }
        }

        if (!calibrate_nlf) {
            auto nlf_params = nlf(NLF_IMX492, iso);
            for (int i = 0; i < levels; i++) {
                // TODO: Further parametrize this
                (*denoiseParameters)[i].lumaSigma *= sqrt(nlf_params[i][0]);
                (*denoiseParameters)[i].cbSigma *= sqrt(nlf_params[i][1]);
                (*denoiseParameters)[i].crSigma *= sqrt(nlf_params[i][2]);
            }
        }

        // Denoise the bottom of the image pyramid
        denoiseImage(glsContext, *(imagePyramid[levels-2]),
                     (*denoiseParameters)[levels-1], /*tight=*/ true,
                     denoisedImagePyramid[levels-1].get());

        for (int i = levels - 2; i >= 0; i--) {
            // Denoise current layer
            denoiseImage(glsContext, i > 0 ? *(imagePyramid[i - 1]) : *image,
                         (*denoiseParameters)[i], /*tight=*/ i > 0 ? true : false,
                         denoisedImagePyramid[i].get());

            // Subtract noise from previous layer
            reassembleImage(glsContext, *(denoisedImagePyramid[i]), *(imagePyramid[i]), *(denoisedImagePyramid[i+1]),
                            (*denoiseParameters)[i].sharpening, (*denoiseParameters)[i].lumaSigma, denoisedImagePyramid[i].get());
        }

        return denoisedImagePyramid[0].get();
    }
};

template <typename T>
gls::rectangle rotate180(const gls::rectangle& rect, const gls::image<T>& image) {
    return {
        image.width - rect.x - rect.width,
        image.height - rect.y - rect.height,
        rect.width,
        rect.height
    };
}

gls::image<gls::rgba_pixel>::unique_ptr demosaicImage(const gls::image<gls::luma_pixel_16>& rawImage, gls::tiff_metadata* metadata,
                                                      const DemosaicParameters& demosaicParameters, int iso, bool auto_white_balance) {
    auto t_start = std::chrono::high_resolution_clock::now();

    BayerPattern bayerPattern;
    float black_level;
    float white_level;
    gls::Vector<4> scale_mul;
    gls::Matrix<3, 3> rgb_cam;

    unpackRawMetadata(rawImage, metadata, &bayerPattern, &black_level, &white_level, &scale_mul, &rgb_cam, auto_white_balance);

//    // L1010626.DNG
//    bool rotate_180 = false;
//    const gls::rectangle gmb_position = {3441, 790, 1531, 991};

    // /Users/fabio/work/IMX492V2Calibration/noise-d65-iso-100.dng
    // bool rotate_180 = false;
    // const gls::rectangle gmb_position = {2871, 1823, 2821, 1965}; // noise-d65-iso-100.dng
    // const gls::rectangle gmb_position = {2954, 1365, 2834, 1925}; // noise-d65-iso-800.dng
    // const gls::rectangle gmb_position = {3057, 1314, 2360, 1577}; // noise-d65-iso-100.dng

    bool rotate_180 = false;
//    const gls::rectangle rotated_gmb_position = { 4537-10, 2351, 1652, 1068 };
//    const gls::rectangle gmb_position = rotate180(rotated_gmb_position, rawImage);

    const gls::rectangle gmb_position = { 2530, 1773, 2792, 1875 }; // IMX492V2Calibration 1
    // const gls::rectangle gmb_position = { 2593, 1766, 2935, 1893 }; // IMX492V2Calibration 2

    gls::Matrix<3, 3> cam_xyz;
    gls::Vector<3> pre_mul;
    const auto raw_nlf = estimateRawParameters(rawImage, &cam_xyz, &pre_mul, black_level, white_level, bayerPattern, gmb_position, rotate_180);

//    auto nlf_params = nlf(NLF_IMX492, iso);
//    std::array<float, 3> raw_nlf = { nlf_params[0][0], nlf_params[0][0], nlf_params[0][0] };

    gls::OpenCLContext glsContext("");
    auto clContext = glsContext.clContext();

    LOG_INFO(TAG) << "Begin demosaicing image..." << std::endl;

    // --- Image Demosaicing ---

    gls::cl_image_2d<gls::luma_pixel_16> clRawImage(clContext, rawImage);
    gls::cl_image_2d<gls::luma_pixel_float> clScaledRawImage(clContext, rawImage.width, rawImage.height);
    scaleRawData(&glsContext, clRawImage, &clScaledRawImage, bayerPattern, scale_mul, black_level / 0xffff);

    const float min_green_variance = 5e-05;
    const float max_green_variance = 5e-03;
    const float nlf_green_variance = std::clamp(raw_nlf[1], min_green_variance, max_green_variance);

    const float nlf_alpha = log2(nlf_green_variance / min_green_variance) / log2(max_green_variance / min_green_variance);

    LOG_INFO(TAG) << "ISO " << iso << ", nlf_green_variance: " << nlf_green_variance << ", nlf_alpha: " << std::fixed << nlf_alpha << std::endl;

    const float raw_sigma_treshold = nlf_alpha > 0.8 ? 32 : nlf_alpha > 0.6 ? 8 : 4;

    gls::cl_image_2d<gls::luma_pixel_float> clGreenImage(clContext, rawImage.width, rawImage.height);
    interpolateGreen(&glsContext, clScaledRawImage, &clGreenImage, bayerPattern, raw_sigma_treshold * sqrt(raw_nlf[1]));

    gls::cl_image_2d<gls::rgba_pixel_float> clLinearRGBImage(clContext, rawImage.width, rawImage.height);
    interpolateRedBlue(&glsContext, clScaledRawImage, clGreenImage, &clLinearRGBImage, bayerPattern, raw_sigma_treshold * sqrt((raw_nlf[0] + raw_nlf[2]) / 2), false);

    // --- Image Denoising ---

    std::array<DenoiseParameters, 5> denoiseParameters = {{
        {
            .lumaSigma = std::lerp(0.25f, 0.75f, nlf_alpha),
            .cbSigma = 4,
            .crSigma = 4,
            .sharpening = std::lerp(1.2f, 0.7f, nlf_alpha)
        },
        {
            .lumaSigma = std::lerp(0.25f, 1.5f, nlf_alpha),
            .cbSigma = 4,
            .crSigma = 4,
            .sharpening = std::lerp(1.2f, 1.0f, nlf_alpha)
        },
        {
            .lumaSigma = std::lerp(0.25f, 3.0f, nlf_alpha),
            .cbSigma = 4,
            .crSigma = 4,
            .sharpening = 1.0
        },
        {
            .lumaSigma = std::lerp(0.25f, 3.0f, nlf_alpha),
            .cbSigma = std::lerp(2.0f, 4.0f, nlf_alpha),
            .crSigma = std::lerp(2.0f, 4.0f, nlf_alpha),
            .sharpening = 1.0
        },
        {
            .lumaSigma = std::lerp(0.25f, 3.0f, nlf_alpha),
            .cbSigma = std::lerp(2.0f, 4.0f, nlf_alpha),
            .crSigma = std::lerp(2.0f, 4.0f, nlf_alpha),
            .sharpening = 1.0
        }
    }};

    // Convert linear image to YCbCr
    const auto cam_to_ycbcr = cam_ycbcr(rgb_cam);

    std::cout << "cam_to_ycbcr:\n" << cam_to_ycbcr << std::endl;

    transformImage(&glsContext, clLinearRGBImage, &clLinearRGBImage, cam_to_ycbcr);

    gls::cl_image_2d<gls::rgba_pixel_float>::unique_ptr despeckledImage = nullptr;
    if (nlf_alpha > 0.6) {
        despeckledImage = std::make_unique<gls::cl_image_2d<gls::rgba_pixel_float>>(glsContext.clContext(), clLinearRGBImage.width, clLinearRGBImage.height);
        applyKernel(&glsContext, "despeckleYCbCrImage", clLinearRGBImage, despeckledImage.get());
    }

    PyramidalDenoise<5> pyramidalDenoise(&glsContext, despeckledImage ? *despeckledImage : clLinearRGBImage);
    auto clDenoisedImage = pyramidalDenoise.denoise(&glsContext, &denoiseParameters, despeckledImage ? despeckledImage.get() : &clLinearRGBImage, rgb_cam, iso, &gmb_position, rotate_180);

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
    float white_level;
    gls::Vector<4> scale_mul;
    gls::Matrix<3, 3> rgb_cam;

    unpackRawMetadata(rawImage, metadata, &bayerPattern, &black_level, &white_level, &scale_mul, &rgb_cam, auto_white_balance);

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
