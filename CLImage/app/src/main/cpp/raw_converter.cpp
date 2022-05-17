//
//  raw_converter.cpp
//  RawPipeline
//
//  Created by Fabio Riccardi on 5/16/22.
//

#include "raw_converter.hpp"

#include "gls_logging.h"

static const char* TAG = "RAW Converter";

void RawConverter::allocateTextures(gls::OpenCLContext* glsContext, int width, int height) {
    auto clContext = glsContext->clContext();

    if (!clRawImage) {
        clRawImage = std::make_unique<gls::cl_image_2d<gls::luma_pixel_16>>(clContext, width, height);
        clGreenImage = std::make_unique<gls::cl_image_2d<gls::luma_pixel_float>>(clContext, width, height);
        clLinearRGBImage = std::make_unique<gls::cl_image_2d<gls::rgba_pixel_float>>(clContext, width, height);
        clScaledRawImage = std::make_unique<gls::cl_image_2d<gls::luma_pixel_float>>(clContext, width, height);
        clsRGBImage = std::make_unique<gls::cl_image_2d<gls::rgba_pixel>>(clContext, width, height);

        pyramidalDenoise = std::make_unique<PyramidalDenoise<5>>(glsContext, width, height);
    }

}

void RawConverter::allocateHighNoiseTextures(gls::OpenCLContext* glsContext, int width, int height) {
    auto clContext = glsContext->clContext();

    if (!rgbaRawImage) {
        rgbaRawImage = std::make_unique<gls::cl_image_2d<gls::rgba_pixel_float>>(clContext, width/2, height/2);
        denoisedRgbaRawImage = std::make_unique<gls::cl_image_2d<gls::rgba_pixel_float>>(clContext, width/2, height/2);
        despeckledImage = std::make_unique<gls::cl_image_2d<gls::rgba_pixel_float>>(clContext, width, height);
    }
}

void RawConverter::allocateFastDemosaicTextures(gls::OpenCLContext* glsContext, int width, int height) {
    auto clContext = glsContext->clContext();

    if (!clFastLinearRGBImage) {
        clFastLinearRGBImage = std::make_unique<gls::cl_image_2d<gls::rgba_pixel_float>>(clContext, width/2, height/2);
        clsFastRGBImage = std::make_unique<gls::cl_image_2d<gls::rgba_pixel>>(clContext, width/2, height/2);
    }
}

gls::cl_image_2d<gls::rgba_pixel>* RawConverter::demosaicImage(const gls::image<gls::luma_pixel_16>& rawImage,
                                                               DemosaicParameters* demosaicParameters,
                                                               const gls::rectangle* gmb_position, bool rotate_180) {
    auto clContext = _glsContext->clContext();

    LOG_INFO(TAG) << "Begin Demosaicing..." << std::endl;

    NoiseModel* noiseModel = &demosaicParameters->noiseModel;

    // TODO: Unify this
    const float min_green_variance = 5e-05;
    const float max_green_variance = 1.6e-02;
    const float nlf_green_variance = std::clamp(noiseModel->rawNlf[1], min_green_variance, max_green_variance);

    const float nlf_alpha = log2(nlf_green_variance / min_green_variance) / log2(max_green_variance / min_green_variance);
    const bool high_noise_image = nlf_alpha > 0.6;

    LOG_INFO(TAG) << "nlf_green_variance: " << nlf_green_variance << ", nlf_alpha: " << std::fixed << nlf_alpha << std::endl;

    allocateTextures(_glsContext, rawImage.width, rawImage.height);
    if (high_noise_image) {
        allocateHighNoiseTextures(_glsContext, rawImage.width, rawImage.height);
    }

    // Copy input data to the OpenCL input buffer
    auto cpuRawImage = clRawImage->mapImage();
    for (int y = 0; y < clRawImage->height; y++) {
        std::copy(rawImage[y], &rawImage[y][clRawImage->width], cpuRawImage[y]);
    }
    clRawImage->unmapImage(cpuRawImage);

    // --- Image Demosaicing ---

    if (nlf_alpha > 0.6) {
        std::cout << "denoiseRawRGBAImage" << std::endl;

        bayerToRawRGBA(_glsContext, *clRawImage, rgbaRawImage.get(), demosaicParameters->bayerPattern);

        despeckleRawRGBAImage(_glsContext,
                              *rgbaRawImage,
                              denoisedRgbaRawImage.get());

        denoiseRawRGBAImage(_glsContext,
                            *denoisedRgbaRawImage,
                            demosaicParameters->noiseModel.rawNlf,
                            rgbaRawImage.get());

        rawRGBAToBayer(_glsContext, *rgbaRawImage, clRawImage.get(), demosaicParameters->bayerPattern);
    }

    scaleRawData(_glsContext, *clRawImage, clScaledRawImage.get(), demosaicParameters->bayerPattern, demosaicParameters->scale_mul,
                 demosaicParameters->black_level / 0xffff);

    const float raw_sigma_treshold = 4; // nlf_alpha > 0.5 ? 32 : nlf_alpha > 0.3 ? 8 : 4;

    interpolateGreen(_glsContext, *clScaledRawImage, clGreenImage.get(), demosaicParameters->bayerPattern, raw_sigma_treshold * sqrt(noiseModel->rawNlf[1]));

    interpolateRedBlue(_glsContext, *clScaledRawImage, *clGreenImage, clLinearRGBImage.get(), demosaicParameters->bayerPattern, raw_sigma_treshold * sqrt((noiseModel->rawNlf[0] + noiseModel->rawNlf[2]) / 2), rotate_180);

    // --- Image Denoising ---

    // Convert linear image to YCbCr
    auto cam_to_ycbcr = cam_ycbcr(demosaicParameters->rgb_cam);

    std::cout << "cam_to_ycbcr: " << cam_to_ycbcr.span() << std::endl;

    transformImage(_glsContext, *clLinearRGBImage, clLinearRGBImage.get(), cam_to_ycbcr);

    if (nlf_alpha > 0.6) {
        std::cout << "despeckleYCbCrImage" << std::endl;
        applyKernel(_glsContext, "despeckleYCbCrImage", *clLinearRGBImage, despeckledImage.get());
    }

    auto clDenoisedImage = pyramidalDenoise->denoise(_glsContext, &demosaicParameters->denoiseParameters,
                                                     nlf_alpha > 0.6 ? despeckledImage.get() : clLinearRGBImage.get(),
                                                     demosaicParameters->rgb_cam, gmb_position, false,
                                                     &noiseModel->pyramidNlf);

    // Convert result back to camera RGB
    transformImage(_glsContext, *clDenoisedImage, clDenoisedImage, inverse(cam_to_ycbcr));

    // --- Image Post Processing ---

    convertTosRGB(_glsContext, *clDenoisedImage, clsRGBImage.get(), demosaicParameters->rgb_cam, *demosaicParameters); // TODO: ???

    return clsRGBImage.get();
}

gls::cl_image_2d<gls::rgba_pixel>* RawConverter::fastDemosaicImage(const gls::image<gls::luma_pixel_16>& rawImage,
                                                                   const DemosaicParameters& demosaicParameters) {
    allocateFastDemosaicTextures(_glsContext, rawImage.width, rawImage.height);

    LOG_INFO(TAG) << "Begin Fast Demosaicing (GPU)..." << std::endl;

    // --- Image Demosaicing ---

    scaleRawData(_glsContext, *clRawImage, clScaledRawImage.get(), demosaicParameters.bayerPattern, demosaicParameters.scale_mul,
                 demosaicParameters.black_level / 0xffff);

    fasteDebayer(_glsContext, *clScaledRawImage, clFastLinearRGBImage.get(), demosaicParameters.bayerPattern);

    // --- Image Post Processing ---

    convertTosRGB(_glsContext, *clFastLinearRGBImage, clsFastRGBImage.get(), demosaicParameters.rgb_cam, demosaicParameters);

    return clsFastRGBImage.get();
}

/*static*/ gls::image<gls::rgb_pixel>::unique_ptr RawConverter::convertToRGBImage(const gls::cl_image_2d<gls::rgba_pixel>& clRGBAImage) {
    auto rgbImage = std::make_unique<gls::image<gls::rgb_pixel>>(clRGBAImage.width, clRGBAImage.height);
    auto rgbaImage = clRGBAImage.mapImage();
    for (int y = 0; y < clRGBAImage.height; y++) {
        for (int x = 0; x < clRGBAImage.width; x++) {
            const auto& p = rgbaImage[y][x];
            (*rgbImage)[y][x] = { p.red, p.green, p.blue };
        }
    }
    clRGBAImage.unmapImage(rgbaImage);
    return rgbImage;
}
