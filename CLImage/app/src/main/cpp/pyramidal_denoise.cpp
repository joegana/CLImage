//
//  pyramidal_denoise.cpp
//  RawPipeline
//
//  Created by Fabio Riccardi on 5/16/22.
//

#include "pyramidal_denoise.hpp"

#include <iomanip>

template
PyramidalDenoise<5>::PyramidalDenoise(gls::OpenCLContext* glsContext, int width, int height, DenoiseAlgorithm denoiseAlgorithm);

template
typename PyramidalDenoise<5>::imageType* PyramidalDenoise<5>::denoise(gls::OpenCLContext* glsContext, std::array<DenoiseParameters, 5>* denoiseParameters,
                                                                      imageType* image, const gls::Matrix<3, 3>& rgb_cam,
                                                                      const gls::rectangle* gmb_position, bool rotate_180,
                                                                      gls::Matrix<5, 6>* nlfParameters);

struct BilateralDenoiser : ImageDenoiser {
    BilateralDenoiser(gls::OpenCLContext* glsContext, int width, int height) : ImageDenoiser(glsContext, width, height) { }

    void denoise(gls::OpenCLContext* glsContext,
              const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
              const gls::Vector<3>& var_a, const gls::Vector<3>& var_b, int pyramidLevel,
              gls::cl_image_2d<gls::rgba_pixel_float>* outputImage) override {
        ::denoiseImage(glsContext, inputImage, var_a, var_b, pyramidLevel == 0 ? false : true, outputImage);
    }
};

struct GuidedFastDenoiser : ImageDenoiser {
    GuidedFastDenoiser(gls::OpenCLContext* glsContext, int width, int height) : ImageDenoiser(glsContext, width, height) { }

    void denoise(gls::OpenCLContext* glsContext,
                  const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                  const gls::Vector<3>& var_a, const gls::Vector<3>& var_b, int pyramidLevel,
                  gls::cl_image_2d<gls::rgba_pixel_float>* outputImage) override {
        ::denoiseImageGuided(glsContext, inputImage, var_a, var_b, outputImage);
    }
};

struct GuidedPreciseDenoiser : ImageDenoiser {
    GuidedFilter guidedFilter;

    GuidedPreciseDenoiser(gls::OpenCLContext* glsContext, int width, int height) :
        ImageDenoiser(glsContext, width, height),
        guidedFilter(glsContext, width, height) { }

    void denoise(gls::OpenCLContext* glsContext,
                  const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                  const gls::Vector<3>& var_a, const gls::Vector<3>& var_b, int pyramidLevel,
                  gls::cl_image_2d<gls::rgba_pixel_float>* outputImage) override {
        guidedFilter.filter(glsContext, inputImage, /*filterSize=*/ 5, var_b, outputImage);
    }
};

template <size_t levels>
PyramidalDenoise<levels>::PyramidalDenoise(gls::OpenCLContext* glsContext, int width, int height, DenoiseAlgorithm denoiseAlgorithm) {
    for (int i = 0, scale = 2; i < levels-1; i++, scale *= 2) {
        imagePyramid[i] = std::make_unique<imageType>(glsContext->clContext(), width/scale, height/scale);
    }
    for (int i = 0, scale = 1; i < levels; i++, scale *= 2) {
        denoisedImagePyramid[i] = std::make_unique<imageType>(glsContext->clContext(), width/scale, height/scale);

        switch (denoiseAlgorithm) {
            case Bilateral:
                denoiser[i] = std::make_unique<BilateralDenoiser>(glsContext, width/scale, height/scale);
                break;
            case GuidedFast:
                if (i < levels - 1) {
                    denoiser[i] = std::make_unique<GuidedFastDenoiser>(glsContext, width/scale, height/scale);
                } else {
                    denoiser[i] = std::make_unique<BilateralDenoiser>(glsContext, width/scale, height/scale);
                }
                break;
            case GuidedPrecise:
                denoiser[i] = std::make_unique<GuidedPreciseDenoiser>(glsContext, width/scale, height/scale);
                break;
        }
    }
}

gls::Vector<6> computeNoiseStatistics(gls::OpenCLContext* glsContext, const gls::cl_image_2d<gls::rgba_pixel_float>& image);

gls::Vector<6> nflMultiplier(const DenoiseParameters &denoiseParameters) {
    float luma_mul = denoiseParameters.luma * denoiseParameters.luma;
    float chroma_mul = denoiseParameters.chroma * denoiseParameters.chroma;
    return { luma_mul, chroma_mul, chroma_mul, luma_mul, chroma_mul, chroma_mul };
}

template <size_t levels>
typename PyramidalDenoise<levels>::imageType* PyramidalDenoise<levels>::denoise(gls::OpenCLContext* glsContext,
                                                                                std::array<DenoiseParameters, levels>* denoiseParameters,
                                                                                imageType* image, const gls::Matrix<3, 3>& rgb_cam,
                                                                                const gls::rectangle* gmb_position, bool rotate_180,
                                                                                gls::Matrix<levels, 6>* nlfParameters) {
    const bool calibrate_nlf = gmb_position != nullptr;

    std::array<std::array<float, 6>, levels> calibrated_nlf;

    for (int i = 0; i < levels; i++) {
        if (i < levels - 1) {
            resampleImage(glsContext, "downsampleImage", i == 0 ? *image : *imagePyramid[i - 1], imagePyramid[i].get());
        }

        if (calibrate_nlf) {
            (*nlfParameters)[i] = computeNoiseStatistics(glsContext, i == 0 ? *image : *(imagePyramid[i - 1]));
        }

        calibrated_nlf[i] = (*nlfParameters)[i] * nflMultiplier((*denoiseParameters)[i]);
    }

    // Denoise the bottom of the image pyramid
    const auto& np = calibrated_nlf[levels-1];
    denoiser[levels-1]->denoise(glsContext, *(imagePyramid[levels-2]),
                                { np[0], np[1], np[2] }, { np[3], np[4], np[5] }, /*pyramidLevel=*/ levels-1,
                                denoisedImagePyramid[levels-1].get());

    for (int i = levels - 2; i >= 0; i--) {
        gls::cl_image_2d<gls::rgba_pixel_float>* denoiseInput = i > 0 ? imagePyramid[i - 1].get() : image;

        // Denoise current layer
        const auto& np = calibrated_nlf[i];
        denoiser[i]->denoise(glsContext, *denoiseInput,
                             { np[0], np[1], np[2] }, { np[3], np[4], np[5] }, /*pyramidLevel=*/ i,
                             denoisedImagePyramid[i].get());

        std::cout << "Reassembling layer " << i << " with sharpening: " << (*denoiseParameters)[i].sharpening << std::endl;
        // Subtract noise from previous layer
        reassembleImage(glsContext, *(denoisedImagePyramid[i]), *(imagePyramid[i]), *(denoisedImagePyramid[i+1]),
                        (*denoiseParameters)[i].sharpening, {np[0], np[3]}, denoisedImagePyramid[i].get());
    }

    return denoisedImagePyramid[0].get();
}

gls::DVector<2> nlfChannel(const gls::DVector<4>& sum, const gls::DVector<4>& sumSq, double N, size_t channel) {
    double b = (N * sumSq[channel] - sum[0] * sum[channel]) / (N * sumSq[0] - sum[0] * sum[0]);
    double a = (sum[channel] - b * sum[0]) / N;
    return {{ a, b }};
}

gls::Vector<6> computeNoiseStatistics(gls::OpenCLContext* glsContext, const gls::cl_image_2d<gls::rgba_pixel_float>& image) {
    gls::cl_image_2d<gls::rgba_pixel_float> noiseStats(glsContext->clContext(), image.width, image.height);
    applyKernel(glsContext, "noiseStatistics", image, &noiseStats);
    const auto noiseStatsCpu = noiseStats.mapImage();

    // Only consider pixels with variance lower than the expected noise value
    const double lumaVarianceMax = 0.001;
    // Limit to pixels the more linear intensity zone of the sensor
    const double lumaIntensityMax = 0.5;
    const double lumaIntensityMin = 0.001;

    // Collect pixel statistics
    gls::DVector<4> sum = {{ 0, 0, 0, 0 }};
    gls::DVector<4> sumSq = {{ 0, 0, 0, 0 }};
    double N = 0;
    noiseStatsCpu.apply([&](const gls::rgba_pixel_float& pp) {
        gls::DVector<4> p = {{ pp[0], pp[1], pp[2], pp[3] }};
        if (p[0] > lumaIntensityMin && p[0] < lumaIntensityMax && p[1] < lumaVarianceMax) {
            sum += p;
            sumSq += p[0] * p;
            N++;
        }
    });

    // Linear regression on pixel statistics to extract a linear noise model: nlf = A + B * Y
    auto nlfY = nlfChannel(sum, sumSq, N, 1);
    auto nlfCb = nlfChannel(sum, sumSq, N, 2);
    auto nlfCr = nlfChannel(sum, sumSq, N, 3);
    gls::DVector<3> nlfA = {{ nlfY[0], nlfCb[0], nlfCr[0] }};
    gls::DVector<3> nlfB = {{ nlfY[1], nlfCb[1], nlfCr[1] }};

    // Estimate regression mean square error
    gls::DVector<3> err2 = {{ 0, 0, 0 }};
    noiseStatsCpu.apply([&](const gls::rgba_pixel_float& p){
        if (p[0] > lumaIntensityMin && p[0] < lumaIntensityMax && p[1] < lumaVarianceMax) {
            auto nlfP = nlfA + nlfB * (double) p[0];

            auto diff = nlfP - gls::DVector<3>{{ p[1], p[2], p[3] }};
            err2 += diff * diff;
        }
    });
    err2 /= N;

//    std::cout << "\nnlf_Y a: " << std::setprecision(4) << std::scientific << nlfA[0] << ", b: " << nlfB[0] << ", err2: " << err2[0]
//              << "\nnlf_Cb a: " << nlfA[1] << ", b: " << nlfB[1] << ", err2: " << err2[1]
//              << "\nnlf_Cr a: " << nlfA[2] << ", b: " << nlfB[2] << ", err2: " << err2[2]
//              << " on " << std::setprecision(1) << std::fixed << 100 * N / (image.width * image.height) << "% pixels" << std::endl;

    // Redo the statistics collection limiting the sample to pixels that fit well the linear model
    sum = {{ 0, 0, 0, 0 }};
    sumSq = {{ 0, 0, 0, 0 }};
    N = 0;
    gls::DVector<3> newErr2 = {{ 0, 0, 0 }};
    noiseStatsCpu.apply([&](const gls::rgba_pixel_float& pp){
        gls::DVector<4> p = {{ pp[0], pp[1], pp[2], pp[3] }};
        if (p[0] > lumaIntensityMin && p[0] < lumaIntensityMax && p[1] < lumaVarianceMax) {
            auto nlfP = nlfA + nlfB * (double) p[0];
            auto diff = nlfP - gls::DVector<3>{{ p[1], p[2], p[3] }};
            diff *= diff;

            // Stay within the linear fit error
            if (diff[0] <= err2[0] && diff[1] <= err2[1] && diff[2] <= err2[2]) {
                sum += p;
                sumSq += p[0] * p;
                N++;
                newErr2 += diff;
            }
        }
    });
    newErr2 /= N;

    // Estimate the new regression parameters
    nlfY = nlfChannel(sum, sumSq, N, 1);
    nlfCb = nlfChannel(sum, sumSq, N, 2);
    nlfCr = nlfChannel(sum, sumSq, N, 3);
    nlfA = {{ nlfY[0], nlfCb[0], nlfCr[0] }};
    nlfB = {{ nlfY[1], nlfCb[1], nlfCr[1] }};

    std::cout << "\nnlf_Y a: " << std::setprecision(4) << std::scientific << nlfA[0] << ", b: " << nlfB[0] << ", err2: " << newErr2[0]
              << "\nnlf_Cb a: " << nlfA[1] << ", b: " << nlfB[1] << ", err2: " << newErr2[1]
              << "\nnlf_Cr a: " << nlfA[2] << ", b: " << nlfB[2] << ", err2: " << newErr2[2]
              << " on " << std::setprecision(1) << std::fixed << 100 * N / (image.width * image.height) << "% pixels" << std::endl;

    assert(err2[0] > newErr2[0] && err2[1] > newErr2[1] && err2[2] > newErr2[2]);

    noiseStats.unmapImage(noiseStatsCpu);

    return {
        (float) nlfA[0], (float) nlfA[1], (float) nlfA[2], // A values
        (float) nlfB[0], (float) nlfB[1], (float) nlfB[2]  // B values
    };
}
