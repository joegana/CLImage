//
//  pyramidal_denoise.cpp
//  RawPipeline
//
//  Created by Fabio Riccardi on 5/16/22.
//

#include "pyramidal_denoise.hpp"

template
PyramidalDenoise<5>::PyramidalDenoise(gls::OpenCLContext* glsContext, int width, int height, DenoiseAlgorithm denoiseAlgorithm);

template
typename PyramidalDenoise<5>::imageType* PyramidalDenoise<5>::denoise(gls::OpenCLContext* glsContext, std::array<DenoiseParameters, 5>* denoiseParameters,
                                                                      imageType* image, const gls::Matrix<3, 3>& rgb_cam,
                                                                      const gls::rectangle* gmb_position, bool rotate_180,
                                                                      gls::Matrix<5, 3>* nlfParameters);

struct BilateralDenoiser : ImageDenoiser {
    BilateralDenoiser(gls::OpenCLContext* glsContext, int width, int height) : ImageDenoiser(glsContext, width, height) { }

    void denoise(gls::OpenCLContext* glsContext,
              const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
              const gls::Vector<3>& sigma, int pyramidLevel,
              gls::cl_image_2d<gls::rgba_pixel_float>* outputImage) override {
        ::denoiseImage(glsContext, inputImage, sigma, pyramidLevel == 0 ? false : true, outputImage);
    }
};

struct GuidedFastDenoiser : ImageDenoiser {
    GuidedFastDenoiser(gls::OpenCLContext* glsContext, int width, int height) : ImageDenoiser(glsContext, width, height) { }

    void denoise(gls::OpenCLContext* glsContext,
                  const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                  const gls::Vector<3>& sigma, int pyramidLevel,
                  gls::cl_image_2d<gls::rgba_pixel_float>* outputImage) override {
        ::denoiseImageGuided(glsContext, inputImage, sigma, outputImage);
    }
};

struct GuidedPreciseDenoiser : ImageDenoiser {
    GuidedFilter guidedFilter;

    GuidedPreciseDenoiser(gls::OpenCLContext* glsContext, int width, int height) :
        ImageDenoiser(glsContext, width, height),
        guidedFilter(glsContext, width, height) { }

    void denoise(gls::OpenCLContext* glsContext,
                  const gls::cl_image_2d<gls::rgba_pixel_float>& inputImage,
                  const gls::Vector<3>& sigma, int pyramidLevel,
                  gls::cl_image_2d<gls::rgba_pixel_float>* outputImage) override {
        guidedFilter.filter(glsContext, inputImage, /*filterSize=*/ 5, sigma, outputImage);
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
                if (i > 1) {
                    denoiser[i] = std::make_unique<BilateralDenoiser>(glsContext, width/scale, height/scale);
                } else {
                    denoiser[i] = std::make_unique<GuidedFastDenoiser>(glsContext, width/scale, height/scale);
                }
                break;
            case GuidedPrecise:
                denoiser[i] = std::make_unique<GuidedPreciseDenoiser>(glsContext, width/scale, height/scale);
                break;
        }
    }
}

template <size_t levels>
typename PyramidalDenoise<levels>::imageType* PyramidalDenoise<levels>::denoise(gls::OpenCLContext* glsContext, std::array<DenoiseParameters, levels>* denoiseParameters,
                   imageType* image, const gls::Matrix<3, 3>& rgb_cam,
                   const gls::rectangle* gmb_position, bool rotate_180,
                   gls::Matrix<levels, 3>* nlfParameters) {
    const bool calibrate_nlf = gmb_position != nullptr;

    if (calibrate_nlf) {
        auto cpuImage = image->toImage();
        (*nlfParameters)[0] = extractNlfFromColorChecker(cpuImage.get(), *gmb_position, rotate_180, 1);
        (*denoiseParameters)[0].luma = (*denoiseParameters)[0].luma * (*denoiseParameters)[0].luma * (*nlfParameters)[0][0];
        (*denoiseParameters)[0].chroma = (*denoiseParameters)[0].chroma * (*denoiseParameters)[0].chroma * (*nlfParameters)[0][1];
    }

    for (int i = 0, scale = 2; i < levels-1; i++, scale *= 2) {
        resampleImage(glsContext, "downsampleImage", i == 0 ? *image : *imagePyramid[i - 1], imagePyramid[i].get());

        if (calibrate_nlf) {
            auto cpuLayer = imagePyramid[i]->toImage();
            (*nlfParameters)[i+1] = extractNlfFromColorChecker(cpuLayer.get(), *gmb_position, rotate_180, scale);
            (*denoiseParameters)[i+1].luma = (*denoiseParameters)[i+1].luma * (*denoiseParameters)[i+1].luma * (*nlfParameters)[i+1][0];
            (*denoiseParameters)[i+1].chroma = (*denoiseParameters)[i+1].chroma * (*denoiseParameters)[i+1].chroma * (*nlfParameters)[i+1][1];
        }
    }

    if (!calibrate_nlf) {
        for (int i = 0; i < levels; i++) {
            (*denoiseParameters)[i].luma = (*denoiseParameters)[i].luma * (*denoiseParameters)[i].luma * (*nlfParameters)[i][0];
            (*denoiseParameters)[i].chroma = (*denoiseParameters)[i].chroma * (*denoiseParameters)[i].chroma * (*nlfParameters)[i][1];
        }
    }

    // Denoise the bottom of the image pyramid
    const auto& np = (*denoiseParameters)[levels-1];
    denoiser[levels-1]->denoise(glsContext, *(imagePyramid[levels-2]),
                                { np.luma, np.chroma, np.chroma }, /*pyramidLevel=*/ levels-1,
                                denoisedImagePyramid[levels-1].get());

    for (int i = levels - 2; i >= 0; i--) {
        // Denoise current layer
        const auto& np = (*denoiseParameters)[i];
        denoiser[i]->denoise(glsContext, i > 0 ? *(imagePyramid[i - 1]) : *image,
                             { np.luma, np.chroma, np.chroma }, /*pyramidLevel=*/ i,
                             denoisedImagePyramid[i].get());

        std::cout << "Reassembling layer " << i << " with sharpening: " << (*denoiseParameters)[i].sharpening << std::endl;
        // Subtract noise from previous layer
        reassembleImage(glsContext, *(denoisedImagePyramid[i]), *(imagePyramid[i]), *(denoisedImagePyramid[i+1]),
                        (*denoiseParameters)[i].sharpening, sqrt((*denoiseParameters)[i].luma), denoisedImagePyramid[i].get());
    }

    return denoisedImagePyramid[0].get();
}
