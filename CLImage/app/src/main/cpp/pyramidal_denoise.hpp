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

#ifndef pyramidal_denoise_h
#define pyramidal_denoise_h

#include "demosaic.hpp"
#include "demosaic_cl.hpp"

#include "guided_filter.hpp"

template <size_t levels>
struct PyramidalDenoise {
    enum DenoiseAlgorithm { Bilateral, GuidedFast, GuidedPrecise };

    DenoiseAlgorithm denoiseAlgorithm;

    typedef gls::cl_image_2d<gls::rgba_pixel_float> imageType;
    std::array<imageType::unique_ptr, levels-1> imagePyramid;
    std::array<imageType::unique_ptr, levels> denoisedImagePyramid;

    std::array<std::unique_ptr<GuidedFilter>, levels> guidedFilter;

    PyramidalDenoise(gls::OpenCLContext* glsContext, const imageType& image, DenoiseAlgorithm _denoiseAlgorithm = GuidedPrecise) : denoiseAlgorithm(_denoiseAlgorithm) {
        for (int i = 0, scale = 2; i < levels-1; i++, scale *= 2) {
            imagePyramid[i] = std::make_unique<imageType>(glsContext->clContext(), image.width/scale, image.height/scale);
        }
        for (int i = 0, scale = 1; i < levels; i++, scale *= 2) {
            denoisedImagePyramid[i] = std::make_unique<imageType>(glsContext->clContext(), image.width/scale, image.height/scale);
            if (denoiseAlgorithm == GuidedPrecise) {
                guidedFilter[i] = std::make_unique<GuidedFilter>(glsContext, image.width/scale, image.height/scale);
            }
        }
    }

    imageType* denoise(gls::OpenCLContext* glsContext, std::array<DenoiseParameters, levels>* denoiseParameters,
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
        switch (denoiseAlgorithm) {
            case Bilateral:
                denoiseImage(glsContext, *(imagePyramid[levels-2]),
                             { sqrt(np.luma), sqrt(np.chroma), sqrt(np.chroma) }, true,
                             denoisedImagePyramid[levels-1].get());
                break;
            case GuidedFast:
                denoiseImageGuided(glsContext, *(imagePyramid[levels-2]),
                                   { np.luma, np.chroma, np.chroma },
                                   denoisedImagePyramid[levels-1].get());
                break;
            case GuidedPrecise:
                guidedFilter[levels-1]->filter(glsContext, *(imagePyramid[levels-2]),
                                               5, {np.luma, np.chroma, np.chroma },
                                               denoisedImagePyramid[levels-1].get());
                break;
        }

        for (int i = levels - 2; i >= 0; i--) {
            // Denoise current layer
            const auto& np = (*denoiseParameters)[i];
            switch (denoiseAlgorithm) {
                case Bilateral:
                    denoiseImage(glsContext, i > 0 ? *(imagePyramid[i - 1]) : *image,
                                 { sqrt(np.luma), sqrt(np.chroma), sqrt(np.chroma) }, i == 0 ? false : true,
                                 denoisedImagePyramid[i].get());
                    break;
                case GuidedFast:
                    denoiseImageGuided(glsContext, i > 0 ? *(imagePyramid[i - 1]) : *image,
                                       { np.luma, np.chroma, np.chroma },
                                       denoisedImagePyramid[i].get());
                    break;
                case GuidedPrecise:
                    guidedFilter[i]->filter(glsContext, i > 0 ? *(imagePyramid[i - 1]) : *image,
                                            5, {np.luma, np.chroma, np.chroma },
                                            denoisedImagePyramid[i].get());
                    break;
            }

            std::cout << "Reassembling layer " << i << " with sharpening: " << (*denoiseParameters)[i].sharpening << std::endl;
            // Subtract noise from previous layer
            reassembleImage(glsContext, *(denoisedImagePyramid[i]), *(imagePyramid[i]), *(denoisedImagePyramid[i+1]),
                            (*denoiseParameters)[i].sharpening, sqrt((*denoiseParameters)[i].luma), denoisedImagePyramid[i].get());
        }

        return denoisedImagePyramid[0].get();
    }
};

#endif /* pyramidal_denoise_h */
