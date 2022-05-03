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

template <size_t levels>
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
                       imageType* image, const gls::Matrix<3, 3>& rgb_cam, int iso,
                       const gls::rectangle* gmb_position, bool rotate_180,
                       std::array<std::array<float, 3>, levels>* nlfParameters) {
        const bool calibrate_nlf = gmb_position != nullptr;

        if (calibrate_nlf) {
            auto cpuImage = image->toImage();
            (*nlfParameters)[0] = extractNlfFromColorChecker(cpuImage.get(), *gmb_position, rotate_180, 1);
            (*denoiseParameters)[0].lumaSigma *= sqrt((*nlfParameters)[0][0]);
            (*denoiseParameters)[0].cbSigma *= sqrt((*nlfParameters)[0][1]);
            (*denoiseParameters)[0].crSigma *= sqrt((*nlfParameters)[0][2]);
        }

        for (int i = 0, scale = 2; i < levels-1; i++, scale *= 2) {
            resampleImage(glsContext, "downsampleImage", i == 0 ? *image : *imagePyramid[i - 1], imagePyramid[i].get());

            if (calibrate_nlf) {
                auto cpuLayer = imagePyramid[i]->toImage();
                (*nlfParameters)[i+1] = extractNlfFromColorChecker(cpuLayer.get(), *gmb_position, rotate_180, scale);
                (*denoiseParameters)[i+1].lumaSigma *= sqrt((*nlfParameters)[i+1][0]);
                (*denoiseParameters)[i+1].cbSigma *= sqrt((*nlfParameters)[i+1][1]);
                (*denoiseParameters)[i+1].crSigma *= sqrt((*nlfParameters)[i+1][2]);
            }
        }

        if (!calibrate_nlf) {
            for (int i = 0; i < levels; i++) {
                (*denoiseParameters)[i].lumaSigma *= sqrt((*nlfParameters)[i][0]);
                (*denoiseParameters)[i].cbSigma *= sqrt((*nlfParameters)[i][1]);
                (*denoiseParameters)[i].crSigma *= sqrt((*nlfParameters)[i][2]);
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

#endif /* pyramidal_denoise_h */
