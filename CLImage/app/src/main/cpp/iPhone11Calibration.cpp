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

#include "CameraCalibration.hpp"

#include "demosaic.hpp"
#include "raw_converter.hpp"

#include <array>
#include <cmath>
#include <filesystem>

static const std::array<NoiseModel, 8> iPhone11 = {{
    // ISO032
    {
        { 1.5279e-04, 1.8825e-04, 2.2834e-04, 1.5218e-04 },
        {
            1.7252e-04, 1.2586e-05, 1.2092e-05,
            -3.1729e-06, 8.0061e-06, 7.1075e-06,
            4.0282e-03, 6.5319e-06, 4.7422e-06,
            5.6949e-03, 4.6350e-06, 3.8942e-06,
            2.6685e-02, 1.4205e-05, 1.4249e-05,
        },
    },
    // ISO064
    {
        { 2.6600e-04, 2.7194e-04, 3.7025e-04, 2.3536e-04 },
        {
            2.7170e-04, 1.8841e-05, 1.8687e-05,
            8.9377e-05, 9.9648e-06, 9.4584e-06,
            2.7749e-03, 8.2764e-06, 6.4766e-06,
            4.5861e-03, 5.7424e-06, 4.7833e-06,
            2.7280e-03, 3.8391e-06, 3.7749e-06,
        },
    },
    // ISO100
    {
        { 3.2022e-04, 4.6378e-04, 5.9206e-04, 3.7185e-04 },
        {
            3.8221e-04, 2.7431e-05, 2.6623e-05,
            7.2573e-05, 1.3106e-05, 1.1431e-05,
            2.9356e-03, 9.2385e-06, 7.8553e-06,
            4.8081e-03, 5.8524e-06, 5.6465e-06,
            1.3567e-02, 4.4086e-06, 5.0561e-06,
        },
    },
    // ISO200
    {
        { 6.5752e-04, 6.9900e-04, 1.0152e-03, 5.8636e-04 },
        {
            6.4875e-04, 4.6680e-05, 4.6704e-05,
            9.5738e-05, 2.1914e-05, 1.9060e-05,
            3.5293e-03, 1.0967e-05, 8.9429e-06,
            6.3428e-03, 5.7878e-06, 5.2719e-06,
            2.4205e-02, 1.2117e-05, 1.2182e-05,
        },
    },
    // ISO400
    {
        { 9.2214e-04, 8.8440e-04, 1.4322e-03, 6.9149e-04 },
        {
            1.2427e-03, 9.1770e-05, 9.2261e-05,
            3.1766e-04, 3.7475e-05, 3.8989e-05,
            1.1625e-04, 1.8365e-05, 1.8792e-05,
            4.1966e-03, 7.6197e-06, 8.9006e-06,
            1.1675e-04, 4.9945e-06, 4.5667e-06,
        },
    },
    // ISO800
    {
        { 9.2300e-04, 6.9199e-04, 1.0921e-03, 6.4538e-04 },
        {
            2.1387e-03, 1.9895e-04, 1.5968e-04,
            4.5572e-04, 8.2960e-05, 5.8384e-05,
            3.4226e-03, 2.6549e-05, 1.7463e-05,
            5.0686e-03, 7.7143e-06, 5.9241e-06,
            2.4267e-02, 9.3522e-06, 1.0058e-05,
        },
    },
    // ISO1600
    {
        { 8.6908e-04, 7.5243e-04, 9.9176e-04, 6.9243e-04 },
        {
            4.8238e-03, 4.2850e-04, 3.7777e-04,
            9.3238e-04, 1.4431e-04, 1.2732e-04,
            1.8783e-02, 4.6528e-05, 3.3383e-05,
            3.9787e-02, 1.1362e-05, 8.9265e-06,
            2.2135e-02, 1.3014e-05, 1.2821e-05,
        },
    },
    // ISO2500
    {
        { 6.8090e-04, 6.8328e-04, 1.1562e-03, 6.6299e-04 },
        {
            6.6625e-03, 6.6268e-04, 6.4364e-04,
            1.2048e-03, 2.6456e-04, 2.2050e-04,
            1.0772e-03, 9.1665e-05, 6.0799e-05,
            2.6334e-03, 3.0128e-05, 2.5044e-05,
            6.9326e-04, 8.9970e-06, 8.5887e-06,
        },
    },
}};

template <int levels>
std::pair<gls::Vector<4>, gls::Matrix<levels, 3>> nlfFromIsoiPhone(const std::array<NoiseModel, 8>& NLFData, int iso) {
    iso = std::clamp(iso, 32, 2500);
    if (iso >= 32 && iso < 64) {
        float a = (iso - 32) / 32;
        return std::pair(lerpRawNLF(NLFData[0].rawNlf, NLFData[1].rawNlf, a), lerpNLF<levels>(NLFData[0].pyramidNlf, NLFData[1].pyramidNlf, a));
    } else if (iso >= 64 && iso < 100) {
        float a = (iso - 64) / 36;
        return std::pair(lerpRawNLF(NLFData[1].rawNlf, NLFData[2].rawNlf, a), lerpNLF<levels>(NLFData[1].pyramidNlf, NLFData[2].pyramidNlf, a));
    } else if (iso >= 100 && iso < 200) {
        float a = (iso - 100) / 100;
        return std::pair(lerpRawNLF(NLFData[2].rawNlf, NLFData[3].rawNlf, a), lerpNLF<levels>(NLFData[2].pyramidNlf, NLFData[3].pyramidNlf, a));
    } else if (iso >= 200 && iso < 400) {
        float a = (iso - 200) / 200;
        return std::pair(lerpRawNLF(NLFData[3].rawNlf, NLFData[4].rawNlf, a), lerpNLF<levels>(NLFData[3].pyramidNlf, NLFData[4].pyramidNlf, a));
    } else if (iso >= 400 && iso < 800) {
        float a = (iso - 400) / 400;
        return std::pair(lerpRawNLF(NLFData[4].rawNlf, NLFData[5].rawNlf, a), lerpNLF<levels>(NLFData[4].pyramidNlf, NLFData[5].pyramidNlf, a));
    } else if (iso >= 800 && iso < 1600) {
        float a = (iso - 800) / 800;
        return std::pair(lerpRawNLF(NLFData[5].rawNlf, NLFData[6].rawNlf, a), lerpNLF<levels>(NLFData[5].pyramidNlf, NLFData[6].pyramidNlf, a));
    } /* else if (iso >= 1600 && iso < 2500) */ {
        float a = (iso - 1600) / 900;
        return std::pair(lerpRawNLF(NLFData[6].rawNlf, NLFData[7].rawNlf, a), lerpNLF<levels>(NLFData[6].pyramidNlf, NLFData[7].pyramidNlf, a));
    }
}

std::array<DenoiseParameters, 5> iPhone11DenoiseParameters(int iso, float varianceBoost) {
    const auto nlf_params = nlfFromIsoiPhone<5>(iPhone11, iso);

    // A reasonable denoising calibration on a fairly large range of Noise Variance values

    const float min_green_variance = 1.2586e-05;
    const float max_green_variance = 6.6268e-04;
    const float nlf_green_variance = std::clamp(nlf_params.second[0][1], min_green_variance, max_green_variance);

    // const float nlf_alpha = log2(nlf_green_variance / min_green_variance) / log2(max_green_variance / min_green_variance);

    const float nlf_alpha = (nlf_green_variance - min_green_variance) / (max_green_variance - min_green_variance);

    const float iso_alpha = std::clamp((nlf_alpha - 0.2) / 0.8, 0.0, 1.0);

    std::cout << "iPhone11DenoiseParameters for ISO " << iso << ", nlf_alpha: " << nlf_alpha << ", iso_alpha: " << iso_alpha << std::endl;

    std::array<DenoiseParameters, 5> denoiseParameters = {{
        {
            .luma = 0.125, // 1 * std::lerp(0.5f, 1.0f, iso_alpha),
            .chroma = 1.0f * std::lerp(1.0f, 16.0f, iso_alpha),
            .sharpening = 1
        },
        {
            .luma = 0.75f * std::lerp(1.0f, 1.0f, iso_alpha),
            .chroma = 0.75f * std::lerp(1.0f, 16.0f, iso_alpha),
            .sharpening = 1
        },
        {
            .luma = 0.5f * std::lerp(1.0f, 1.0f, iso_alpha),
            .chroma = 0.5f * std::lerp(1.0f, 8.0f, iso_alpha),
            .sharpening = 1
        },
        {
            .luma = 0.25f * std::lerp(1.0f, 1.0f, iso_alpha),
            .chroma = 0.25f * std::lerp(1.0f, 8.0f, iso_alpha),
            .sharpening = 1
        },
        {
            .luma = 0.25f * std::lerp(1.0f, 1.0f, iso_alpha),
            .chroma = 0.25f * std::lerp(1.0f, 8.0f, iso_alpha),
            .sharpening = 1
        }
    }};

    return denoiseParameters;
}

gls::image<gls::rgb_pixel>::unique_ptr demosaiciPhone11(RawConverter* rawConverter, const std::filesystem::path& input_path) {
    DemosaicParameters demosaicParameters = {
        .rgbConversionParameters = {
            .contrast = 1.05,
            .saturation = 1.0,
            .toneCurveSlope = 3.5,
        }
    };

    gls::tiff_metadata dng_metadata, exif_metadata;
    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    float exposure_multiplier = unpackDNGMetadata(*inputImage, &dng_metadata, &demosaicParameters, /*auto_white_balance=*/ false, nullptr /* &gmb_position */, /*rotate_180=*/ false);

    const auto iso = getVector<uint16_t>(exif_metadata, EXIFTAG_ISOSPEEDRATINGS)[0];

    std::cout << "EXIF ISO: " << iso << std::endl;

    const auto nlfParams = nlfFromIsoiPhone<5>(iPhone11, iso);
    demosaicParameters.noiseModel.rawNlf = nlfParams.first;
    demosaicParameters.noiseModel.pyramidNlf = nlfParams.second;
    demosaicParameters.denoiseParameters = iPhone11DenoiseParameters(iso, exposure_multiplier);

    return RawConverter::convertToRGBImage(*rawConverter->demosaicImage(*inputImage, &demosaicParameters, nullptr /* &gmb_position */, /*rotate_180=*/ false));
}

gls::image<gls::rgb_pixel>::unique_ptr calibrateiPhone11(RawConverter* rawConverter,
                                                         const std::filesystem::path& input_path,
                                                         DemosaicParameters* demosaicParameters,
                                                         int iso, const gls::rectangle& gmb_position) {
    gls::tiff_metadata dng_metadata, exif_metadata;
    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    float exposure_multiplier = unpackDNGMetadata(*inputImage, &dng_metadata, demosaicParameters, /*auto_white_balance=*/ false, &gmb_position, /*rotate_180=*/ false);

    // See if the ISO value is present and override
    const auto exifIsoSpeedRatings = getVector<uint16_t>(exif_metadata, EXIFTAG_ISOSPEEDRATINGS);
    if (exifIsoSpeedRatings.size() > 0) {
        iso = exifIsoSpeedRatings[0];
    }

    demosaicParameters->denoiseParameters = iPhone11DenoiseParameters(iso, exposure_multiplier);

    return RawConverter::convertToRGBImage(*rawConverter->demosaicImage(*inputImage, demosaicParameters, &gmb_position, /*rotate_180=*/ false));
}

void calibrateiPhone11(RawConverter* rawConverter, const std::filesystem::path& input_dir) {
    struct CalibrationEntry {
        int iso;
        const char* fileName;
        gls::rectangle gmb_position;
        bool rotated;
    };

    std::array<CalibrationEntry, 8> calibration_files = {{
        { 32,   "IPHONE11hSLI0032NRD.dng", { 1798, 2199, 382, 269 }, false },
        { 64,   "IPHONE11hSLI0064NRD.dng", { 1799, 2200, 382, 269 }, false },
        { 100,  "IPHONE11hSLI0100NRD.dng", { 1800, 2200, 382, 269 }, false },
        { 200,  "IPHONE11hSLI0200NRD.dng", { 1796, 2199, 382, 269 }, false },
        { 400,  "IPHONE11hSLI0400NRD.dng", { 1796, 2204, 382, 269 }, false },
        { 800,  "IPHONE11hSLI0800NRD.dng", { 1795, 2199, 382, 269 }, false },
        { 1600, "IPHONE11hSLI1600NRD.dng", { 1793, 2195, 382, 269 }, false },
        { 2500, "IPHONE11hSLI2500NRD.dng", { 1794, 2200, 382, 269 }, false }
    }};

    std::array<NoiseModel, 10> noiseModel;

    for (int i = 0; i < calibration_files.size(); i++) {
        auto& entry = calibration_files[i];
        const auto input_path = input_dir / entry.fileName;

        DemosaicParameters demosaicParameters = {
            .rgbConversionParameters = {
                .contrast = 1.05,
                .saturation = 1.0,
                .toneCurveSlope = 3.5,
            }
        };

        const auto rgb_image = calibrateiPhone11(rawConverter, input_path, &demosaicParameters, entry.iso, entry.gmb_position);
        rgb_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_cal_rawnr_rgb.png", /*skip_alpha=*/ true);

        noiseModel[i] = demosaicParameters.noiseModel;
    }

    std::cout << "Calibration table for iPhone 11:" << std::endl;
    for (int i = 0; i < calibration_files.size(); i++) {
        std::cout << "{" << std::endl;
        std::cout << "{ " << noiseModel[i].rawNlf << " }," << std::endl;
        std::cout << "{\n" << noiseModel[i].pyramidNlf << "\n}," << std::endl;
        std::cout << "}," << std::endl;
    }
}
