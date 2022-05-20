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

static const std::array<NoiseModel, 7> RicohGRIII = {{
    {
        { 4.7786e-05, 7.0285e-05, 5.9201e-05, 6.5856e-05 },
        {
            4.5720e-05, 2.3777e-06, 1.5821e-06,
            2.0424e-05, 1.4056e-06, 1.0889e-06,
            1.6773e-05, 9.6919e-07, 8.8397e-07,
            1.4787e-05, 8.6361e-07, 8.3637e-07,
            3.0582e-05, 7.4963e-07, 7.6393e-07,
        },
    },
    {
        { 8.0831e-05, 9.6862e-05, 8.8963e-05, 1.0063e-04 },
        {
            6.6643e-05, 3.6983e-06, 2.2089e-06,
            2.5383e-05, 2.0230e-06, 1.4075e-06,
            1.9858e-05, 1.2260e-06, 1.0517e-06,
            1.6017e-05, 9.7217e-07, 9.3810e-07,
            3.3651e-05, 7.9622e-07, 8.0892e-07,
        },
    },
    {
        { 7.1581e-05, 1.1445e-04, 8.5317e-05, 1.2074e-04 },
        {
            7.0863e-05, 3.3528e-06, 1.8619e-06,
            2.5732e-05, 1.9853e-06, 1.2551e-06,
            1.9308e-05, 1.2932e-06, 9.8890e-07,
            1.5179e-05, 9.1213e-07, 8.2284e-07,
            3.2264e-05, 7.0968e-07, 7.0154e-07,
        },
    },
    {
        { 8.8933e-05, 1.8886e-04, 1.1623e-04, 1.8335e-04 },
        {
            1.0443e-04, 4.1103e-06, 2.3506e-06,
            3.2963e-05, 2.6106e-06, 1.6543e-06,
            2.0633e-05, 1.6881e-06, 1.2421e-06,
            1.6206e-05, 1.1154e-06, 9.9552e-07,
            2.3780e-05, 8.0237e-07, 8.3455e-07,
        },
    },
    {
        { 1.7511e-04, 3.5926e-04, 2.0178e-04, 4.0296e-04 },
        {
            2.0727e-04, 5.9579e-06, 3.2454e-06,
            5.7235e-05, 3.2627e-06, 2.0565e-06,
            3.2934e-05, 2.1103e-06, 1.6711e-06,
            2.4354e-05, 1.5862e-06, 1.4398e-06,
            2.0294e-05, 1.4445e-06, 1.4201e-06,
        },
    },
    {
        { 2.8041e-04, 6.4946e-04, 3.6862e-04, 6.4677e-04 },
        {
            3.6810e-04, 1.0648e-05, 5.4911e-06,
            7.9420e-05, 5.2539e-06, 2.7678e-06,
            2.9848e-05, 2.8308e-06, 1.6249e-06,
            2.0416e-05, 1.8278e-06, 1.2563e-06,
            2.3040e-05, 1.1539e-06, 9.6554e-07,
        },
    },
    {
        { 6.1764e-04, 1.3863e-03, 7.9310e-04, 1.4253e-03 },
        {
            7.9701e-04, 2.3553e-05, 1.2125e-05,
            1.4415e-04, 1.1029e-05, 6.0596e-06,
            4.9362e-05, 5.7188e-06, 3.8819e-06,
            2.0299e-05, 3.7682e-06, 2.9595e-06,
            3.3097e-05, 2.1722e-06, 2.0549e-06,
        },
    },
}
};

template <int levels>
static std::pair<gls::Vector<4>, gls::Matrix<levels, 3>> nlfFromIso(const std::array<NoiseModel, 7>& NLFData, int iso) {
    iso = std::clamp(iso, 100, 50000);
    if (iso >= 100 && iso < 200) {
        float a = (iso - 100) / 100;
        return std::pair(lerpRawNLF(NLFData[0].rawNlf, NLFData[1].rawNlf, a), lerpNLF<levels>(NLFData[0].pyramidNlf, NLFData[1].pyramidNlf, a));
    } else if (iso >= 200 && iso < 400) {
        float a = (iso - 200) / 200;
        return std::pair(lerpRawNLF(NLFData[1].rawNlf, NLFData[2].rawNlf, a), lerpNLF<levels>(NLFData[1].pyramidNlf, NLFData[2].pyramidNlf, a));
    } else if (iso >= 400 && iso < 800) {
        float a = (iso - 400) / 400;
        return std::pair(lerpRawNLF(NLFData[2].rawNlf, NLFData[3].rawNlf, a), lerpNLF<levels>(NLFData[2].pyramidNlf, NLFData[3].pyramidNlf, a));
    } else if (iso >= 800 && iso < 1600) {
        float a = (iso - 800) / 800;
        return std::pair(lerpRawNLF(NLFData[3].rawNlf, NLFData[4].rawNlf, a), lerpNLF<levels>(NLFData[3].pyramidNlf, NLFData[4].pyramidNlf, a));
    } else if (iso >= 1600 && iso < 3200) {
        float a = (iso - 1600) / 1600;
        return std::pair(lerpRawNLF(NLFData[4].rawNlf, NLFData[5].rawNlf, a), lerpNLF<levels>(NLFData[4].pyramidNlf, NLFData[5].pyramidNlf, a));
    } else /* if (iso >= 3200 && iso < 6400) */ {
        float a = (iso - 3200) / 3200;
        return std::pair(lerpRawNLF(NLFData[5].rawNlf, NLFData[6].rawNlf, a), lerpNLF<levels>(NLFData[5].pyramidNlf, NLFData[6].pyramidNlf, a));
    }
}

std::array<DenoiseParameters, 5> RicohGRIIIDenoiseParameters(int iso) {
    const auto nlf_params = nlfFromIso<5>(RicohGRIII, iso);
    // const auto nlf_params = nlfFromIso<5>(RicohGRIII, iso);

    // A reasonable denoising calibration on a fairly large range of Noise Variance values

    const float min_green_variance = 5e-05;
    const float max_green_variance = 1.6e-02;
    const float nlf_green_variance = std::clamp(nlf_params.first[1], min_green_variance, max_green_variance);

    const float nlf_alpha = log2(nlf_green_variance / min_green_variance) / log2(max_green_variance / min_green_variance);

    // Bilateral RicohGRIII
    std::array<DenoiseParameters, 5> denoiseParameters = {{
        {
            .luma = std::lerp(0.25f, 0.5f, nlf_alpha),
            .chroma = std::lerp(1.0f, 4.0f, nlf_alpha),
            .sharpening = std::lerp(2.0f, 1.0f, nlf_alpha)
        },
        {
            .luma = std::lerp(0.75f, 2.0f, nlf_alpha),
            .chroma = std::lerp(0.75f, 4.0f, nlf_alpha),
            .sharpening = 1.1,
        },
        {
            .luma = std::lerp(0.75f, 8.0f, nlf_alpha),
            .chroma = std::lerp(0.75f, 4.0f, nlf_alpha),
            .sharpening = 1.075
        },
        {
            .luma = std::lerp(0.25f, 4.0f, nlf_alpha),
            .chroma = std::lerp(0.25f, 4.0f, nlf_alpha),
            .sharpening = 1.05
        },
        {
            .luma = std::lerp(0.125f, 2.0f, nlf_alpha),
            .chroma = std::lerp(0.125f, 4.0f, nlf_alpha),
            .sharpening = 1
        }
    }};

    // GuidedFast RicohGRIII
//    std::array<DenoiseParameters, 5> denoiseParameters = {{
//        {
//            .luma = 0.25, // std::lerp(0.5f, 1.0f, nlf_alpha),
//            .chroma = std::lerp(1.0f, 4.0f, nlf_alpha),
//            .sharpening = std::lerp(2.0f, 1.0f, nlf_alpha)
//        },
//        {
//            .luma = std::lerp(0.75f, 2.0f, nlf_alpha),
//            .chroma = std::lerp(0.75f, 4.0f, nlf_alpha),
//            .sharpening = 1.1,
//        },
//        {
//            .luma = std::lerp(0.75f, 4.0f, nlf_alpha),
//            .chroma = std::lerp(0.75f, 4.0f, nlf_alpha),
//            .sharpening = 1.075
//        },
//        {
//            .luma = std::lerp(0.25f, 4.0f, nlf_alpha),
//            .chroma = std::lerp(0.25f, 4.0f, nlf_alpha),
//            .sharpening = 1.05
//        },
//        {
//            .luma = std::lerp(0.125f, 2.0f, nlf_alpha),
//            .chroma = std::lerp(0.125f, 4.0f, nlf_alpha),
//            .sharpening = 1
//        }
//    }};

    return denoiseParameters;
}

gls::image<gls::rgb_pixel>::unique_ptr calibrateRicohGRIII(RawConverter* rawConverter,
                                                        const std::filesystem::path& input_path,
                                                        DemosaicParameters* demosaicParameters,
                                                        int iso, const gls::rectangle& gmb_position) {
    gls::tiff_metadata dng_metadata, exif_metadata;
    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    unpackDNGMetadata(*inputImage, &dng_metadata, demosaicParameters, /*auto_white_balance=*/ false, &gmb_position, /*rotate_180=*/ false);

    // See if the ISO value is present and override
    const auto exifIsoSpeedRatings = getVector<uint16_t>(exif_metadata, EXIFTAG_ISOSPEEDRATINGS);
    if (exifIsoSpeedRatings.size() > 0) {
        iso = exifIsoSpeedRatings[0];
    }

    demosaicParameters->denoiseParameters = RicohGRIIIDenoiseParameters(iso);

    return RawConverter::convertToRGBImage(*rawConverter->demosaicImage(*inputImage, demosaicParameters, &gmb_position, /*rotate_180=*/ false));
}

void calibrateRicohGRIII(RawConverter* rawConverter, const std::filesystem::path& input_dir) {
    struct CalibrationEntry {
        int iso;
        const char* fileName;
        gls::rectangle gmb_position;
        bool rotated;
    };

    std::array<CalibrationEntry, 7> calibration_files = {{
        { 100,  "R0000914_ISO100.DNG", { 2520, 560, 1123, 733 }, false },
        { 200,  "R0000917_ISO200.DNG", { 2520, 560, 1123, 733 }, false },
        { 400,  "R0000920_ISO400.DNG", { 2520, 560, 1123, 733 }, false },
        { 800,  "R0000923_ISO800.DNG", { 2520, 560, 1123, 733 }, false },
        { 1600, "R0000926_ISO1600.DNG", { 2520, 560, 1123, 733 }, false },
        { 3200, "R0000929_ISO3200.DNG", { 2520, 560, 1123, 733 }, false },
        { 6400, "R0000932_ISO6400.DNG", { 2520, 560, 1123, 733 }, false },
    }};

    std::array<NoiseModel, 7> noiseModel;

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

        const auto rgb_image = calibrateRicohGRIII(rawConverter, input_path, &demosaicParameters, entry.iso, entry.gmb_position);
        rgb_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_cal_bilateral.png", /*skip_alpha=*/ true);

        noiseModel[i] = demosaicParameters.noiseModel;
    }

    std::cout << "Calibration table for RicohGRIII:" << std::endl;
    for (int i = 0; i < calibration_files.size(); i++) {
        std::cout << "{" << std::endl;
        std::cout << "{ " << noiseModel[i].rawNlf << " }," << std::endl;
        std::cout << "{\n" << noiseModel[i].pyramidNlf << "\n}," << std::endl;
        std::cout << "}," << std::endl;
    }
}

gls::image<gls::rgb_pixel>::unique_ptr demosaicRicohGRIII2DNG(RawConverter* rawConverter, const std::filesystem::path& input_path) {
    DemosaicParameters demosaicParameters = {
        .rgbConversionParameters = {
            .contrast = 1.05,
            .saturation = 1.0,
            .toneCurveSlope = 3.5,
        }
    };

    gls::tiff_metadata dng_metadata, exif_metadata;
    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    unpackDNGMetadata(*inputImage, &dng_metadata, &demosaicParameters, /*auto_white_balance=*/ false, nullptr /* &gmb_position */, /*rotate_180=*/ false);

    const auto iso = getVector<uint16_t>(exif_metadata, EXIFTAG_ISOSPEEDRATINGS)[0];

    std::cout << "EXIF ISO: " << iso << std::endl;

    const auto nlfParams = nlfFromIso<5>(RicohGRIII, iso);
    demosaicParameters.noiseModel.rawNlf = nlfParams.first;
    demosaicParameters.noiseModel.pyramidNlf = nlfParams.second;
    demosaicParameters.denoiseParameters = RicohGRIIIDenoiseParameters(iso);

    return RawConverter::convertToRGBImage(*rawConverter->demosaicImage(*inputImage, &demosaicParameters, nullptr /* &gmb_position */, /*rotate_180=*/ false));
}
