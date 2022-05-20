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

#include <array>
#include <cmath>
#include <filesystem>

static const std::array<NoiseModel, 10> LeicaQ2 = {{
    {
        { 4.1532e-05, 5.1049e-05, 4.9132e-05, 5.1023e-05 },
        {
            4.1312e-05, 1.0074e-05, 9.4039e-06,
            1.6662e-05, 8.9870e-06, 8.7271e-06,
            1.2807e-05, 8.4670e-06, 8.3917e-06,
            1.3421e-05, 8.4607e-06, 8.4367e-06,
            1.9038e-05, 8.7060e-06, 8.6979e-06,
        },
    },
    {
        { 7.5068e-05, 8.8781e-05, 8.2993e-05, 8.7150e-05 },
        {
            6.7970e-05, 1.1851e-05, 1.0426e-05,
            2.0293e-05, 9.6843e-06, 9.0915e-06,
            1.4108e-05, 8.6619e-06, 8.4703e-06,
            1.2932e-05, 8.4371e-06, 8.3854e-06,
            1.9436e-05, 8.6586e-06, 8.6350e-06,
        },
    },
    {
        { 1.4439e-04, 1.5661e-04, 1.5666e-04, 1.5110e-04 },
        {
            1.1609e-04, 1.5258e-05, 1.2506e-05,
            2.4162e-05, 1.0841e-05, 9.7919e-06,
            1.1994e-05, 8.8801e-06, 8.5668e-06,
            1.0726e-05, 8.4753e-06, 8.3922e-06,
            1.9235e-05, 8.4539e-06, 8.4401e-06,
        },
    },
    {
        { 2.8469e-04, 3.0171e-04, 2.8955e-04, 3.1108e-04 },
        {
            2.3128e-04, 2.1952e-05, 1.6445e-05,
            4.4189e-05, 1.3393e-05, 1.1182e-05,
            1.8693e-05, 9.6161e-06, 8.9189e-06,
            1.4783e-05, 8.5263e-06, 8.3429e-06,
            2.1452e-05, 8.4616e-06, 8.4081e-06,
        },
    },
    {
        { 5.5517e-04, 5.6939e-04, 5.5710e-04, 5.6703e-04 },
        {
            4.3019e-04, 3.4346e-05, 2.3918e-05,
            7.0441e-05, 1.7729e-05, 1.3690e-05,
            2.2866e-05, 1.0344e-05, 9.2142e-06,
            1.7440e-05, 8.8394e-06, 8.5461e-06,
            2.2038e-05, 8.6508e-06, 8.5369e-06,
        },
    },
    {
        { 1.0746e-03, 1.1374e-03, 1.0807e-03, 1.0729e-03 },
        {
            8.1951e-04, 7.1597e-05, 4.8875e-05,
            1.2823e-04, 3.5454e-05, 2.6665e-05,
            3.9349e-05, 2.0165e-05, 1.7603e-05,
            1.9780e-05, 1.5566e-05, 1.4709e-05,
            1.8258e-05, 1.3133e-05, 1.2844e-05,
        },
    },
    {
        { 2.2234e-03, 2.2147e-03, 2.1940e-03, 2.1301e-03 },
        {
            8.4795e-04, 1.0960e-04, 7.2299e-05,
            1.9413e-04, 5.7065e-05, 4.0752e-05,
            6.4663e-05, 2.9441e-05, 2.4181e-05,
            2.9165e-05, 1.9559e-05, 1.8103e-05,
            1.9366e-05, 1.9547e-05, 1.8951e-05,
        },
    },
    {
        { 4.2268e-03, 4.6509e-03, 4.4503e-03, 4.5066e-03 },
        {
            1.7452e-03, 2.0121e-04, 1.2382e-04,
            3.8566e-04, 9.6547e-05, 6.0973e-05,
            1.1020e-04, 3.9928e-05, 2.8150e-05,
            3.6234e-05, 1.9695e-05, 1.6357e-05,
            1.4462e-05, 1.5014e-05, 1.3909e-05,
        },
    },
    {
        { 7.8770e-03, 8.5880e-03, 8.0832e-03, 8.6783e-03 },
        {
            3.4289e-03, 4.1534e-04, 2.2744e-04,
            7.5898e-04, 1.9094e-04, 1.0786e-04,
            2.2905e-04, 7.4909e-05, 4.4680e-05,
            7.9700e-05, 3.0407e-05, 2.3642e-05,
            3.0847e-05, 2.0387e-05, 1.9512e-05,
        },
    },
    {
        { 1.8687e-02, 1.6641e-02, 1.6274e-02, 1.6624e-02 },
        {
            6.9207e-03, 1.0348e-03, 5.4667e-04,
            1.3573e-03, 4.5881e-04, 2.4605e-04,
            3.5136e-04, 1.6777e-04, 9.9908e-05,
            8.1214e-05, 6.5037e-05, 4.8126e-05,
            3.4860e-05, 4.4629e-05, 4.1289e-05,
        },
    },
}
};

template <int levels>
static std::pair<gls::Vector<4>, gls::Matrix<levels, 3>> nlfFromIso(const std::array<NoiseModel, 10>& NLFData, int iso) {
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
    } else if (iso >= 3200 && iso < 6400) {
        float a = (iso - 3200) / 3200;
        return std::pair(lerpRawNLF(NLFData[5].rawNlf, NLFData[6].rawNlf, a), lerpNLF<levels>(NLFData[5].pyramidNlf, NLFData[6].pyramidNlf, a));
    } else if (iso >= 6400 && iso < 12500) {
        float a = (iso - 6100) / 6100;
        return std::pair(lerpRawNLF(NLFData[6].rawNlf, NLFData[7].rawNlf, a), lerpNLF<levels>(NLFData[6].pyramidNlf, NLFData[7].pyramidNlf, a));
    } else if (iso >= 12500 && iso < 25000) {
        float a = (iso - 12500) / 12500;
        return std::pair(lerpRawNLF(NLFData[7].rawNlf, NLFData[8].rawNlf, a), lerpNLF<levels>(NLFData[7].pyramidNlf, NLFData[8].pyramidNlf, a));
    } else /* if (iso >= 25000 && iso <= 50000) */ {
        float a = (iso - 25000) / 25000;
        return std::pair(lerpRawNLF(NLFData[8].rawNlf, NLFData[9].rawNlf, a), lerpNLF<levels>(NLFData[8].pyramidNlf, NLFData[9].pyramidNlf, a));
    }
}

std::array<DenoiseParameters, 5> LeicaQ2DenoiseParameters(int iso) {
    const auto nlf_params = nlfFromIso<5>(LeicaQ2, iso);

    // A reasonable denoising calibration on a fairly large range of Noise Variance values

    const float min_green_variance = 5e-05;
    const float max_green_variance = 1.6e-02;
    const float nlf_green_variance = std::clamp(nlf_params.first[1], min_green_variance, max_green_variance);

    const float nlf_alpha = log2(nlf_green_variance / min_green_variance) / log2(max_green_variance / min_green_variance);

    // Bilateral
    std::array<DenoiseParameters, 5> denoiseParameters = {{
        {
            .luma = std::lerp(0.5f, 1.0f, nlf_alpha),
            .chroma = std::lerp(1.0f, 8.0f, nlf_alpha),
            .sharpening = std::lerp(1.5f, 0.7f, nlf_alpha)
        },
        {
            .luma = std::lerp(0.75f, 1.0f, nlf_alpha),
            .chroma = std::lerp(0.75f, 4.0f, nlf_alpha),
            .sharpening = std::lerp(1.0f, 0.8f, nlf_alpha),
        },
        {
            .luma = std::lerp(0.75f, 4.0f, nlf_alpha),
            .chroma = std::lerp(0.75f, 4.0f, nlf_alpha),
            .sharpening = 1
        },
        {
            .luma = std::lerp(0.25f, 4.0f, nlf_alpha),
            .chroma = std::lerp(0.25f, 2.0f, nlf_alpha),
            .sharpening = 1
        },
        {
            .luma = std::lerp(0.125f, 2.0f, nlf_alpha),
            .chroma = std::lerp(0.125f, 2.0f, nlf_alpha),
            .sharpening = 1
        }
    }};

//    // GuidedFast
//    std::array<DenoiseParameters, 5> denoiseParameters = {{
//        {
//            .luma = 0.5f, // std::lerp(0.5f, 1.0f, nlf_alpha),
//            .chroma = std::lerp(1.0f, 16.0f, nlf_alpha),
//            .sharpening = std::lerp(1.5f, 0.7f, nlf_alpha)
//        },
//        {
//            .luma = std::lerp(0.75f, 1.0f, nlf_alpha),
//            .chroma = std::lerp(0.75f, 8.0f, nlf_alpha),
//            .sharpening = std::lerp(1.0f, 0.8f, nlf_alpha),
//        },
//        {
//            .luma = std::lerp(0.75f, 4.0f, nlf_alpha),
//            .chroma = std::lerp(0.75f, 4.0f, nlf_alpha),
//            .sharpening = 1
//        },
//        {
//            .luma = std::lerp(0.25f, 4.0f, nlf_alpha),
//            .chroma = std::lerp(0.25f, 2.0f, nlf_alpha),
//            .sharpening = 1
//        },
//        {
//            .luma = std::lerp(0.125f, 2.0f, nlf_alpha),
//            .chroma = std::lerp(0.125f, 2.0f, nlf_alpha),
//            .sharpening = 1
//        }
//    }};

    return denoiseParameters;
}

gls::image<gls::rgb_pixel>::unique_ptr calibrateLeicaQ2(RawConverter* rawConverter,
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

    demosaicParameters->denoiseParameters = LeicaQ2DenoiseParameters(iso);

    return RawConverter::convertToRGBImage(*rawConverter->demosaicImage(*inputImage, demosaicParameters, &gmb_position, /*rotate_180=*/ false));
}

void calibrateLeicaQ2(RawConverter* rawConverter, const std::filesystem::path& input_dir) {
    struct CalibrationEntry {
        int iso;
        const char* fileName;
        gls::rectangle gmb_position;
        bool rotated;
    };

    std::array<CalibrationEntry, 10> calibration_files = {{
        { 100,   "L1010611.DNG", { 3430, 800, 1549, 1006 }, false },
        { 200,   "L1010614.DNG", { 3430, 800, 1549, 1006 }, false },
        { 400,   "L1010617.DNG", { 3430, 800, 1549, 1006 }, false },
        { 800,   "L1010620.DNG", { 3430, 800, 1549, 1006 }, false },
        { 1600,  "L1010623.DNG", { 3430, 800, 1549, 1006 }, false },
        { 3200,  "L1010626.DNG", { 3430, 800, 1549, 1006 }, false },
        { 6400,  "L1010629.DNG", { 3430, 800, 1549, 1006 }, false },
        { 12500, "L1010632.DNG", { 3430, 800, 1549, 1006 }, false },
        { 25000, "L1010635.DNG", { 3430, 800, 1549, 1006 }, false },
        { 50000, "L1010638.DNG", { 3430, 800, 1549, 1006 }, false },
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

        const auto rgb_image = calibrateLeicaQ2(rawConverter, input_path, &demosaicParameters, entry.iso, entry.gmb_position);
        rgb_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_cal.png", /*skip_alpha=*/ true);

        noiseModel[i] = demosaicParameters.noiseModel;
    }

    std::cout << "Calibration table for LeicaQ2:" << std::endl;
    for (int i = 0; i < calibration_files.size(); i++) {
        std::cout << "{" << std::endl;
        std::cout << "{ " << noiseModel[i].rawNlf << " }," << std::endl;
        std::cout << "{\n" << noiseModel[i].pyramidNlf << "\n}," << std::endl;
        std::cout << "}," << std::endl;
    }
}

gls::image<gls::rgb_pixel>::unique_ptr demosaicLeicaQ2DNG(RawConverter* rawConverter, const std::filesystem::path& input_path) {
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

    const auto nlfParams = nlfFromIso<5>(LeicaQ2, iso);
    demosaicParameters.noiseModel.rawNlf = nlfParams.first;
    demosaicParameters.noiseModel.pyramidNlf = nlfParams.second;
    demosaicParameters.denoiseParameters = LeicaQ2DenoiseParameters(iso);

    return RawConverter::convertToRGBImage(*rawConverter->demosaicImage(*inputImage, &demosaicParameters, nullptr /* &gmb_position */, /*rotate_180=*/ false));
}
