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

#include <filesystem>
#include <string>
#include <cmath>
#include <chrono>

#include "gls_logging.h"
#include "gls_image.hpp"

#include "raw_converter.hpp"

#include "demosaic.hpp"
#include "gls_tiff_metadata.hpp"

#include "gls_linalg.hpp"

static const char* TAG = "RawPipeline Test";

static const std::array<NoiseModel, 6> NLF_IMX492 = {
{
    // ISO 100
    {
        { 5.8995e-05, 5.4218e-05, 5.1745e-05, 5.1355e-05 },
        {
            5.3481e-05, 3.5281e-06, 3.2405e-06,
            1.8786e-05, 2.7665e-06, 2.6178e-06,
            7.1357e-06, 2.1609e-06, 2.1094e-06,
            2.8133e-06, 1.8945e-06, 1.8811e-06,
            2.4673e-06, 1.7760e-06, 1.7709e-06,
        },
    },
    // ISO 200
    {
        { 1.0815e-04, 9.9500e-05, 9.7302e-05, 1.0157e-04 },
        {
            1.0065e-04, 8.8690e-06, 8.0997e-06,
            3.5935e-05, 6.7345e-06, 6.3653e-06,
            1.3741e-05, 5.0173e-06, 4.9229e-06,
            5.7054e-06, 4.3027e-06, 4.2826e-06,
            4.2223e-06, 4.0360e-06, 4.0372e-06,
        },
    },
    // ISO 400
    {
        { 2.0683e-04, 1.9365e-04, 1.9477e-04, 1.9835e-04 },
        {
            1.9074e-04, 9.3163e-06, 8.1737e-06,
            6.4521e-05, 6.4864e-06, 5.8953e-06,
            2.2293e-05, 4.0991e-06, 3.8989e-06,
            7.6019e-06, 3.0536e-06, 3.0201e-06,
            4.1372e-06, 2.6878e-06, 2.6857e-06,
        },
    },
    // ISO 800
    {
        { 4.2513e-04, 3.7694e-04, 4.0732e-04, 3.8399e-04 },
        {
            3.5420e-04, 1.5293e-05, 1.3642e-05,
            1.0796e-04, 9.9380e-06, 9.2489e-06,
            3.2953e-05, 5.7006e-06, 5.5769e-06,
            9.0486e-06, 3.9290e-06, 3.9320e-06,
            2.6178e-06, 3.2478e-06, 3.2460e-06,
        },
    },
    // ISO 1600
    {
        { 8.7268e-04, 7.4568e-04, 8.0404e-04, 7.9171e-04 },
        {
            6.9019e-04, 4.3119e-05, 3.8188e-05,
            2.0779e-04, 2.7140e-05, 2.4992e-05,
            5.8765e-05, 1.4258e-05, 1.3654e-05,
            1.5557e-05, 8.7038e-06, 8.6474e-06,
            6.6261e-06, 7.0836e-06, 7.1265e-06,
        },
    },
    // ISO 3200
    {
        { 2.5013e-03, 1.7729e-03, 2.2981e-03, 1.7412e-03 },
        {
            9.4670e-04, 4.4112e-05, 4.0506e-05,
            3.0640e-04, 2.6943e-05, 2.5852e-05,
            9.2139e-05, 1.3823e-05, 1.3805e-05,
            2.6873e-05, 7.9319e-06, 7.9875e-06,
            8.8264e-06, 5.9351e-06, 5.8881e-06,
        },
    }
}};

std::array<DenoiseParameters, 5> IMX492DenoiseParameters(int iso) {
    const gls::Matrix<5, 3> nlf_params = nlfFromIso<5>(NLF_IMX492, iso);

    // A reasonable denoising calibration on a fairly large range of Noise Variance values

    const float min_green_variance = 5e-05;
    const float max_green_variance = 5e-03;
    const float nlf_green_variance = std::clamp(nlf_params[0][1], min_green_variance, max_green_variance);

    const float nlf_alpha = log2(nlf_green_variance / min_green_variance) / log2(max_green_variance / min_green_variance);

    std::array<DenoiseParameters, 5> denoiseParameters = {{
        {
            .luma = std::lerp(0.25f, 2.0f, nlf_alpha),
            .chroma = 4,
            .sharpening = std::lerp(1.2f, 0.7f, nlf_alpha)
        },
        {
            .luma = std::lerp(0.25f, 2.0f, nlf_alpha),
            .chroma = 4,
            .sharpening = std::lerp(1.2f, 1.0f, nlf_alpha)
        },
        {
            .luma = std::lerp(0.25f, 3.0f, nlf_alpha),
            .chroma = 4,
            .sharpening = 1.0
        },
        {
            .luma = std::lerp(0.25f, 3.0f, nlf_alpha),
            .chroma = 2, // std::lerp(2.0f, 4.0f, nlf_alpha),
            .sharpening = 1.0
        },
        {
            .luma = std::lerp(0.25f, 3.0f, nlf_alpha),
            .chroma = 1, // std::lerp(2.0f, 4.0f, nlf_alpha),
            .sharpening = 1.0
        }
    }};

    return denoiseParameters;
}

gls::image<gls::rgb_pixel>::unique_ptr demosaicIMX492DNG(const std::filesystem::path& input_path) {
    DemosaicParameters demosaicParameters = {
        .rgbConversionParameters = {
            .contrast = 1.05,
            .saturation = 1.0,
            .toneCurveSlope = 3.5,
        }
    };

    gls::tiff_metadata dng_metadata, exif_metadata;
    // Leaky IR
    dng_metadata.insert({ TIFFTAG_COLORMATRIX1, std::vector<float>{ 1.4955, -0.6760, -0.1453, -0.1341, 1.0072, 0.1269, -0.0647, 0.1987, 0.4304 } });
    dng_metadata.insert({ TIFFTAG_ASSHOTNEUTRAL, std::vector<float>{ 1 / 1.73344, 1, 1 / 1.68018 } });

    // Gzikai
//    dng_metadata.insert({ TIFFTAG_COLORMATRIX1, std::vector<float>{ 1.6864, -0.7955, -0.3059, 0.0838, 1.0317, -0.1155, 0.0586, 0.0492, 0.3925 } });
//    dng_metadata.insert({ TIFFTAG_ASSHOTNEUTRAL, std::vector<float>{ 1 / 2.0777, 1.0000, 1 / 1.8516 } });

//    // Kolari
//    dng_metadata.insert({ TIFFTAG_COLORMATRIX1, std::vector<float>{ 0.6963, -0.2447, -0.1353, -0.3240, 1.2994, 0.0246, -0.0190, 0.1222, 0.4874 } });
//    dng_metadata.insert({ TIFFTAG_ASSHOTNEUTRAL, std::vector<float>{ 1 / 3.7730, 1.0000, 1 / 1.6040 } });

    dng_metadata.insert({ TIFFTAG_CFAREPEATPATTERNDIM, std::vector<uint16_t>{ 2, 2 } });
    dng_metadata.insert({ TIFFTAG_CFAPATTERN, std::vector<uint8_t>{ 1, 2, 0, 1 } });
    dng_metadata.insert({ TIFFTAG_BLACKLEVEL, std::vector<float>{ 0 } });
    dng_metadata.insert({ TIFFTAG_WHITELEVEL, std::vector<uint32_t>{ 0xfff } });
    dng_metadata.insert({ TIFFTAG_MAKE, "Glass Imaging" });
    dng_metadata.insert({ TIFFTAG_UNIQUECAMERAMODEL, "Glass 1" });

    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    unpackDNGMetadata(*inputImage, &dng_metadata, &demosaicParameters, /*auto_white_balance=*/ true, /*gmb_position=*/ nullptr, /*rotate_180=*/ false);

    bool iso = 400;
    const auto exifIsoSpeedRatings = getVector<uint16_t>(exif_metadata, EXIFTAG_ISOSPEEDRATINGS);
    if (exifIsoSpeedRatings.size() > 0) {
        iso = exifIsoSpeedRatings[0];
    }

    demosaicParameters.noiseModel.pyramidNlf = nlfFromIso<5>(NLF_IMX492, iso);
    demosaicParameters.denoiseParameters = IMX492DenoiseParameters(iso);

    return demosaicImage(*inputImage, &demosaicParameters, /*gmb_position=*/ nullptr, /*rotate_180=*/ true);
}

gls::image<gls::rgb_pixel>::unique_ptr calibrateIMX492DNG(const std::filesystem::path& input_path, DemosaicParameters* demosaicParameters,
                                                           int iso, const gls::rectangle& rotated_gmb_position, bool rotate_180) {
    gls::tiff_metadata dng_metadata, exif_metadata;
    dng_metadata.insert({ TIFFTAG_COLORMATRIX1, std::vector<float>{ 1, 0, 0, 0, 1, 0, 0, 0, 1 } });
    dng_metadata.insert({ TIFFTAG_ASSHOTNEUTRAL, std::vector<float>{ 1, 1, 1 } });
    dng_metadata.insert({ TIFFTAG_CFAREPEATPATTERNDIM, std::vector<uint16_t>{ 2, 2 } });
    dng_metadata.insert({ TIFFTAG_CFAPATTERN, std::vector<uint8_t>{ 1, 2, 0, 1 } });
    dng_metadata.insert({ TIFFTAG_BLACKLEVEL, std::vector<float>{ 0 } });
    dng_metadata.insert({ TIFFTAG_WHITELEVEL, std::vector<uint32_t>{ 0xfff } });
    dng_metadata.insert({ TIFFTAG_MAKE, "Glass Imaging" });
    dng_metadata.insert({ TIFFTAG_UNIQUECAMERAMODEL, "Glass 1" });

    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    const gls::rectangle gmb_position = rotate180(rotated_gmb_position, *inputImage);

    unpackDNGMetadata(*inputImage, &dng_metadata, demosaicParameters, /*auto_white_balance=*/ false, &gmb_position, rotate_180);

    // See if the ISO value is present and override
    const auto exifIsoSpeedRatings = getVector<uint16_t>(exif_metadata, EXIFTAG_ISOSPEEDRATINGS);
    if (exifIsoSpeedRatings.size() > 0) {
        iso = exifIsoSpeedRatings[0];
    }

    // demosaicParameters->noiseModel.pyramidNlf = nlfFromIso<5>(NLF_IMX492, iso);
    demosaicParameters->denoiseParameters = IMX492DenoiseParameters(iso);

    return demosaicImage(*inputImage, demosaicParameters, &rotated_gmb_position, rotate_180);
}

void calibrateIMX492(const std::filesystem::path& input_dir) {
    struct CalibrationEntry {
        int iso;
        const char* fileName;
        gls::rectangle gmb_position;
        bool rotated;
    };

    std::array<CalibrationEntry, 6> calibration_files = {{
        { 100, "calibration_100_33333_2022-04-21-17-52-58-947.dng", { 4537-10, 2370, 1652, 1068 }, true },
        { 200, "calibration_200_25000_2022-04-21-17-53-05-454.dng", { 4537-10, 2370, 1652, 1068 }, true },
        { 400, "calibration_400_8333_2022-04-21-17-53-12-201.dng", { 4537-10, 2370, 1652, 1068 }, true },
        { 800, "calibration_800_33333_2022-04-21-17-54-56-251.dng", { 4537-80, 2351, 1652, 1068 }, true },
        { 1600, "calibration_1600_25000_2022-04-21-17-55-04-540.dng", { 4537-80, 2351, 1652, 1068 }, true },
        { 3200, "calibration_3200_8333_2022-04-21-17-55-21-836.dng", { 4537-80, 2351, 1652, 1068 }, true },
    }};

    std::array<NoiseModel, 6> noiseModel;

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

        const auto rgb_image = calibrateIMX492DNG(input_path, &demosaicParameters, entry.iso, entry.gmb_position, entry.rotated);
        rgb_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_cal_rgb.png", /*skip_alpha=*/ true);

        noiseModel[i] = demosaicParameters.noiseModel;
    }

    std::cout << "Calibration table for IMX492:" << std::endl;
    for (int i = 0; i < calibration_files.size(); i++) {
        std::cout << "{ " << noiseModel[i].rawNlf << " }," << std::endl;
        std::cout << "{\n" << noiseModel[i].pyramidNlf << "\n}," << std::endl;
    }
}

static const std::array<NoiseModel, 10> LeicaQ2 = {
{
    {
        { 4.1532e-05, 5.1049e-05, 4.9132e-05, 5.1023e-05 },
        {
            4.3674e-05, 1.0379e-05, 9.6441e-06,
            1.8552e-05, 9.2747e-06, 8.9338e-06,
            1.3263e-05, 8.5899e-06, 8.4892e-06,
            1.3445e-05, 8.5151e-06, 8.4863e-06,
            1.8466e-05, 8.7083e-06, 8.7012e-06,
        },
    },
    {
        { 7.5068e-05, 8.8781e-05, 8.2993e-05, 8.7150e-05 },
        {
            7.2777e-05, 1.2411e-05, 1.0841e-05,
            2.4231e-05, 1.0241e-05, 9.4705e-06,
            1.4979e-05, 8.8609e-06, 8.5951e-06,
            1.3182e-05, 8.4775e-06, 8.4095e-06,
            1.9735e-05, 8.7217e-06, 8.6898e-06,
        },
    },
    {
        { 1.4439e-04, 1.5661e-04, 1.5666e-04, 1.5110e-04 },
        {
            1.2523e-04, 1.6498e-05, 1.3502e-05,
            3.1692e-05, 1.2006e-05, 1.0631e-05,
            1.3537e-05, 9.3939e-06, 8.9393e-06,
            1.1223e-05, 8.6556e-06, 8.5320e-06,
            1.8784e-05, 8.4549e-06, 8.4397e-06,
        },
    },
    {
        { 2.8469e-04, 3.0171e-04, 2.8955e-04, 3.1108e-04 },
        {
            2.5045e-04, 2.4108e-05, 1.7872e-05,
            5.9054e-05, 1.5406e-05, 1.2380e-05,
            2.1917e-05, 1.0282e-05, 9.2603e-06,
            1.5235e-05, 8.5958e-06, 8.3176e-06,
            2.1378e-05, 8.4406e-06, 8.3601e-06,
        },
    },
    {
        { 5.5517e-04, 5.6939e-04, 5.5710e-04, 5.6703e-04 },
        {
            4.6474e-04, 3.8187e-05, 2.7072e-05,
            9.9054e-05, 2.1815e-05, 1.6527e-05,
            2.9705e-05, 1.1910e-05, 1.0273e-05,
            1.8941e-05, 9.3757e-06, 8.9502e-06,
            2.1832e-05, 8.6840e-06, 8.5610e-06,
        },
    },
    {
        { 1.0746e-03, 1.1374e-03, 1.0807e-03, 1.0729e-03 },
        {
            6.2536e-04, 7.3135e-05, 5.0166e-05,
            1.6314e-04, 4.1916e-05, 3.0638e-05,
            5.0844e-05, 2.3085e-05, 1.9377e-05,
            2.3116e-05, 1.6441e-05, 1.5230e-05,
            1.9482e-05, 1.3091e-05, 1.2686e-05,
        },
    },
    {
        { 2.2234e-03, 2.2147e-03, 2.1940e-03, 2.1301e-03 },
        {
            1.1869e-03, 1.3923e-04, 8.9097e-05,
            2.9877e-04, 7.4865e-05, 5.0645e-05,
            8.5321e-05, 3.5109e-05, 2.7355e-05,
            3.4933e-05, 2.0874e-05, 1.8783e-05,
            2.0153e-05, 1.9063e-05, 1.8411e-05,
        },
    },
    {
        { 4.2268e-03, 4.6509e-03, 4.4503e-03, 4.5066e-03 },
        {
            2.4664e-03, 2.6431e-04, 1.5686e-04,
            6.0625e-04, 1.3315e-04, 8.1000e-05,
            1.6467e-04, 5.1311e-05, 3.4506e-05,
            5.0127e-05, 2.1918e-05, 1.7752e-05,
            1.9439e-05, 1.5526e-05, 1.4372e-05,
        },
    },
    {
        { 7.8770e-03, 8.5880e-03, 8.0832e-03, 8.6783e-03 },
        {
            4.8651e-03, 5.6487e-04, 3.0475e-04,
            1.1912e-03, 2.7469e-04, 1.5318e-04,
            3.1822e-04, 1.0242e-04, 6.1245e-05,
            1.0085e-04, 4.0556e-05, 3.0931e-05,
            3.3533e-05, 2.4757e-05, 2.3498e-05,
        },
    },
    {
        { 1.8687e-02, 1.6641e-02, 1.6274e-02, 1.6624e-02 },
        {
            1.0067e-02, 1.4355e-03, 7.4581e-04,
            2.2852e-03, 6.7995e-04, 3.6244e-04,
            5.5398e-04, 2.4559e-04, 1.4715e-04,
            1.1395e-04, 9.4160e-05, 6.9658e-05,
            4.0989e-05, 6.4033e-05, 5.9701e-05,
        },
    },
}
};

std::array<DenoiseParameters, 5> LeicaQ2DenoiseParameters(int iso) {
    const auto nlf_params = nlfFromIso<5>(LeicaQ2, iso);

    // A reasonable denoising calibration on a fairly large range of Noise Variance values

    const float min_green_variance = 5e-05;
    const float max_green_variance = 1.6e-02;
    const float nlf_green_variance = std::clamp(nlf_params.first[1], min_green_variance, max_green_variance);

    const float nlf_alpha = log2(nlf_green_variance / min_green_variance) / log2(max_green_variance / min_green_variance);

    const float iso_alpha = 2 * std::clamp(nlf_alpha - 0.5, 0.0, 0.5);

    std::array<DenoiseParameters, 5> denoiseParameters = {{
        {
            .luma = std::lerp(0.5f, 1.0f, iso_alpha),
            .chroma = std::lerp(1.0f, 4.0f, iso_alpha),
            .sharpening = std::lerp(1.2f, 1.0f, nlf_alpha)
        },
        {
            .luma = std::lerp(0.75f, 2.0f, iso_alpha),
            .chroma = std::lerp(0.75f, 2.0f, iso_alpha),
            .sharpening = 1,
        },
        {
            .luma = std::lerp(0.75f, 8.0f, iso_alpha),
            .chroma = std::lerp(0.75f, 4.0f, iso_alpha),
            .sharpening = 1
        },
        {
            .luma = std::lerp(0.25f, 4.0f, iso_alpha),
            .chroma = std::lerp(0.25f, 2.0f, iso_alpha),
            .sharpening = 1
        },
        {
            .luma = std::lerp(0.125f, 2.0f, iso_alpha),
            .chroma = std::lerp(0.125f, 2.0f, iso_alpha),
            .sharpening = 1
        }
    }};

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
        rgb_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_cal_mix_rgb.png", /*skip_alpha=*/ true);

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


void copyMetadata(const gls::tiff_metadata& source, gls::tiff_metadata* destination, ttag_t tag) {
    const auto entry = source.find(tag);
    if (entry != source.end()) {
        destination->insert({ tag, entry->second });
    }
}

void saveStrippedDNG(const std::string& file_name, const gls::image<gls::luma_pixel_16>& inputImage, const gls::tiff_metadata& dng_metadata, const gls::tiff_metadata& exif_metadata) {
    gls::tiff_metadata my_exif_metadata;
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_FNUMBER);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_EXPOSUREPROGRAM);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_EXPOSURETIME);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_ISOSPEEDRATINGS);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_DATETIMEORIGINAL);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_DATETIMEDIGITIZED);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_OFFSETTIME);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_OFFSETTIMEORIGINAL);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_OFFSETTIMEDIGITIZED);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_COMPONENTSCONFIGURATION);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_SHUTTERSPEEDVALUE);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_APERTUREVALUE);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_EXPOSUREBIASVALUE);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_MAXAPERTUREVALUE);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_METERINGMODE);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_LIGHTSOURCE);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_FLASH);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_FOCALLENGTH);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_COLORSPACE);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_SENSINGMETHOD);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_WHITEBALANCE);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_BODYSERIALNUMBER);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_LENSSPECIFICATION);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_LENSMAKE);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_LENSMODEL);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_LENSSERIALNUMBER);

    // Write out a stripped DNG files with minimal metadata
    inputImage.write_dng_file(file_name, /*compression=*/ gls::JPEG, &dng_metadata, &my_exif_metadata);
}

void transcodeAdobeDNG(const std::filesystem::path& input_path) {
    gls::tiff_metadata dng_metadata, exif_metadata;
    dng_metadata.insert({ TIFFTAG_COLORMATRIX1, std::vector<float>{ 1.4955, -0.6760, -0.1453, -0.1341, 1.0072, 0.1269, -0.0647, 0.1987, 0.4304 } });
    dng_metadata.insert({ TIFFTAG_ASSHOTNEUTRAL, std::vector<float>{ 1 / 1.73344, 1, 1 / 1.68018 } });
//    dng_metadata.insert({ TIFFTAG_CFAREPEATPATTERNDIM, std::vector<uint16_t>{ 2, 2 } });
//    dng_metadata.insert({ TIFFTAG_CFAPATTERN, std::vector<uint8_t>{ 2, 1, 1, 0 } });
//    dng_metadata.insert({ TIFFTAG_BLACKLEVEL, std::vector<float>{ 0 } });
//    // dng_metadata.insert({ TIFFTAG_WHITELEVEL, std::vector<uint32_t>{ 0x3fff } });
    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    auto output_file = (input_path.parent_path() / input_path.stem()).string() + "_fixed.dng";
    saveStrippedDNG(output_file, *inputImage, dng_metadata, exif_metadata);
}

gls::image<gls::rgb_pixel>::unique_ptr demosaicAdobeDNG(const std::filesystem::path& input_path) {
    DemosaicParameters demosaicParameters = {
        .rgbConversionParameters = {
            .contrast = 1.05,
            .saturation = 1.0,
            .toneCurveSlope = 3.5,
        }
    };

    bool rotate_180 = false;
    // const gls::rectangle gmb_position = { 3425, 770, 1554, 1019 }; // Leica Q2 DPReview
    // const gls::rectangle gmb_position = { 2434, 506, 1124, 729 }; // Ricoh GR III DPReview

    gls::tiff_metadata dng_metadata, exif_metadata;
    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    unpackDNGMetadata(*inputImage, &dng_metadata, &demosaicParameters, /*auto_white_balance=*/ false, nullptr /* &gmb_position */, rotate_180);

    const auto iso = getVector<uint16_t>(exif_metadata, EXIFTAG_ISOSPEEDRATINGS)[0];

    // demosaicParameters->noiseModel.pyramidNlf = nlfFromIso<5>(NLF_IMX492, iso);
    demosaicParameters.denoiseParameters = IMX492DenoiseParameters(iso);

//    auto output_file = (input_path.parent_path() / input_path.stem()).string() + "_my.dng";
//    saveStrippedDNG(output_file, *inputImage, dng_metadata, exif_metadata);

    return demosaicImage(*inputImage, &demosaicParameters, nullptr /* &gmb_position */, rotate_180);
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

    const float min_green_variance = 5e-05;
    const float max_green_variance = 1.6e-02;
    const float nlf_green_variance = std::clamp(nlf_params.first[1], min_green_variance, max_green_variance);

    const float nlf_alpha = log2(nlf_green_variance / min_green_variance) / log2(max_green_variance / min_green_variance);

    const float iso_alpha = 2 * std::clamp(nlf_alpha - 0.5, 0.0, 0.5);

    std::array<DenoiseParameters, 5> denoiseParameters = {{
        {
            .luma = std::lerp(0.5f, 2.0f, iso_alpha),
            .chroma = std::lerp(1.0f, 16.0f, iso_alpha),
            .sharpening = std::lerp(2.0f, 0.7f, nlf_alpha)
        },
        {
            .luma = std::lerp(0.75f, 4.0f, iso_alpha),
            .chroma = std::lerp(0.75f, 8.0f, iso_alpha),
            .sharpening = std::lerp(1.0f, 0.8f, nlf_alpha)
        },
        {
            .luma = std::lerp(0.75f, 2.0f, iso_alpha),
            .chroma = std::lerp(0.75f, 8.0f, iso_alpha),
            .sharpening = 1
        },
        {
            .luma = std::lerp(0.25f, 1.0f, iso_alpha),
            .chroma = std::lerp(0.25f, 8.0f, iso_alpha),
            .sharpening = 1
        },
        {
            .luma = std::lerp(0.125f, 0.5f, iso_alpha),
            .chroma = std::lerp(0.125f, 8.0f, iso_alpha),
            .sharpening = 1
        }
    }};

//    std::array<DenoiseParameters, 5> denoiseParameters = {{
//        {
//            .luma = 0.25,
//            .chroma = 64,
//            .sharpening = 1
//        },
//        {
//            .luma = 0.75f,
//            .chroma = 32 * 0.75f,
//            .sharpening = 1
//        },
//        {
//            .luma = 0.5f,
//            .chroma = 16 * 0.5f,
//            .sharpening = 1
//        },
//        {
//            .luma = 0.25f,
//            .chroma = 8 * 0.25f,
//            .sharpening = 1
//        },
//        {
//            .luma = 0.125f,
//            .chroma = 4 * 0.125f,
//            .sharpening = 1
//        }
//    }};

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

gls::image<gls::rgb_pixel>::unique_ptr demosaicIMX492V2DNG(const std::filesystem::path& input_path) {
    DemosaicParameters demosaicParameters = {
        .rgbConversionParameters = {
            .contrast = 1.05,
            .saturation = 1.0,
            .toneCurveSlope = 3.5,
        }
    };

    gls::tiff_metadata dng_metadata, exif_metadata;
    dng_metadata.insert({ TIFFTAG_COLORMATRIX1, std::vector<float>{ 0.2468, -0.2485, 0.9018, -0.9060, 1.9690, -0.0630, 3.4279, -1.6351, -0.5652 } });
    dng_metadata.insert({ TIFFTAG_ASSHOTNEUTRAL, std::vector<float>{ 1 / 1.0737, 1.0000, 1 / 1.0313 } });
    dng_metadata.insert({ TIFFTAG_CFAREPEATPATTERNDIM, std::vector<uint16_t>{ 2, 2 } });
    dng_metadata.insert({ TIFFTAG_CFAPATTERN, std::vector<uint8_t>{ 2, 1, 1, 0 } });

    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata);

    unpackDNGMetadata(*inputImage, &dng_metadata, &demosaicParameters, /*auto_white_balance=*/ false, nullptr, false);

    return demosaicImage(*inputImage, &demosaicParameters, nullptr, false);
}

int main(int argc, const char* argv[]) {
    printf("RawPipeline Test!\n");

    if (argc > 1) {
//        auto input_dir = std::filesystem::path(argv[1]);

//        std::vector<std::filesystem::path> directory_listing;
//        std::copy(std::filesystem::directory_iterator(input_dir), std::filesystem::directory_iterator(),
//                  std::back_inserter(directory_listing));
//        std::sort(directory_listing.begin(), directory_listing.end());
//
//        for (const auto& input_path : directory_listing) {
//            if (input_path.extension() != ".dng" || input_path.filename().string().starts_with(".")) {
//                continue;
//            }
//
//            LOG_INFO(TAG) << "Processing: " << input_path.filename() << std::endl;
//
//            // transcodeAdobeDNG(input_path);
//            const auto rgb_image = demosaicIMX492DNG(input_path);
//            rgb_image->write_jpeg_file((input_path.parent_path() / input_path.stem()).string() + "_rgb.jpg", 95);
//        }

        auto input_path = std::filesystem::path(argv[1]);
//
//        LOG_INFO(TAG) << "Calibrating IMX492 sensor from data in: " << input_path.filename() << std::endl;
//
//        DemosaicParameters demosaicParameters = {
//            .rgbConversionParameters = {
//                .contrast = 1.05,
//                .saturation = 1.0,
//                .toneCurveSlope = 3.5,
//            }
//        };
//
//        bool rotate_180 = true;
//        // const gls::rectangle& rotated_gmb_position = { 3198, 2237, 1857, 1209 }; // 2022-05-03-14-28-19-729.coords.txt
//        // const gls::rectangle& rotated_gmb_position = { 2767, 1821, 2887, 1909 }; // 2022-05-03-14-33-36-986.coords.txt
//
//        // const gls::rectangle& rotated_gmb_position = { 2835, 2709, 2100, 1352 }; // 2022-05-03-17-13-18-618.coords.txt
//        // const gls::rectangle& rotated_gmb_position = { 3124, 2150, 2113, 1363 }; // 2022-05-03-17-14-20-048.coords.txt
//
//        // const gls::rectangle& rotated_gmb_position = { 3606, 2361, 2538, 1907 }; // 2022-05-04-15-38-08-961.coords.txt
//        const gls::rectangle& rotated_gmb_position = { 3863, 2427, 2401, 1742 }; // 2022-05-04-15-38-08-961.coords.txt
//
//        const auto rgb_image = calibrateIMX492DNG(input_path, &demosaicParameters, /*iso=*/ 100, rotated_gmb_position, rotate_180);

        gls::OpenCLContext glsContext("");
        RawConverter rawConverter(&glsContext);

        // calibrateIMX492(input_path.parent_path());
        calibrateLeicaQ2(&rawConverter, input_path.parent_path());

        // calibrateiPhone11(&rawConverter, input_path.parent_path());

        LOG_INFO(TAG) << "Processing: " << input_path.filename() << std::endl;

        const auto rgb_image = demosaiciPhone11(&rawConverter, input_path);
        rgb_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_rgb.png", /*skip_alpha=*/ true);
    }
}
