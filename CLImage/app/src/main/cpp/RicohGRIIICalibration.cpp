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
    // ISO 100
    {
        { 4.8e-05, 7.0e-05, 5.9e-05, 6.6e-05 },
        {
            8.7e-06, 3.0e-07, 2.8e-07, 9.1e-05, 6.5e-06, 1.2e-05,
            1.0e-05, 3.3e-07, 2.9e-07, 2.1e-04, 5.0e-06, 5.9e-06,
            2.1e-06, 3.9e-07, 3.7e-07, 9.8e-04, 7.5e-06, 5.5e-06,
            2.0e-05, 6.9e-07, 6.0e-07, 1.7e-03, 4.6e-06, 3.9e-06,
            2.1e-05, 6.8e-07, 9.7e-07, 4.6e-03, 1.9e-05, 1.3e-05,
        },
    },
    // ISO 200
    {
        { 8.1e-05, 9.7e-05, 8.9e-05, 1.0e-04 },
        {
            8.7e-06, 3.1e-07, 3.0e-07, 1.1e-04, 1.0e-05, 2.0e-05,
            1.0e-05, 3.3e-07, 2.9e-07, 2.2e-04, 6.4e-06, 9.4e-06,
            2.0e-06, 3.8e-07, 3.6e-07, 9.8e-04, 7.9e-06, 6.6e-06,
            2.0e-05, 6.9e-07, 6.0e-07, 1.8e-03, 4.8e-06, 4.3e-06,
            2.1e-05, 6.8e-07, 9.6e-07, 4.6e-03, 1.9e-05, 1.3e-05,
        },
    },
    // ISO 400
    {
        { 7.2e-05, 1.1e-04, 8.5e-05, 1.2e-04 },
        {
            8.8e-06, 3.8e-07, 3.5e-07, 1.2e-04, 8.5e-06, 1.8e-05,
            1.0e-05, 3.4e-07, 2.9e-07, 2.2e-04, 6.8e-06, 1.1e-05,
            2.1e-06, 3.8e-07, 3.5e-07, 9.8e-04, 8.7e-06, 8.3e-06,
            2.0e-05, 6.8e-07, 5.9e-07, 1.8e-03, 5.0e-06, 4.9e-06,
            2.0e-05, 6.8e-07, 9.7e-07, 4.6e-03, 1.9e-05, 1.4e-05,
        },
    },
    // ISO 800
    {
        { 8.9e-05, 1.9e-04, 1.2e-04, 1.8e-04 },
        {
            8.8e-06, 4.1e-07, 3.9e-07, 1.6e-04, 9.2e-06, 2.1e-05,
            1.0e-05, 3.5e-07, 3.1e-07, 2.3e-04, 8.1e-06, 1.5e-05,
            2.2e-06, 3.8e-07, 3.6e-07, 9.8e-04, 9.9e-06, 1.1e-05,
            2.0e-05, 6.8e-07, 5.9e-07, 1.8e-03, 5.5e-06, 5.8e-06,
            2.0e-05, 6.7e-07, 9.6e-07, 4.6e-03, 2.0e-05, 1.4e-05,
        },
    },
    // ISO 1600
    {
        { 1.8e-04, 3.6e-04, 2.0e-04, 4.0e-04 },
        {
            8.9e-06, 4.7e-07, 3.7e-07, 2.5e-04, 1.3e-05, 3.1e-05,
            1.1e-05, 3.7e-07, 3.1e-07, 2.3e-04, 8.8e-06, 1.7e-05,
            2.7e-06, 3.8e-07, 3.6e-07, 9.2e-04, 1.1e-05, 1.3e-05,
            2.0e-05, 7.0e-07, 6.3e-07, 1.6e-03, 6.0e-06, 7.0e-06,
            2.5e-05, 7.3e-07, 1.1e-06, 4.4e-03, 2.0e-05, 1.4e-05,
        },
    },
    // ISO 3200
    {
        { 2.8e-04, 6.5e-04, 3.7e-04, 6.5e-04 },
        {
            8.6e-06, 5.2e-07, 4.2e-07, 2.5e-04, 1.8e-05, 4.2e-05,
            1.0e-05, 4.0e-07, 3.4e-07, 2.5e-04, 1.4e-05, 2.8e-05,
            2.6e-06, 3.8e-07, 3.6e-07, 9.5e-04, 1.4e-05, 2.2e-05,
            2.1e-05, 7.0e-07, 6.4e-07, 1.7e-03, 7.9e-06, 1.2e-05,
            2.2e-05, 7.1e-07, 1.0e-06, 4.5e-03, 2.0e-05, 1.6e-05,
        },
    },
    // ISO 6400
    {
        { 6.2e-04, 1.4e-03, 7.9e-04, 1.4e-03 },
        {
            9.8e-06, 7.8e-07, 4.8e-07, 4.9e-04, 3.9e-05, 9.0e-05,
            1.1e-05, 5.5e-07, 4.9e-07, 2.9e-04, 2.1e-05, 5.0e-05,
            3.1e-06, 4.3e-07, 4.7e-07, 9.2e-04, 2.0e-05, 3.4e-05,
            2.0e-05, 7.7e-07, 8.0e-07, 1.6e-03, 1.0e-05, 1.8e-05,
            2.8e-05, 7.9e-07, 1.1e-06, 4.2e-03, 2.1e-05, 1.9e-05,
        },
    },
}};

template <int levels>
static std::pair<gls::Vector<4>, gls::Matrix<levels, 6>> nlfFromIso(const std::array<NoiseModel, 7>& NLFData, int iso) {
    iso = std::clamp(iso, 100, 6400);
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

std::pair<float, std::array<DenoiseParameters, 5>>  RicohGRIIIDenoiseParameters(int iso) {
    const auto nlf_params = nlfFromIso<5>(RicohGRIII, iso);

    // A reasonable denoising calibration on a fairly large range of Noise Variance values
    const float min_green_variance = RicohGRIII[0].rawNlf[1];
    const float max_green_variance = RicohGRIII[RicohGRIII.size()-1].rawNlf[1];
    const float nlf_green_variance = std::clamp(nlf_params.first[1], min_green_variance, max_green_variance);
    const float nlf_alpha = log2(nlf_green_variance / min_green_variance) / log2(max_green_variance / min_green_variance);

    std::cout << "RicohGRIIIDenoiseParameters nlf_alpha: " << nlf_alpha << std::endl;

    // Bilateral RicohGRIII
    std::array<DenoiseParameters, 5> denoiseParameters = {{
        {
            .luma = 0.125f * std::lerp(1.0f, 2.0f, nlf_alpha),
            .chroma = std::lerp(1.0f, 8.0f, nlf_alpha),
            .sharpening = std::lerp(1.5f, 1.0f, nlf_alpha)
        },
        {
            .luma = 0.25f * std::lerp(1.0f, 2.0f, nlf_alpha),
            .chroma = std::lerp(1.0f, 8.0f, nlf_alpha),
            .sharpening = 1.1 // std::lerp(1.0f, 0.8f, nlf_alpha),
        },
        {
            .luma = 0.5f * std::lerp(1.0f, 2.0f, nlf_alpha),
            .chroma = std::lerp(1.0f, 4.0f, nlf_alpha),
            .sharpening = 1
        },
        {
            .luma = 0.25f * std::lerp(1.0f, 2.0f, nlf_alpha),
            .chroma = std::lerp(1.0f, 4.0f, nlf_alpha),
            .sharpening = 1
        },
        {
            .luma = 0.125f * std::lerp(1.0f, 2.0f, nlf_alpha),
            .chroma = std::lerp(1.0f, 8.0f, nlf_alpha),
            .sharpening = 1
        }
    }};

    return { nlf_alpha, denoiseParameters };
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

    const auto denoiseParameters = RicohGRIIIDenoiseParameters(iso);
    demosaicParameters->noiseLevel = denoiseParameters.first;
    demosaicParameters->denoiseParameters = denoiseParameters.second;

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
        { 100,  "R0000914_ISO100.DNG",  { 2437, 506, 1123, 733 }, false },
        { 200,  "R0000917_ISO200.DNG",  { 2437, 506, 1123, 733 }, false },
        { 400,  "R0000920_ISO400.DNG",  { 2437, 506, 1123, 733 }, false },
        { 800,  "R0000923_ISO800.DNG",  { 2437, 506, 1123, 733 }, false },
        { 1600, "R0000926_ISO1600.DNG", { 2437, 506, 1123, 733 }, false },
        { 3200, "R0000929_ISO3200.DNG", { 2437, 506, 1123, 733 }, false },
        { 6400, "R0000932_ISO6400.DNG", { 2437, 506, 1123, 733 }, false },
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
        rgb_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_cal.png", /*skip_alpha=*/ true);

        noiseModel[i] = demosaicParameters.noiseModel;
    }

    std::cout << "Calibration table for RicohGRIII:" << std::endl;
    for (int i = 0; i < calibration_files.size(); i++) {
        std::cout << "// ISO " << calibration_files[i].iso << std::endl;
        std::cout << "{" << std::endl;
        std::cout << "{ " << noiseModel[i].rawNlf << " }," << std::endl;
        std::cout << "{\n" << noiseModel[i].pyramidNlf << "\n}," << std::endl;
        std::cout << "}," << std::endl;
    }
}

gls::image<gls::rgb_pixel>::unique_ptr demosaicRicohGRIII2DNG(RawConverter* rawConverter, const std::filesystem::path& input_path) {
    DemosaicParameters demosaicParameters = {
        .rgbConversionParameters = {
            .localToneMapping = true
        }
    };

    gls::tiff_metadata dng_metadata, exif_metadata;
    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    unpackDNGMetadata(*inputImage, &dng_metadata, &demosaicParameters, /*auto_white_balance=*/ false, nullptr /* &gmb_position */, /*rotate_180=*/ false);

    const auto iso = getVector<uint16_t>(exif_metadata, EXIFTAG_ISOSPEEDRATINGS)[0];

    std::cout << "EXIF ISO: " << iso << std::endl;

    const auto nlfParams = nlfFromIso<5>(RicohGRIII, iso);
    const auto denoiseParameters = RicohGRIIIDenoiseParameters(iso);
    demosaicParameters.noiseModel.rawNlf = nlfParams.first;
    demosaicParameters.noiseModel.pyramidNlf = nlfParams.second;
    demosaicParameters.noiseLevel = denoiseParameters.first;
    demosaicParameters.denoiseParameters = denoiseParameters.second;

    return RawConverter::convertToRGBImage(*rawConverter->demosaicImage(*inputImage, &demosaicParameters, nullptr /* &gmb_position */, /*rotate_180=*/ false));
}
