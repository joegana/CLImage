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

#include "demosaic.hpp"
#include "gls_tiff_metadata.hpp"

#include "gls_linalg.hpp"

static const char* TAG = "RawPipeline Test";

static const float NLF_IMX492V2[6][5][3] = {
    // ISO 100
    {
        { 1.1817e-04, 7.6074e-05, 7.2803e-05 },
        { 5.8687e-05, 5.8594e-05, 5.7425e-05 },
        { 4.2721e-05, 4.8179e-05, 4.7825e-05 },
        { 4.1860e-05, 4.3691e-05, 4.3555e-05 },
        { 3.9072e-05, 4.2436e-05, 4.2401e-05 }
    },
    // ISO 200
    {
        { 1.9630e-04, 1.0408e-04, 9.8109e-05 },
        { 7.9714e-05, 7.0176e-05, 6.7911e-05 },
        { 4.8079e-05, 4.9691e-05, 4.9017e-05 },
        { 4.2638e-05, 4.1427e-05, 4.1235e-05 },
        { 3.8430e-05, 3.8918e-05, 3.8855e-05 }
    },
    // ISO 400
    {
        { 3.3663e-04, 1.6478e-04, 1.5480e-04 },
        { 1.1355e-04, 9.9108e-05, 9.5292e-05 },
        { 5.7377e-05, 5.9418e-05, 5.8118e-05 },
        { 4.5197e-05, 4.3951e-05, 4.3608e-05 },
        { 4.0192e-05, 3.9776e-05, 3.9616e-05 }
    },
    // ISO 800
    {
        { 5.7325e-04, 6.4165e-04, 6.1982e-04 },
        { 1.7906e-04, 5.0318e-04, 4.9554e-04 },
        { 7.7888e-05, 4.1861e-04, 4.1637e-04 },
        { 5.7175e-05, 3.8530e-04, 3.8466e-04 },
        { 4.9265e-05, 3.7448e-04, 3.7418e-04 }
    },
    // ISO 1600
    {
        { 1.0741e-03, 8.7726e-04, 8.4156e-04 },
        { 2.8819e-04, 5.9887e-04, 5.8824e-04 },
        { 1.0201e-04, 4.3271e-04, 4.3060e-04 },
        { 6.2075e-05, 3.7145e-04, 3.7136e-04 },
        { 5.0261e-05, 3.5170e-04, 3.5205e-04 }
    },
    // ISO 3200
    {
        { 2.1776e-03, 1.4413e-03, 1.3562e-03 },
        { 5.5808e-04, 8.5935e-04, 8.3257e-04 },
        { 1.8079e-04, 5.1135e-04, 5.0549e-04 },
        { 8.7985e-05, 3.7854e-04, 3.7755e-04 },
        { 5.6157e-05, 3.4096e-04, 3.4040e-04 }
    }
};

static const float NLF_IMX492[6][5][3] = {
    // ISO 100
    {
        { 4.9688e-05, 7.3893e-06, 7.2321e-06 },
        { 1.7964e-05, 6.7149e-06, 6.6420e-06 },
        { 9.6783e-06, 6.1881e-06, 6.1788e-06 },
        { 6.0442e-06, 5.9603e-06, 5.9610e-06 },
        { 4.8379e-06, 5.9137e-06, 5.9116e-06 }
    },
    // ISO 200
    {
        { 9.3366e-05, 1.7681e-05, 1.7271e-05 },
        { 3.2632e-05, 1.5746e-05, 1.5601e-05 },
        { 1.5216e-05, 1.4169e-05, 1.4172e-05 },
        { 8.6612e-06, 1.3534e-05, 1.3546e-05 },
        { 6.5333e-06, 1.3421e-05, 1.3422e-05 }
    },
    // ISO 400
    {
        { 1.7849e-04, 1.3025e-05, 1.2329e-05 },
        { 5.8104e-05, 1.0410e-05, 1.0084e-05 },
        { 2.2722e-05, 8.2682e-06, 8.2175e-06 },
        { 1.0522e-05, 7.4140e-06, 7.4126e-06 },
        { 6.8284e-06, 7.2507e-06, 7.2523e-06 }
    },
    // ISO 800
    {
        { 3.4231e-04, 1.4014e-05, 1.2784e-05 },
        { 1.0282e-04, 9.0075e-06, 8.5997e-06 },
        { 3.2621e-05, 5.0612e-06, 5.0776e-06 },
        { 1.1993e-05, 3.4746e-06, 3.5347e-06 },
        { 9.5276e-06, 3.0545e-06, 3.0695e-06 }
    },
    // ISO 1600
    {
        { 6.9735e-04, 4.0897e-05, 3.7083e-05 },
        { 2.1621e-04, 2.5779e-05, 2.4293e-05 },
        { 6.9542e-05, 1.3638e-05, 1.3477e-05 },
        { 2.4663e-05, 8.5841e-06, 8.5838e-06 },
        { 1.6873e-05, 7.2737e-06, 7.3092e-06 }
    },
    // ISO 3200
    {
        { 1.0793e-03, 4.7093e-05, 4.2599e-05 },
        { 4.1953e-04, 2.9475e-05, 2.7927e-05 },
        { 1.3875e-04, 1.5666e-05, 1.5445e-05 },
        { 4.0298e-05, 9.8978e-06, 9.8024e-06 },
        { 1.5069e-05, 8.1720e-06, 8.0665e-06 }
    },
};

std::array<DenoiseParameters, 5> IMX492DenoiseParameters(int iso) {
    const gls::Matrix<5, 3> nlf_params = nlfFromIso(NLF_IMX492, iso);

    const float min_green_variance = 5e-05;
    const float max_green_variance = 5e-03;
    const float nlf_green_variance = std::clamp(nlf_params[0][1], min_green_variance, max_green_variance);

    const float nlf_alpha = log2(nlf_green_variance / min_green_variance) / log2(max_green_variance / min_green_variance);

    std::array<DenoiseParameters, 5> denoiseParameters = {{
        {
            .lumaSigma = std::lerp(0.25f, 0.75f, nlf_alpha),
            .cbSigma = 4,
            .crSigma = 4,
            .sharpening = std::lerp(1.2f, 0.7f, nlf_alpha)
        },
        {
            .lumaSigma = std::lerp(0.25f, 1.5f, nlf_alpha),
            .cbSigma = 4,
            .crSigma = 4,
            .sharpening = std::lerp(1.2f, 1.0f, nlf_alpha)
        },
        {
            .lumaSigma = std::lerp(0.25f, 3.0f, nlf_alpha),
            .cbSigma = 4,
            .crSigma = 4,
            .sharpening = 1.0
        },
        {
            .lumaSigma = std::lerp(0.25f, 3.0f, nlf_alpha),
            .cbSigma = std::lerp(2.0f, 4.0f, nlf_alpha),
            .crSigma = std::lerp(2.0f, 4.0f, nlf_alpha),
            .sharpening = 1.0
        },
        {
            .lumaSigma = std::lerp(0.25f, 3.0f, nlf_alpha),
            .cbSigma = std::lerp(2.0f, 4.0f, nlf_alpha),
            .crSigma = std::lerp(2.0f, 4.0f, nlf_alpha),
            .sharpening = 1.0
        }
    }};

//    for (int i = 0; i < 5; i++) {
//        denoiseParameters[i].lumaSigma *= sqrt(nlf_params[i][0]);
//        denoiseParameters[i].cbSigma *= sqrt(nlf_params[i][1]);
//        denoiseParameters[i].crSigma *= sqrt(nlf_params[i][2]);
//    }

    return denoiseParameters;
}

gls::image<gls::rgba_pixel>::unique_ptr demosaicIMX492DNG(const std::filesystem::path& input_path) {
    DemosaicParameters demosaicParameters = {
        .rgbConversionParameters.contrast = 1.05,
        .rgbConversionParameters.saturation = 1.0,
        .rgbConversionParameters.toneCurveSlope = 3.5,
    };

    gls::tiff_metadata dng_metadata, exif_metadata;
    dng_metadata.insert({ TIFFTAG_COLORMATRIX1, std::vector<float>{ 0.9998, -0.3819, -0.0747, 0.0013, 0.8728, 0.1259, 0.0472, 0.1029, 0.3574 } });
    dng_metadata.insert({ TIFFTAG_ASSHOTNEUTRAL, std::vector<float>{ 1 / 2.0756, 1.0000, 1 / 1.8832 } });
    dng_metadata.insert({ TIFFTAG_CFAREPEATPATTERNDIM, std::vector<uint16_t>{ 2, 2 } });
    dng_metadata.insert({ TIFFTAG_CFAPATTERN, std::vector<uint8_t>{ 1, 2, 0, 1 } });
    dng_metadata.insert({ TIFFTAG_BLACKLEVEL, std::vector<float>{ 0 } });
    dng_metadata.insert({ TIFFTAG_WHITELEVEL, std::vector<uint32_t>{ 0xfff } });
    dng_metadata.insert({ TIFFTAG_MAKE, "Glass Imaging" });
    dng_metadata.insert({ TIFFTAG_UNIQUECAMERAMODEL, "Glass 1" });

    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata);

    unpackDNGMetadata(*inputImage, &dng_metadata, &demosaicParameters, /*auto_white_balance=*/ false, /*gmb_position=*/ nullptr, /*rotate_180=*/ false);

    bool iso = 400;
    const auto exifIsoSpeedRatings = getVector<uint16_t>(exif_metadata, EXIFTAG_ISOSPEEDRATINGS);
    if (exifIsoSpeedRatings.size() > 0) {
        iso = exifIsoSpeedRatings[0];
    }

    demosaicParameters.pyramidNlfParameters = nlfFromIso(NLF_IMX492, iso);
    demosaicParameters.pyramidDenoiseParameters = IMX492DenoiseParameters(iso);

    return demosaicImage(*inputImage, &dng_metadata, &demosaicParameters, iso, /*auto_white_balance=*/ false, /*gmb_position=*/ nullptr, /*rotate_180=*/ true);
}

gls::image<gls::rgba_pixel>::unique_ptr calibrateIMX492DNG(const std::filesystem::path& input_path, DemosaicParameters* demosaicParameters,
                                                           int iso, const gls::rectangle& rotated_gmb_position, bool rotate_180) {
    gls::tiff_metadata dng_metadata;
    dng_metadata.insert({ TIFFTAG_COLORMATRIX1, std::vector<float>{ 0.9998, -0.3819, -0.0747, 0.0013, 0.8728, 0.1259, 0.0472, 0.1029, 0.3574 } });
    dng_metadata.insert({ TIFFTAG_ASSHOTNEUTRAL, std::vector<float>{ 1 / 2.0756, 1.0000, 1 / 1.8832 } });
    dng_metadata.insert({ TIFFTAG_CFAREPEATPATTERNDIM, std::vector<uint16_t>{ 2, 2 } });
    dng_metadata.insert({ TIFFTAG_CFAPATTERN, std::vector<uint8_t>{ 1, 2, 0, 1 } });
    dng_metadata.insert({ TIFFTAG_BLACKLEVEL, std::vector<float>{ 0 } });
    dng_metadata.insert({ TIFFTAG_WHITELEVEL, std::vector<uint32_t>{ 0xfff } });
    dng_metadata.insert({ TIFFTAG_MAKE, "Glass Imaging" });
    dng_metadata.insert({ TIFFTAG_UNIQUECAMERAMODEL, "Glass 1" });

    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata);

    const gls::rectangle gmb_position = rotate180(rotated_gmb_position, *inputImage);

    unpackDNGMetadata(*inputImage, &dng_metadata, demosaicParameters, /*auto_white_balance=*/ false, &gmb_position, rotate_180);

    demosaicParameters->pyramidNlfParameters = nlfFromIso(NLF_IMX492, iso);
    demosaicParameters->pyramidDenoiseParameters = IMX492DenoiseParameters(iso);

    return demosaicImage(*inputImage, &dng_metadata, demosaicParameters, /*iso=*/ iso, /*auto_white_balance=*/ false, &rotated_gmb_position, rotate_180);
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

    for (auto& entry : calibration_files) {
        const auto input_path = input_dir / entry.fileName;

        DemosaicParameters demosaicParameters = {
            .rgbConversionParameters.contrast = 1.05,
            .rgbConversionParameters.saturation = 1.0,
            .rgbConversionParameters.toneCurveSlope = 3.5,
        };

        const auto rgb_image = calibrateIMX492DNG(input_path, &demosaicParameters, entry.iso, entry.gmb_position, entry.rotated);
        rgb_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_cal_rgb.png", /*skip_alpha=*/ true);


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

gls::image<gls::rgba_pixel>::unique_ptr demosaicAdobeDNG(const std::filesystem::path& input_path) {
    DemosaicParameters demosaicParameters = {
        .rgbConversionParameters.contrast = 1.05,
        .rgbConversionParameters.saturation = 1.0,
        .rgbConversionParameters.toneCurveSlope = 3.5,
    };

    bool rotate_180 = false;
    const gls::rectangle gmb_position = { 3425, 770, 1554, 1019 }; // Leica Q2 DPReview
    // const gls::rectangle gmb_position = { 2434, 506, 1124, 729 }; // Ricoh GR III DPReview

    gls::tiff_metadata dng_metadata, exif_metadata;
    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    unpackDNGMetadata(*inputImage, &dng_metadata, &demosaicParameters, /*auto_white_balance=*/ false, &gmb_position, rotate_180);

    const auto iso = getVector<uint16_t>(exif_metadata, EXIFTAG_ISOSPEEDRATINGS)[0];

//    auto output_file = (input_path.parent_path() / input_path.stem()).string() + "_my.dng";
//    saveStrippedDNG(output_file, *inputImage, dng_metadata, exif_metadata);

    return demosaicImage(*inputImage, &dng_metadata, &demosaicParameters, /*iso=*/ iso, /*auto_white_balance=*/ false, &gmb_position, rotate_180);
}

gls::image<gls::rgba_pixel>::unique_ptr demosaicIMX492V2DNG(const std::filesystem::path& input_path) {
    DemosaicParameters demosaicParameters = {
        .rgbConversionParameters.contrast = 1.05,
        .rgbConversionParameters.saturation = 1.0,
        .rgbConversionParameters.toneCurveSlope = 3.5,
    };

    gls::tiff_metadata dng_metadata, exif_metadata;
    dng_metadata.insert({ TIFFTAG_COLORMATRIX1, std::vector<float>{ 0.2468, -0.2485, 0.9018, -0.9060, 1.9690, -0.0630, 3.4279, -1.6351, -0.5652 } });
    dng_metadata.insert({ TIFFTAG_ASSHOTNEUTRAL, std::vector<float>{ 1 / 1.0737, 1.0000, 1 / 1.0313 } });
    dng_metadata.insert({ TIFFTAG_CFAREPEATPATTERNDIM, std::vector<uint16_t>{ 2, 2 } });
    dng_metadata.insert({ TIFFTAG_CFAPATTERN, std::vector<uint8_t>{ 2, 1, 1, 0 } });

    const auto iso = 100; // getVector<uint16_t>(exif_metadata, EXIFTAG_ISOSPEEDRATINGS)[0];

    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata);

    unpackDNGMetadata(*inputImage, &dng_metadata, &demosaicParameters, /*auto_white_balance=*/ false, nullptr, false);

    return demosaicImage(*inputImage, &dng_metadata, &demosaicParameters, /*iso=*/ iso, /*auto_white_balance=*/ false, nullptr, false);
}

int main(int argc, const char* argv[]) {
    printf("RawPipeline Test!\n");

    if (argc > 1) {
//        auto input_dir = std::filesystem::path(argv[1]);
//
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
//            const auto rgb_image = demosaicAdobeDNG(input_path);
//            rgb_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_rgb.png", /*skip_alpha=*/ true);
//        }

        auto input_path = std::filesystem::path(argv[1]);

        LOG_INFO(TAG) << "Calibrating IMX492 sensor from data in: " << input_path.filename() << std::endl;

        calibrateIMX492(input_path.parent_path());

        LOG_INFO(TAG) << "Processing: " << input_path.filename() << std::endl;

        const auto rgb_image = demosaicIMX492DNG(input_path);
        rgb_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_rgb.png", /*skip_alpha=*/ true);
    }
}
