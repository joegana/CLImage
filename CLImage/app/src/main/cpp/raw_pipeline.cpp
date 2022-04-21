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

gls::image<gls::rgba_pixel>::unique_ptr demosaicIMX492V2DNG(const std::filesystem::path& input_path) {
    const DemosaicParameters demosaicParameters = {
        .contrast = 1.05,
        .saturation = 1.0,
        .toneCurveSlope = 3.5,
        .sharpening = 1.25,
        .sharpeningRadius = 7,
    };

    gls::tiff_metadata metadata;
    metadata.insert({ TIFFTAG_COLORMATRIX1, std::vector<float>{ 1.0781, -0.4173, -0.0976, -0.0633, 0.9661, 0.0972, 0.0073, 0.1349, 0.3481 } });
    // metadata.insert({ TIFFTAG_ASSHOTNEUTRAL, std::vector<float>{ 1 / 2.001460, 1, 1 / 1.864002 } });
    metadata.insert({ TIFFTAG_CFAREPEATPATTERNDIM, std::vector<uint16_t>{ 2, 2 } });
    metadata.insert({ TIFFTAG_CFAPATTERN, std::vector<uint8_t>{ 2, 1, 1, 0 } });
    metadata.insert({ TIFFTAG_BLACKLEVEL, std::vector<float>{ 0 } });
    metadata.insert({ TIFFTAG_WHITELEVEL, std::vector<uint32_t>{ 0xfff } });

    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &metadata);
    return demosaicImage(*inputImage, &metadata, demosaicParameters, /*auto_white_balance=*/ false);
}

gls::image<gls::rgba_pixel>::unique_ptr demosaicIMX492DNG(const std::filesystem::path& input_path) {
    const DemosaicParameters demosaicParameters = {
        .contrast = 1.2,
        .saturation = 1.0,
        .toneCurveSlope = 3.5,
        .sharpening = 1.25,
        .sharpeningRadius = 7,
    };

    gls::tiff_metadata metadata;
    metadata.insert({ TIFFTAG_COLORMATRIX1, std::vector<float>{ 1.0781, -0.4173, -0.0976, -0.0633, 0.9661, 0.0972, 0.0073, 0.1349, 0.3481 } });
    metadata.insert({ TIFFTAG_ASSHOTNEUTRAL, std::vector<float>{ 1 / 2.001460, 1, 1 / 1.864002 } });
    metadata.insert({ TIFFTAG_CFAREPEATPATTERNDIM, std::vector<uint16_t>{ 2, 2 } });
    metadata.insert({ TIFFTAG_CFAPATTERN, std::vector<uint8_t>{ 1, 2, 0, 1 } });
    metadata.insert({ TIFFTAG_BLACKLEVEL, std::vector<float>{ 0 } });
    metadata.insert({ TIFFTAG_WHITELEVEL, std::vector<uint32_t>{ 0xfff } });
    metadata.insert({ TIFFTAG_MAKE, "Glass Imaging" });
    metadata.insert({ TIFFTAG_UNIQUECAMERAMODEL, "Glass 1" });

    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &metadata);
    return demosaicImage(*inputImage, &metadata, demosaicParameters, /*auto_white_balance=*/ true);
}

gls::image<gls::rgba_pixel>::unique_ptr demosaicIMX492PNG(const std::filesystem::path& input_path) {
    const DemosaicParameters demosaicParameters = {
        .contrast = 1.05,
        .saturation = 1.0,
        .toneCurveSlope = 3.5,
        .sharpening = 1.25,
        .sharpeningRadius = 7,
    };

    gls::tiff_metadata metadata;
    metadata.insert({ TIFFTAG_COLORMATRIX1, std::vector<float>{ 1.0781, -0.4173, -0.0976, -0.0633, 0.9661, 0.0972, 0.0073, 0.1349, 0.3481 } });
    metadata.insert({ TIFFTAG_ASSHOTNEUTRAL, std::vector<float>{ 1 / 2.001460, 1, 1 / 1.864002 } });
    metadata.insert({ TIFFTAG_CFAREPEATPATTERNDIM, std::vector<uint16_t>{ 2, 2 } });
    metadata.insert({ TIFFTAG_CFAPATTERN, std::vector<uint8_t>{ 1, 2, 0, 1 } });
    metadata.insert({ TIFFTAG_BLACKLEVEL, std::vector<float>{ 0 } });
    metadata.insert({ TIFFTAG_WHITELEVEL, std::vector<uint32_t>{ 0xfff } });
    metadata.insert({ TIFFTAG_MAKE, "Glass Imaging" });
    metadata.insert({ TIFFTAG_UNIQUECAMERAMODEL, "Glass 1" });

    const auto inputImage = gls::image<gls::luma_pixel_16>::read_png_file(input_path.string());
    return demosaicImage(*inputImage, &metadata, demosaicParameters, /*auto_white_balance=*/ false);
}

void copyMetadata(const gls::tiff_metadata& source, gls::tiff_metadata* destination, ttag_t tag) {
    const auto entry = source.find(tag);
    if (entry != source.end()) {
        destination->insert({ tag, entry->second });
    }
}

gls::image<gls::rgba_pixel>::unique_ptr demosaicAdobeDNG(const std::filesystem::path& input_path) {
    const DemosaicParameters demosaicParameters = {
        .contrast = 1.05,
        .saturation = 1.0,
        .toneCurveSlope = 3.5,
        .sharpening = 1.25,
        .sharpeningRadius = 5,
    };

    gls::tiff_metadata dng_metadata, exif_metadata;
    const auto inputImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);

    auto rgb_image = demosaicImage(*inputImage, &dng_metadata, demosaicParameters, /*auto_white_balance=*/ false);

    gls::tiff_metadata my_exif_metadata;
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_FNUMBER);
    copyMetadata(exif_metadata, &my_exif_metadata, EXIFTAG_EXPOSUREPROGRAM);
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
    auto output_file = (input_path.parent_path() / input_path.stem()).string() + "_my.dng";
    inputImage->write_dng_file(output_file, /*compression=*/ gls::JPEG, &dng_metadata, &my_exif_metadata);

    return rgb_image;
}

int main(int argc, const char* argv[]) {
    printf("RawPipeline Test!\n");

    if (argc > 1) {
        auto input_path = std::filesystem::path(argv[1]);

        LOG_INFO(TAG) << "Processing: " << input_path.filename() << std::endl;

        const auto rgb_image = demosaicAdobeDNG(input_path);
        rgb_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_rgb.png", /*skip_alpha=*/ true);
    }
}
