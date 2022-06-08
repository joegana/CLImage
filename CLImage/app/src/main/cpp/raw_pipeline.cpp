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

#include "CameraCalibration.hpp"

static const char* TAG = "RawPipeline Test";

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

int main(int argc, const char* argv[]) {
    printf("RawPipeline Test!\n");

    if (argc > 1) {
        gls::OpenCLContext glsContext("");
        RawConverter rawConverter(&glsContext);

        auto input_path = std::filesystem::path(argv[1]);

        // calibrateIMX571(&rawConverter, input_path.parent_path());
        // calibrateIMX492(&rawConverter, input_path.parent_path());
        // calibrateCanonEOSRP(&rawConverter, input_path.parent_path());
        // calibrateiPhone11(&rawConverter, input_path.parent_path());
        // calibrateRicohGRIII(&rawConverter, input_path.parent_path());
        // calibrateSonya6400(&rawConverter, input_path.parent_path());

        auto input_dir = std::filesystem::path(input_path.parent_path());
        std::vector<std::filesystem::path> directory_listing;
        std::copy(std::filesystem::directory_iterator(input_dir), std::filesystem::directory_iterator(),
                  std::back_inserter(directory_listing));
        std::sort(directory_listing.begin(), directory_listing.end());

        for (const auto& input_path : directory_listing) {
            const auto extension = input_path.extension();
            if ((extension != ".dng" && extension != ".DNG") || input_path.filename().string().starts_with(".")) {
                continue;
            }

            LOG_INFO(TAG) << "Processing: " << input_path.filename() << std::endl;

            // transcodeAdobeDNG(input_path);
            // const auto rgb_image = demosaicIMX571DNG(&rawConverter, input_path);
            // const auto rgb_image = demosaicSonya6400DNG(&rawConverter, input_path);
            // const auto rgb_image = demosaicCanonEOSRPDNG(&rawConverter, input_path);
            // const auto rgb_image = demosaiciPhone11(&rawConverter, input_path);
            const auto rgb_image = demosaicRicohGRIII2DNG(&rawConverter, input_path);
            // const auto rgb_image = demosaicLeicaQ2DNG(&rawConverter, input_path);
            rgb_image->write_jpeg_file((input_path.parent_path() / input_path.stem()).string() + "_rgb_new_ltmb.jpg", 95);
        }

//        LOG_INFO(TAG) << "Processing: " << input_path.filename() << std::endl;
//
//        // const auto rgb_image = demosaiciPhone11(&rawConverter, input_path);
//        // const auto rgb_image = demosaicCanonEOSRPDNG(&rawConverter, input_path);
//        const auto rgb_image = demosaicSonya6400DNG(&rawConverter, input_path);
//        // const auto rgb_image = demosaicRicohGRIII2DNG(&rawConverter, input_path);
//        rgb_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_rgb.png", /*skip_alpha=*/ true);

//        {
//            gls::tiff_metadata dng_metadata, exif_metadata;
//            const auto rawImage = gls::image<gls::luma_pixel_16>::read_dng_file(input_path.string(), &dng_metadata, &exif_metadata);
//            auto cpu_image = demosaicImageCPU(*rawImage, &dng_metadata, false);
//            cpu_image->apply([](gls::rgb_pixel_16* p, int x, int y) {
//                *p = {
//                    (uint16_t) (0xffff * sqrt((*p)[0] / (float) 0xffff)),
//                    (uint16_t) (0xffff * sqrt((*p)[1] / (float) 0xffff)),
//                    (uint16_t) (0xffff * sqrt((*p)[2] / (float) 0xffff))
//                };
//            });
//            cpu_image->write_png_file((input_path.parent_path() / input_path.stem()).string() + "_cpu_rgb.png", /*skip_alpha=*/ true);
//        }
    }
}
