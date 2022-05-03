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

#ifndef demosaic_hpp
#define demosaic_hpp

#include "gls_image.hpp"
#include "gls_tiff_metadata.hpp"
#include "gls_linalg.hpp"

enum BayerPattern {
    grbg = 0,
    gbrg = 1,
    rggb = 2,
    bggr = 3
};

typedef struct DenoiseParameters {
    float lumaSigma;
    float cbSigma;
    float crSigma;
    float sharpening;
} DenoiseParameters;

typedef struct RGBConversionParameters {
    float contrast;
    float saturation;
    float toneCurveSlope;
} RGBConversionParameters;

typedef struct NoiseModel {
    gls::Vector<4> rawNlf;
    gls::Matrix<5, 3> pyramidNlf;
} NoiseModel;

typedef struct DemosaicParameters {
    // Basic Debayering Parameters
    BayerPattern bayerPattern;
    float black_level;
    float white_level;
    gls::Vector<4> scale_mul;
    gls::Matrix<3, 3> rgb_cam;

    // Noise Estimation and Reduction parameters
    NoiseModel noiseModel;
    std::array<DenoiseParameters, 5> denoiseParameters;

    // Camera Color Space to RGB Parameters
    RGBConversionParameters rgbConversionParameters;
} DemosaicParameters;

const gls::point bayerOffsets[4][4] = {
    { {1, 0}, {0, 0}, {0, 1}, {1, 1} }, // grbg
    { {0, 1}, {0, 0}, {1, 0}, {1, 1} }, // gbrg
    { {0, 0}, {1, 0}, {1, 1}, {0, 1} }, // rggb
    { {1, 1}, {1, 0}, {0, 0}, {0, 1} }  // bggr
};

// sRGB -> XYZ D65 Transform: xyz_rgb * rgb_color -> xyz_color
const gls::Matrix<3, 3> xyz_rgb = {
    { 0.4124564, 0.3575761, 0.1804375 },
    { 0.2126729, 0.7151522, 0.0721750 },
    { 0.0193339, 0.1191920, 0.9503041 }
};

// XYZ D65 -> sRGB Transform: rgb_xyz * xyx_color -> rgb_color
const gls::Matrix<3, 3> rgb_xyz = {
    {  3.2404542, -1.5371385, -0.4985314 },
    { -0.9692660,  1.8760108,  0.0415560 },
    {  0.0556434, -0.2040259,  1.0572252 }
};

inline uint16_t clamp_uint16(int x) { return x < 0 ? 0 : x > 0xffff ? 0xffff : x; }
inline uint8_t clamp_uint8(int x) { return x < 0 ? 0 : x > 0xff ? 0xff : x; }

template <typename T>
gls::rectangle rotate180(const gls::rectangle& rect, const gls::image<T>& image) {
    return {
        image.width - rect.x - rect.width,
        image.height - rect.y - rect.height,
        rect.width,
        rect.height
    };
}

template <int levels>
gls::Matrix<levels, 3> lerpNLF(const gls::Matrix<levels, 3>& NLFData0, const gls::Matrix<levels, 3>& NLFData1, float a) {
    gls::Matrix<levels, 3> result;
    for (int j = 0; j < levels; j++) {
        for (int i = 0; i < 3; i++) {
            result[j][i] = std::lerp(NLFData0[j][i], NLFData1[j][i], a);
        }
    }
    return result;
}

template <int levels>
gls::Matrix<levels, 3> nlfFromIso(const std::array<NoiseModel, 6>& NLFData, int iso) {
    iso = std::clamp(iso, 100, 3200);
    if (iso >= 100 && iso < 200) {
        float a = (iso - 100) / 100;
        return lerpNLF<levels>(NLFData[0].pyramidNlf, NLFData[1].pyramidNlf, a);
    } else if (iso >= 200 && iso < 400) {
        float a = (iso - 200) / 200;
        return lerpNLF<levels>(NLFData[1].pyramidNlf, NLFData[2].pyramidNlf, a);
    } else if (iso >= 400 && iso < 800) {
        float a = (iso - 400) / 400;
        return lerpNLF<levels>(NLFData[2].pyramidNlf, NLFData[3].pyramidNlf, a);
    } else if (iso >= 800 && iso < 1600) {
        float a = (iso - 800) / 800;
        return lerpNLF<levels>(NLFData[3].pyramidNlf, NLFData[4].pyramidNlf, a);
    } else /* if (iso >= 1600 && iso <= 3200) */ {
        float a = (iso - 1600) / 1600;
        return lerpNLF<levels>(NLFData[4].pyramidNlf, NLFData[5].pyramidNlf, a);
    }
}

void white_balance(const gls::image<gls::luma_pixel_16>& rawImage, gls::Vector<3>* wb_mul, uint32_t white, uint32_t black, BayerPattern bayerPattern);

void interpolateGreen(const gls::image<gls::luma_pixel_16>& rawImage,
                      gls::image<gls::rgb_pixel_16>* rgbImage, BayerPattern bayerPattern);

void interpolateRedBlue(gls::image<gls::rgb_pixel_16>* image, BayerPattern bayerPattern);

gls::image<gls::rgb_pixel_16>::unique_ptr demosaicImageCPU(const gls::image<gls::luma_pixel_16>& rawImage,
                                                        gls::tiff_metadata* metadata, bool auto_white_balance);

gls::image<gls::rgba_pixel>::unique_ptr demosaicImage(const gls::image<gls::luma_pixel_16>& rawImage, gls::tiff_metadata* metadata,
                                                      DemosaicParameters* demosaicParameters, bool auto_white_balance,
                                                      const gls::rectangle* gmb_position, bool rotate_180);

gls::image<gls::rgba_pixel>::unique_ptr fastDemosaicImage(const gls::image<gls::luma_pixel_16>& rawImage, gls::tiff_metadata* metadata,
                                                          const DemosaicParameters& demosaicParameters, bool auto_white_balance);

gls::Matrix<3, 3> cam_xyz_coeff(gls::Vector<3>* pre_mul, const gls::Matrix<3, 3>& cam_xyz);

void colorcheck(const gls::image<gls::luma_pixel_16>& rawImage, BayerPattern bayerPattern, uint32_t black, std::array<gls::rectangle, 24> gmb_samples);

void white_balance(const gls::image<gls::luma_pixel_16>& rawImage, gls::Vector<3>* wb_mul, uint32_t white, uint32_t black, BayerPattern bayerPattern);

void unpackDNGMetadata(const gls::image<gls::luma_pixel_16>& rawImage,
                       gls::tiff_metadata* dng_metadata,
                       DemosaicParameters* demosaicParameters,
                       bool auto_white_balance, const gls::rectangle* gmb_position, bool rotate_180);

gls::Matrix<3, 3> cam_ycbcr(const gls::Matrix<3, 3>& rgb_cam);

gls::Vector<3> extractNlfFromColorChecker(gls::image<gls::rgba_pixel_float>* yCbCrImage, const gls::rectangle gmb_position, bool rotate_180, int scale);

enum GMBColors {
    DarkSkin        = 0,
    LightSkin       = 1,
    BlueSky         = 2,
    Foliage         = 3,
    BlueFlower      = 4,
    BluishGreen     = 5,
    Orange          = 6,
    PurplishBlue    = 7,
    ModerateRed     = 8,
    Purple          = 9,
    YellowGreen     = 10,
    OrangeYellow    = 11,
    Blue            = 12,
    Green           = 13,
    Red             = 14,
    Yellow          = 15,
    Magenta         = 16,
    Cyan            = 17,
    White           = 18,
    Neutral_8       = 19,
    Neutral_6_5     = 20,
    Neutral_5       = 21,
    Neutral_3_5     = 22,
    Black           = 23
};

extern const char* GMBColorNames[24];

struct PatchStats {
    gls::Vector<3> mean;
    gls::Vector<3> variance;
};

struct RawPatchStats {
    gls::Vector<4> mean;
    gls::Vector<4> variance;
};

void colorCheckerRawStats(const gls::image<gls::luma_pixel_16>& rawImage, float black_level, float white_level, BayerPattern bayerPattern, const gls::rectangle& gmb_position, bool rotate_180, std::array<RawPatchStats, 24>* stats);

gls::Vector<4> estimateRawParameters(const gls::image<gls::luma_pixel_16>& rawImage, gls::Matrix<3, 3>* cam_xyz, gls::Vector<3>* pre_mul,
                                     float black_level, float white_level, BayerPattern bayerPattern, const gls::rectangle& gmb_position, bool rotate_180);

void colorcheck(const std::array<RawPatchStats, 24>& rawStats);

#endif /* demosaic_hpp */
