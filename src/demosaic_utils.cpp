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

#include "demosaic.hpp"

#include <numeric>
#include <iomanip>

#include "gls_color_science.hpp"
#include "gls_image.hpp"

gls::Matrix<3, 3> cam_xyz_coeff(gls::Vector<3>& pre_mul, const gls::Matrix<3, 3>& cam_xyz) {
    // Compute sRGB -> XYZ -> Camera
    auto rgb_cam = cam_xyz * xyz_rgb;

    // Normalize rgb_cam so that rgb_cam * (1,1,1) == (1,1,1).
    // This maximizes the uint16 dynamic range and makes sure
    // that highlight clipping is white in both camera and target
    // color spaces, so that clipping doesn't turn pink

    auto cam_white = rgb_cam * gls::Vector<3>({ 1, 1, 1 });

    gls::Matrix<3, 3> mPreMul = {
        { 1 / cam_white[0], 0, 0 },
        { 0, 1 / cam_white[1], 0 },
        { 0, 0, 1 / cam_white[2] }
    };

    for (int i = 0; i < 3; i++) {
        if (cam_white[i] > 0.00001) {
            pre_mul[i] = 1 / cam_white[i];
        } else {
            throw std::range_error("");
        }
    }

    // Return Camera -> sRGB
    return inverse(mPreMul * rgb_cam);
}

gls::rectangle alignToQuad(const gls::rectangle& rect) {
    gls::rectangle alignedRect = rect;
    if (alignedRect.y & 1) {
        alignedRect.y += 1;
        alignedRect.height -= 1;
    }
    if (alignedRect.height & 1) {
        alignedRect.height -= 1;
    }
    if (alignedRect.x & 1) {
        alignedRect.x += 1;
        alignedRect.width -= 1;
    }
    if (alignedRect.width & 1) {
        alignedRect.width -= 1;
    }
    return alignedRect;
}

void colorcheck(const gls::image<gls::luma_pixel_16>& rawImage, BayerPattern bayerPattern, uint32_t black, std::array<gls::rectangle, 24> gmb_samples) {
// ColorChecker Chart under 6500-kelvin illumination
  static gls::Matrix<gmb_samples.size(), 3> gmb_xyY = {
    { 0.400, 0.350, 10.1 },        // Dark Skin
    { 0.377, 0.345, 35.8 },        // Light Skin
    { 0.247, 0.251, 19.3 },        // Blue Sky
    { 0.337, 0.422, 13.3 },        // Foliage
    { 0.265, 0.240, 24.3 },        // Blue Flower
    { 0.261, 0.343, 43.1 },        // Bluish Green
    { 0.506, 0.407, 30.1 },        // Orange
    { 0.211, 0.175, 12.0 },        // Purplish Blue
    { 0.453, 0.306, 19.8 },        // Moderate Red
    { 0.285, 0.202, 6.6  },        // Purple
    { 0.380, 0.489, 44.3 },        // Yellow Green
    { 0.473, 0.438, 43.1 },        // Orange Yellow
    { 0.187, 0.129, 6.1  },        // Blue
    { 0.305, 0.478, 23.4 },        // Green
    { 0.539, 0.313, 12.0 },        // Red
    { 0.448, 0.470, 59.1 },        // Yellow
    { 0.364, 0.233, 19.8 },        // Magenta
    { 0.196, 0.252, 19.8 },        // Cyan
    { 0.310, 0.316, 90.0 },        // White
    { 0.310, 0.316, 59.1 },        // Neutral 8
    { 0.310, 0.316, 36.2 },        // Neutral 6.5
    { 0.310, 0.316, 19.8 },        // Neutral 5
    { 0.310, 0.316, 9.0 },         // Neutral 3.5
    { 0.310, 0.316, 3.1 } };       // Black

    const auto offsets = bayerOffsets[bayerPattern];

    auto* writeRawImage = (gls::image<gls::luma_pixel_16>*) &rawImage;

    gls::Matrix<gmb_samples.size(), 4> gmb_cam;
    gls::Matrix<gmb_samples.size(), 3> gmb_xyz;

    for (int sq = 0; sq < gmb_samples.size(); sq++) {
        std::array<int, 3> count { /* zero */ };
        auto patch = alignToQuad(gmb_samples[sq]);
        for (int y = patch.y; y < patch.y + patch.height; y += 2) {
            for (int x = patch.x; x < patch.x + patch.width; x += 2) {
                for (int c = 0; c < 4; c++) {
                    const auto& o = offsets[c];
                    int val = rawImage[y + o.y][x + o.x];
                    gmb_cam[sq][c == 3 ? 1 : c] += (float) val;
                    count[c == 3 ? 1 : c]++;
                    // Mark image to identify sampled areas
                    (*writeRawImage)[y + o.y][x + o.x] = black + (val - black) / 2;
                }
            }
        }

        for (int c = 0; c < 3; c++) {
            gmb_cam[sq][c] = gmb_cam[sq][c] / (float) count[c] - (float) black;
        }
        gmb_xyz[sq][0] = gmb_xyY[sq][2] * gmb_xyY[sq][0] / gmb_xyY[sq][1];
        gmb_xyz[sq][1] = gmb_xyY[sq][2];
        gmb_xyz[sq][2] = gmb_xyY[sq][2] * (1 - gmb_xyY[sq][0] - gmb_xyY[sq][1]) / gmb_xyY[sq][1];
    }

    gls::Matrix<gmb_samples.size(), 3> inverse = pseudoinverse(gmb_xyz);

    gls::Matrix<3, 3> cam_xyz;
    for (int pass=0; pass < 2; pass++) {
        for (int i = 0; i < 3 /* colors */; i++) {
            for (int j = 0; j < 3; j++) {
                cam_xyz[i][j] = 0;
                for (int k = 0; k < gmb_samples.size(); k++)
                    cam_xyz[i][j] += gmb_cam[k][i] * inverse[k][j];
            }
        }

        gls::Vector<3> pre_mul;
        cam_xyz_coeff(pre_mul, cam_xyz);

        gls::Vector<4> balance;
        for (int c = 0; c < 4; c++) {
            balance[c] = pre_mul[c == 3 ? 1 : c] * gmb_cam[20][c];
        }
        for (int sq = 0; sq < gmb_samples.size(); sq++) {
            for (int c = 0; c < 4; c++) {
                gmb_cam[sq][c] *= balance[c];
            }
        }
    }

    float norm = 1 / (cam_xyz[1][0] + cam_xyz[1][1] + cam_xyz[1][2]);
    printf("Color Matrix: ");
    for (int c = 0; c < 3; c++) {
        for (int j = 0; j < 3; j++)
            printf("%.4f, ", cam_xyz[c][j] * norm);
    }
    printf("\n");
}

void white_balance(const gls::image<gls::luma_pixel_16>& rawImage, gls::Vector<3>* wb_mul, uint32_t white, uint32_t black, BayerPattern bayerPattern) {
    const auto offsets = bayerOffsets[bayerPattern];

    std::array<float, 8> fsum { /* zero */ };
    for (int y = 0; y < rawImage.height/2; y += 8) {
        for (int x = 0; x < rawImage.width/2; x += 8) {
            std::array<uint32_t, 8> sum { /* zero */ };
            for (int j = y; j < 8 && j < rawImage.height/2; j++) {
                for (int i = x; i < 8 && i < rawImage.width/2; i++) {
                    for (int c = 0; c < 4; c++) {
                        const auto& o = offsets[c];
                        uint32_t val = rawImage[2 * j + o.y][2 * i + o.x];
                        if (val > white - 25) {
                            goto skip_block;
                        }
                        if ((val -= black) < 0) {
                            val = 0;
                        }
                        sum[c] += val;
                        sum[c+4]++;
                    }
                }
            }
            for (int i = 0; i < 8; i++) {
                fsum[i] += (float) sum[i];
            }
            skip_block:
            ;
        }
    }
    // Aggregate green2 data to green
    fsum[1] += fsum[3];
    fsum[5] += fsum[7];

    for (int c = 0; c < 3; c++) {
        if (fsum[c] != 0) {
            (*wb_mul)[c] = fsum[c+4] / fsum[c];
        }
    }
    // Normalize with green = 1
    *wb_mul = *wb_mul / (*wb_mul)[1];
}

// Coordinates of the GretagMacbeth ColorChecker squares
// x, y, width, height from 2022-04-12-10-43-56-566.dng
static std::array<gls::rectangle, 24> gmb_samples = {{
    { 4886, 2882, 285, 273 },
    { 4505, 2899, 272, 235 },
    { 4122, 2892, 262, 240 },
    { 3742, 2900, 256, 225 },
    { 3352, 2897, 258, 227 },
    { 2946, 2904, 282, 231 },
    { 4900, 2526, 274, 244 },
    { 4513, 2526, 262, 237 },
    { 4133, 2529, 235, 227 },
    { 3733, 2523, 254, 237 },
    { 3347, 2530, 245, 234 },
    { 2932, 2531, 283, 233 },
    { 4899, 2151, 283, 252 },
    { 4519, 2155, 261, 245 },
    { 4119, 2157, 269, 245 },
    { 3737, 2160, 246, 226 },
    { 3335, 2168, 261, 239 },
    { 2957, 2183, 233, 214 },
    { 4923, 1784, 265, 243 },
    { 4531, 1801, 250, 219 },
    { 4137, 1792, 234, 226 },
    { 3729, 1790, 254, 230 },
    { 3337, 1793, 250, 232 },
    { 2917, 1800, 265, 228 },
}};

void unpackRawMetadata(const gls::image<gls::luma_pixel_16>& rawImage,
                       gls::tiff_metadata* metadata,
                       BayerPattern *bayerPattern,
                       float *black_level,
                       gls::Vector<4> *scale_mul,
                       gls::Matrix<3, 3> *rgb_cam,
                       bool auto_white_balance) {
    const auto color_matrix1 = getVector<float>(*metadata, TIFFTAG_COLORMATRIX1);
    const auto color_matrix2 = getVector<float>(*metadata, TIFFTAG_COLORMATRIX2);

    // If present ColorMatrix2 is usually D65 and ColorMatrix1 is Standard Light A
    const auto& color_matrix = color_matrix2.empty() ? color_matrix1 : color_matrix2;

    auto as_shot_neutral = getVector<float>(*metadata, TIFFTAG_ASSHOTNEUTRAL);
    const auto black_level_vec = getVector<float>(*metadata, TIFFTAG_BLACKLEVEL);
    const auto white_level_vec = getVector<uint32_t>(*metadata, TIFFTAG_WHITELEVEL);
    const auto cfa_pattern = getVector<uint8_t>(*metadata, TIFFTAG_CFAPATTERN);

    *black_level = black_level_vec.empty() ? 0 : black_level_vec[0];
    const uint32_t white_level = white_level_vec.empty() ? 0xffff : white_level_vec[0];

    *bayerPattern = std::memcmp(cfa_pattern.data(), "\00\01\01\02", 4) == 0 ? BayerPattern::rggb
                  : std::memcmp(cfa_pattern.data(), "\02\01\01\00", 4) == 0 ? BayerPattern::bggr
                  : std::memcmp(cfa_pattern.data(), "\01\00\02\01", 4) == 0 ? BayerPattern::grbg
                  : BayerPattern::gbrg;

    std::cout << "as_shot_neutral: " << gls::Vector<3>(as_shot_neutral) << std::endl;

    // Uncomment this to characterize sensor
    // colorcheck(rawImage, *bayerPattern, *black_level, gmb_samples);

    gls::Vector<3> cam_mul = 1.0 / gls::Vector<3>(as_shot_neutral);

    // TODO: this should be CameraCalibration * ColorMatrix * AsShotWhite
    gls::Matrix<3, 3> cam_xyz = color_matrix;

    std::cout << "cam_xyz:\n" << cam_xyz << std::endl;

    gls::Vector<3> pre_mul;
    *rgb_cam = cam_xyz_coeff(pre_mul, cam_xyz);

    std::cout << "*** pre_mul: " << pre_mul << std::endl;
    std::cout << "*** cam_mul: " << cam_mul << std::endl;

    // Save the whitening transformation
    const auto inv_cam_white = pre_mul;

    if (auto_white_balance) {
        white_balance(rawImage, &cam_mul, white_level, *black_level, *bayerPattern);

        printf("Auto White Balance: %f, %f, %f\n", cam_mul[0], cam_mul[1], cam_mul[2]);

        // Convert cam_mul from camera to XYZ
        const auto cam_mul_xyz = cam_xyz * cam_mul;
        std::cout << "cam_mul_xyz: " << cam_mul_xyz << ", CCT: " << XYZtoCorColorTemp(cam_mul_xyz) << std::endl;

        for (int c = 0; c < 3; c++) {
            as_shot_neutral[c] = 1 / cam_mul[c];
        }
        (*metadata)[TIFFTAG_ASSHOTNEUTRAL] = as_shot_neutral;
    }

    gls::Matrix<3, 3> mCamMul = {
        { pre_mul[0] / cam_mul[0], 0, 0 },
        { 0, pre_mul[1] / cam_mul[1], 0 },
        { 0, 0, pre_mul[2] / cam_mul[2] }
    };

    // If cam_mul is available use that instead of pre_mul
    for (int i = 0; i < 3; i++) {
        pre_mul[i] = cam_mul[i];
    }

    {
        const gls::Vector<3> d65_white = { 0.95047, 1.0, 1.08883 };
        const gls::Vector<3> d50_white = { 0.9642, 1.0000, 0.8249 };

        std::cout << "XYZ D65 White -> sRGB: " << rgb_xyz * d65_white << std::endl;
        std::cout << "sRGB White -> XYZ D65: " << xyz_rgb * gls::Vector({1, 1, 1}) << std::endl;
        std::cout << "inverse(cam_xyz) * (1 / pre_mul): " << inverse(cam_xyz) * (1 / pre_mul) << std::endl;

        auto cam_white = inverse(cam_xyz) * xyz_rgb * gls::Vector<3>({ 1, 1, 1 });
        std::cout << "inverse(cam_xyz) * cam_white: " << cam_xyz * cam_white << ", CCT: " << XYZtoCorColorTemp(cam_xyz * cam_white) << std::endl;
        std::cout << "xyz_rgb * gls::Vector<3>({ 1, 1, 1 }): " << xyz_rgb * gls::Vector<3>({ 1, 1, 1 }) << ", CCT: " << XYZtoCorColorTemp(xyz_rgb * gls::Vector<3>({ 1, 1, 1 })) << std::endl;

        const auto wb_out = xyz_rgb * *rgb_cam * mCamMul * gls::Vector({1, 1, 1});
        std::cout << "wb_out: " << wb_out << ", CCT: " << XYZtoCorColorTemp(wb_out) << std::endl;

        const auto no_wb_out = xyz_rgb * *rgb_cam * (1 / inv_cam_white);
        std::cout << "no_wb_out: " << wb_out << ", CCT: " << XYZtoCorColorTemp(no_wb_out) << std::endl;
    }

    // Scale Input Image
    auto minmax = std::minmax_element(std::begin(pre_mul), std::end(pre_mul));
    for (int c = 0; c < 4; c++) {
        int pre_mul_idx = c == 3 ? 1 : c;
        printf("pre_mul[c]: %f, *minmax.second: %f, white_level: %d\n", pre_mul[pre_mul_idx], *minmax.second, white_level);
        (*scale_mul)[c] = (pre_mul[pre_mul_idx] / *minmax.first) * 65535.0 / (white_level - *black_level);
    }
    printf("scale_mul: %f, %f, %f, %f\n", (*scale_mul)[0], (*scale_mul)[1], (*scale_mul)[2], (*scale_mul)[3]);
}

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

const char* GMBColorNames[24] {
    "DarkSkin",
    "LightSkin",
    "BlueSky",
    "Foliage",
    "BlueFlower",
    "BluishGreen",
    "Orange",
    "PurplishBlue",
    "ModerateRed",
    "Purple",
    "YellowGreen",
    "OrangeYellow",
    "Blue",
    "Green",
    "Red",
    "Yellow",
    "Magenta",
    "Cyan",
    "White",
    "Neutral_8",
    "Neutral_6_5",
    "Neutral_5",
    "Neutral_3_5",
    "Black"
};

struct PatchStats {
    std::array<float, 3> mean;
    std::array<float, 3> variance;
};

float square(float x) {
    return x * x;
}

// Collect mean and variance of ColorChecker patches
void colorCheckerStats(gls::image<gls::rgba_pixel_float>* image, const gls::rectangle& gmb_position, std::array<PatchStats, 24>* stats) {
    // std::cout << "rectangle: " << gmb_position.x << ", " << gmb_position.y << ", " << gmb_position.width << ", " << gmb_position.height << std::endl;

    int patch_width = gmb_position.width / 6;
    int patch_height = gmb_position.height / 4;

    int patchIdx = 0;
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 6; col++, patchIdx++) {
            gls::rectangle patch = {
                gmb_position.x + col * patch_width + (int) (0.2 * patch_width),
                gmb_position.y + row * patch_height + (int) (0.2 * patch_height),
                (int) (0.6 * patch_width),
                (int) (0.6 * patch_height) };

            int patchSamples = patch.width * patch.height;

            float avgY = 0;
            float avgCb = 0;
            float avgCr = 0;

            for (int y = 0; y < patch.height; y++) {
                for (int x = 0; x < patch.width; x++) {
                    const auto& p = (*image)[patch.y + y][patch.x + x];
                    avgY += p[0];
                    avgCr += p[1];
                    avgCb += p[2];
                }
            }

            avgY /= patchSamples;
            avgCb /= patchSamples;
            avgCr /= patchSamples;

            float varY = 0;
            float varCb = 0;
            float varCr = 0;

            for (int y = 0; y < patch.height; y++) {
                for (int x = 0; x < patch.width; x++) {
                    const auto& p = (*image)[patch.y + y][patch.x + x];
                    varY += square(p[0] - avgY);
                    varCr += square(p[1] - avgCb);
                    varCb += square(p[2] - avgCr);

                    (*image)[patch.y + y][patch.x + x] = {0, 0, 0, 0};
                }
            }

            varY /= patchSamples;
            varCb /= patchSamples;
            varCr /= patchSamples;

//            std::cout << std::setw(12) << std::setfill(' ') << GMBColorNames[patchIdx] << " - avg(" << patchSamples << "): {"
//                      << std::setprecision(2) << avgY << ", " << avgCb << ", " << avgCr << "}, var: {" << varY << ", " << varCb << ", " << varCr << "}" << std::endl;

            (*stats)[patchIdx] = {{avgY, avgCb, avgCr}, {varY, varCb, varCr}};
        }
    }
}

// Slope Regression of a set of points
template <size_t N>
std::pair<float, float> linear_regression(const std::array<float, N>& x, const std::array<float, N>& y) {
    const auto s_x  = std::accumulate(x.begin(), x.end(), 0.0);
    const auto s_y  = std::accumulate(y.begin(), y.end(), 0.0);
    const auto s_xx = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
    const auto s_xy = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
    const auto b    = (N * s_xy - s_x * s_y) / (N * s_xx - s_x * s_x);
    const auto a    = (s_y - b * s_x) / N;
    return { a, b };
}

// Estimate the Sensor's Noise Level Function (NLF: variance vs intensity), which is linear going through zero
std::array<float, 3> estimateNlfParameters(gls::image<gls::rgba_pixel_float>* image, const gls::rectangle& gmb_position) {
    std::array<PatchStats, 24> stats;
    colorCheckerStats(image, gmb_position, &stats);

    std::array<float, 6> y_intensity = {
        stats[Black].mean[0],
        stats[Neutral_3_5].mean[0],
        stats[Neutral_5].mean[0],
        stats[Neutral_6_5].mean[0],
        stats[Neutral_8].mean[0],
        stats[White].mean[0]
    };

    std::array<float, 6> y_variance = {
        stats[Black].variance[0],
        stats[Neutral_3_5].variance[0],
        stats[Neutral_5].variance[0],
        stats[Neutral_6_5].variance[0],
        stats[Neutral_8].variance[0],
        stats[White].variance[0]
    };

    std::array<float, 6> cb_variance = {
        stats[Black].variance[1],
        stats[Neutral_3_5].variance[1],
        stats[Neutral_5].variance[1],
        stats[Neutral_6_5].variance[1],
        stats[Neutral_8].variance[1],
        stats[White].variance[1]
    };

    std::array<float, 6> cr_variance = {
        stats[Black].variance[2],
        stats[Neutral_3_5].variance[2],
        stats[Neutral_5].variance[2],
        stats[Neutral_6_5].variance[2],
        stats[Neutral_8].variance[2],
        stats[White].variance[2]
    };

//    std::cout << "NLF Stats:" << std::endl;
//    for (int patch = Black; patch >= White; patch--) {
//        std::cout << std::setprecision(4) << std::setw(4) << stats[patch].mean[0] << "\t"
//                  << stats[patch].variance[0] << "\t" << stats[patch].variance[1] << "\t" << stats[patch].variance[2] << std::endl;
//    }

    auto nlf_y = linear_regression(y_intensity, y_variance);
    auto nlf_cb = linear_regression(y_intensity, cb_variance);
    auto nlf_cr = linear_regression(y_intensity, cr_variance);

//    std::cout << std::setprecision(4) << std::setw(4)
//              << "nlf_y: " << nlf_y.first << ":" << nlf_y.second
//              << ", nlf_cb: " << nlf_cb.first << ":" << nlf_cb.second
//              << ", nlf_cr: " << nlf_cr.first << ":" << nlf_cr.second << std::endl;

    // NFL for Y passes by 0, just use the slope, NFL for Cb and and Cr is mostly flat, just return the average
    return {nlf_y.second, nlf_cb.first + 0.5f * nlf_cb.first, nlf_cr.first + 0.5f * nlf_cr.second};
}

std::array<float, 3> extractNlfFromColorChecker(gls::image<gls::rgba_pixel_float>* yCbCrImage, const gls::rectangle gmb_position, int scale) {
    const gls::rectangle position = {
        (int) round(gmb_position.x / (float) scale),
        (int) round(gmb_position.y / (float) scale),
        (int) round(gmb_position.width / (float) scale),
        (int) round(gmb_position.height / (float) scale)
    };
    std::array<float, 3> nlf_parameters = estimateNlfParameters(yCbCrImage, position);
    std::cout << "Scale " << scale << " nlf parameters: " << nlf_parameters[0] << ", " << nlf_parameters[1] << ", " << nlf_parameters[2] << std::endl;

    gls::image<gls::rgb_pixel> output(yCbCrImage->width, yCbCrImage->height);
    for (int y = 0; y < output.height; y++) {
        for (int x = 0; x < output.width; x++) {
            const auto& p = (*yCbCrImage)[y][x];
            output[y][x] = {clamp_uint8(255 * p[0]), clamp_uint8(128 * p[1] + 127), clamp_uint8(128 * p[2] + 127)};
        }
    }
    output.write_png_file("/Users/fabio/ColorChecker" + std::to_string(scale) + ".png");

    return nlf_parameters;
}

gls::Matrix<3, 3> cam_ycbcr(const gls::Matrix<3, 3>& rgb_cam) {
    // Convert image to YCbCr for Luma/Chroma Denoising, use the camera's primaries to derive the transform
    const auto cam_y = xyz_rgb[1] * rgb_cam;
    const auto KR = cam_y[0];
    const auto KG = cam_y[1];
    const auto KB = cam_y[2];

    // See: https://en.wikipedia.org/wiki/YCbCr
    return {
        {  KR,                      KG,                     KB                   },
        { -0.5f * KR / (1.0f - KB), -0.5f * KG / (1 - KB),  0.5f                 },
        {  0.5f,                    -0.5f * KG / (1 - KR), -0.5f * KB / (1 - KR) }
    };
}
