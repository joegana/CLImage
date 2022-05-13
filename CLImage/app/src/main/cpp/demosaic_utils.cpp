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
#include "ThreadPool.hpp"

gls::Matrix<3, 3> cam_xyz_coeff(gls::Vector<3>* pre_mul, const gls::Matrix<3, 3>& cam_xyz) {
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
            (*pre_mul)[i] = 1 / cam_white[i];
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

std::ostream& operator<<(std::ostream& os, const std::span<const float>& s) {
    for (int i = 0; i < s.size(); i++) {
        os << s[i];
        if (i < s.size() - 1) {
            os << ", ";
        }
    }
    return os;
}

void matrixFromColorChecker(const std::array<RawPatchStats, 24>& rawStats, gls::Matrix<3, 3>* cam_xyz, gls::Vector<3>* pre_mul) {
// ColorChecker Chart under 6500-kelvin illumination
  static const gls::Matrix<24, 3> gmb_xyY = {
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

    gls::Matrix<24, 3> gmb_xyz;
    for (int sq = 0; sq < 24; sq++) {
        gmb_xyz[sq][0] = gmb_xyY[sq][2] * gmb_xyY[sq][0] / gmb_xyY[sq][1];
        gmb_xyz[sq][1] = gmb_xyY[sq][2];
        gmb_xyz[sq][2] = gmb_xyY[sq][2] * (1 - gmb_xyY[sq][0] - gmb_xyY[sq][1]) / gmb_xyY[sq][1];
    }

    gls::Matrix<24, 3> inverse = pseudoinverse(gmb_xyz);

    for (int pass=0; pass < 2; pass++) {
        for (int i = 0; i < 3 /* colors */; i++) {
            for (int j = 0; j < 3; j++) {
                (*cam_xyz)[i][j] = 0;
                for (int k = 0; k < 24; k++) {
                    (*cam_xyz)[i][j] += rawStats[k].mean[i] * inverse[k][j];
                }
            }
        }

        cam_xyz_coeff(pre_mul, *cam_xyz);
    }

    // Normalize the matrix
    *cam_xyz = *cam_xyz / ((*cam_xyz)[1][0] + (*cam_xyz)[1][1] + (*cam_xyz)[1][2]);
    *pre_mul = *pre_mul / (*pre_mul)[1];

    std::cout << "ColorChecker Color Matrix: " << std::fixed << std::setw(6) << std::setprecision(4) << cam_xyz->span() << std::endl;
    std::cout << "ColorChecker White Point: " << std::fixed << std::setw(6) << std::setprecision(4) << *pre_mul << std::endl;
}

void colorcheck(const gls::image<gls::luma_pixel_16>& rawImage, BayerPattern bayerPattern, uint32_t black, std::array<gls::rectangle, 24> gmb_samples) {
// ColorChecker Chart under 6500-kelvin illumination
  static gls::Matrix<24, 3> gmb_xyY = {
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

    gls::Matrix<24, 4> gmb_cam;
    gls::Matrix<24, 3> gmb_xyz;

    for (int sq = 0; sq < 24; sq++) {
        gmb_cam[sq] = {0, 0, 0, 0};
        std::array<int, 3> count = { 0, 0, 0 };
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

    gls::Matrix<24, 3> inverse = pseudoinverse(gmb_xyz);

    gls::Matrix<3, 3> cam_xyz;
    for (int pass=0; pass < 2; pass++) {
        for (int i = 0; i < 3 /* colors */; i++) {
            for (int j = 0; j < 3; j++) {
                cam_xyz[i][j] = 0;
                for (int k = 0; k < 24; k++)
                    cam_xyz[i][j] += gmb_cam[k][i] * inverse[k][j];
            }
        }

        gls::Vector<3> pre_mul;
        cam_xyz_coeff(&pre_mul, cam_xyz);

        gls::Vector<4> balance;
        for (int c = 0; c < 4; c++) {
            balance[c] = pre_mul[c == 3 ? 1 : c] * gmb_cam[20][c];
        }
        for (int sq = 0; sq < 24; sq++) {
            for (int c = 0; c < 4; c++) {
                gmb_cam[sq][c] *= balance[c];
            }
        }
    }

    float norm = 1 / (cam_xyz[1][0] + cam_xyz[1][1] + cam_xyz[1][2]);
    printf("DCRaw Color Matrix: ");
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

void unpackDNGMetadata(const gls::image<gls::luma_pixel_16>& rawImage,
                       gls::tiff_metadata* dng_metadata,
                       DemosaicParameters* demosaicParameters,
                       bool auto_white_balance,
                       const gls::rectangle* gmb_position, bool rotate_180) {
    const auto color_matrix1 = getVector<float>(*dng_metadata, TIFFTAG_COLORMATRIX1);
    const auto color_matrix2 = getVector<float>(*dng_metadata, TIFFTAG_COLORMATRIX2);

    // If present ColorMatrix2 is usually D65 and ColorMatrix1 is Standard Light A
    const auto& color_matrix = color_matrix2.empty() ? color_matrix1 : color_matrix2;

    auto as_shot_neutral = getVector<float>(*dng_metadata, TIFFTAG_ASSHOTNEUTRAL);
    std::cout << "as_shot_neutral: " << gls::Vector<3>(as_shot_neutral) << std::endl;

    const auto black_level_vec = getVector<float>(*dng_metadata, TIFFTAG_BLACKLEVEL);
    const auto white_level_vec = getVector<uint32_t>(*dng_metadata, TIFFTAG_WHITELEVEL);
    const auto cfa_pattern = getVector<uint8_t>(*dng_metadata, TIFFTAG_CFAPATTERN);

    demosaicParameters->black_level = black_level_vec.empty() ? 0 : black_level_vec[0];
    demosaicParameters->white_level = white_level_vec.empty() ? 0xffff : white_level_vec[0];

    demosaicParameters->bayerPattern = std::memcmp(cfa_pattern.data(), "\00\01\01\02", 4) == 0 ? BayerPattern::rggb
                                     : std::memcmp(cfa_pattern.data(), "\02\01\01\00", 4) == 0 ? BayerPattern::bggr
                                     : std::memcmp(cfa_pattern.data(), "\01\00\02\01", 4) == 0 ? BayerPattern::grbg
                                     : BayerPattern::gbrg;

    std::cout << "bayerPattern: " << demosaicParameters->bayerPattern << std::endl;

    gls::Vector<3> pre_mul;
    gls::Matrix<3, 3> cam_xyz;
    if (gmb_position) {
        demosaicParameters->noiseModel.rawNlf = estimateRawParameters(rawImage, &cam_xyz, &pre_mul,
                                                                      demosaicParameters->black_level,
                                                                      demosaicParameters->white_level,
                                                                      demosaicParameters->bayerPattern,
                                                                      *gmb_position, rotate_180);

        // Obtain the rgb_cam matrix and pre_mul
        demosaicParameters->rgb_cam = cam_xyz_coeff(&pre_mul, cam_xyz);
    } else {
        cam_xyz = color_matrix;
        // Obtain the rgb_cam matrix and pre_mul
        demosaicParameters->rgb_cam = cam_xyz_coeff(&pre_mul, cam_xyz);

        // If cam_mul is available use that instead of pre_mul
        if (!as_shot_neutral.empty()) {
            pre_mul = 1.0 / gls::Vector<3>(as_shot_neutral);
        }
    }

    std::cout << "cam_xyz: " << std::fixed << cam_xyz.span() << std::endl;
    std::cout << "*** pre_mul: " << pre_mul / pre_mul[1] << std::endl;

    if (auto_white_balance) {
//        gls::Vector<3> cam_mul;
//        white_balance(rawImage, &cam_mul, demosaicParameters->white_level, demosaicParameters->black_level, demosaicParameters->bayerPattern);

        auto cam_to_ycbcr = cam_ycbcr(demosaicParameters->rgb_cam);
        gls::Vector<3> cam_mul = autoWhiteBalance(rawImage, cam_to_ycbcr,
                                                  demosaicParameters->white_level, demosaicParameters->black_level, demosaicParameters->bayerPattern);

        printf("Auto White Balance: %f, %f, %f\n", cam_mul[0], cam_mul[1], cam_mul[2]);

        // Convert cam_mul from camera to XYZ
        const auto cam_mul_xyz = cam_xyz * cam_mul;
        std::cout << "cam_mul_xyz: " << cam_mul_xyz << ", CCT: " << XYZtoCorColorTemp(cam_mul_xyz) << std::endl;

        for (int c = 0; c < 3; c++) {
            as_shot_neutral[c] = 1 / cam_mul[c];
        }
        (*dng_metadata)[TIFFTAG_ASSHOTNEUTRAL] = as_shot_neutral;

        pre_mul = cam_mul;
    }

//    // Save the whitening transformation
//    const auto inv_cam_white = pre_mul;
//
//    gls::Matrix<3, 3> mCamMul = {
//        { pre_mul[0] / cam_mul[0], 0, 0 },
//        { 0, pre_mul[1] / cam_mul[1], 0 },
//        { 0, 0, pre_mul[2] / cam_mul[2] }
//    };
//    {
//        const gls::Vector<3> d65_white = { 0.95047, 1.0, 1.08883 };
//        const gls::Vector<3> d50_white = { 0.9642, 1.0000, 0.8249 };
//
//        std::cout << "XYZ D65 White -> sRGB: " << rgb_xyz * d65_white << std::endl;
//        std::cout << "sRGB White -> XYZ D65: " << xyz_rgb * gls::Vector({1, 1, 1}) << std::endl;
//        std::cout << "inverse(cam_xyz) * (1 / pre_mul): " << inverse(cam_xyz) * (1 / pre_mul) << std::endl;
//
//        auto cam_white = inverse(cam_xyz) * xyz_rgb * gls::Vector<3>({ 1, 1, 1 });
//        std::cout << "inverse(cam_xyz) * cam_white: " << cam_xyz * cam_white << ", CCT: " << XYZtoCorColorTemp(cam_xyz * cam_white) << std::endl;
//        std::cout << "xyz_rgb * gls::Vector<3>({ 1, 1, 1 }): " << xyz_rgb * gls::Vector<3>({ 1, 1, 1 }) << ", CCT: " << XYZtoCorColorTemp(xyz_rgb * gls::Vector<3>({ 1, 1, 1 })) << std::endl;
//
//        const auto wb_out = xyz_rgb * demosaicParameters->rgb_cam * mCamMul * gls::Vector({1, 1, 1});
//        std::cout << "wb_out: " << wb_out << ", CCT: " << XYZtoCorColorTemp(wb_out) << std::endl;
//
//        const auto no_wb_out = xyz_rgb * demosaicParameters->rgb_cam * (1 / inv_cam_white);
//        std::cout << "no_wb_out: " << wb_out << ", CCT: " << XYZtoCorColorTemp(no_wb_out) << std::endl;
//    }

    // Scale Input Image
    auto minmax = std::minmax_element(std::begin(pre_mul), std::end(pre_mul));
    for (int c = 0; c < 4; c++) {
        int pre_mul_idx = c == 3 ? 1 : c;
        (demosaicParameters->scale_mul)[c] = (pre_mul[pre_mul_idx] / *minmax.first) * 65535.0 / (demosaicParameters->white_level - demosaicParameters->black_level);
    }
    printf("scale_mul: %f, %f, %f, %f\n", (demosaicParameters->scale_mul)[0], (demosaicParameters->scale_mul)[1], (demosaicParameters->scale_mul)[2], (demosaicParameters->scale_mul)[3]);
}

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

float square(float x) {
    return x * x;
}

enum { red = 0, green = 1, blue = 2, green2 = 3 };

// Collect mean and variance of ColorChecker patches
void colorCheckerRawStats(const gls::image<gls::luma_pixel_16>& rawImage, float black_level, float white_level, BayerPattern bayerPattern, const gls::rectangle& gmb_position, bool rotate_180, std::array<RawPatchStats, 24>* stats) {
    gls::image<gls::luma_pixel_16> green_channel(rawImage.width/2, rawImage.height/2);

//    gls::image<gls::luma_pixel_16>* zapMama = (gls::image<gls::luma_pixel_16> *) &rawImage;

    std::cout << "colorCheckerRawStats rectangle: " << gmb_position.x << ", " << gmb_position.y << ", " << gmb_position.width << ", " << gmb_position.height << std::endl;

    int patch_width = gmb_position.width / 6;
    int patch_height = gmb_position.height / 4;

    auto offsets = bayerOffsets[bayerPattern];
    const gls::point r = offsets[red];
    const gls::point g = offsets[green];
    const gls::point b = offsets[blue];
    const gls::point g2 = offsets[green2];

    int patchIdx = 0;
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 6; col++, patchIdx++) {
            gls::rectangle patch = alignToQuad({
                gmb_position.x + col * patch_width + (int) (0.25 * patch_width),
                gmb_position.y + row * patch_height + (int) (0.25 * patch_height),
                (int) (0.5 * patch_width),
                (int) (0.5 * patch_height) });

            const int patchSamples = patch.width * patch.height / 4;

            float avgG1 = 0;
            float avgG2 = 0;
            float avgR = 0;
            float avgB = 0;

            for (int y = 0; y < patch.height; y += 2) {
                for (int x = 0; x < patch.width; x += 2) {
                    const int y_off = patch.y + y;
                    const int x_off = patch.x + x;
                    gls::rgba_pixel_fp32 p = {
                        (rawImage[y_off + r.y][x_off + r.x] - black_level) / white_level,
                        (rawImage[y_off + g.y][x_off + g.x] - black_level) / white_level,
                        (rawImage[y_off + b.y][x_off + b.x] - black_level) / white_level,
                        (rawImage[y_off + g2.y][x_off + g2.x] - black_level) / white_level
                    };

                    avgR += p[0];
                    avgG1 += p[1];
                    avgB += p[2];
                    avgG2 += p[3];
                }
            }

            avgR /= patchSamples;
            avgG1 /= patchSamples;
            avgB /= patchSamples;
            avgG2 /= patchSamples;

            float varR = 0;
            float varG1 = 0;
            float varB = 0;
            float varG2 = 0;

            for (int y = 0; y < patch.height; y += 2) {
                for (int x = 0; x < patch.width; x += 2) {
                    const int y_off = patch.y + y;
                    const int x_off = patch.x + x;
                    gls::rgba_pixel_fp32 p = {
                        (rawImage[y_off + r.y][x_off + r.x] - black_level) / white_level,
                        (rawImage[y_off + g.y][x_off + g.x] - black_level) / white_level,
                        (rawImage[y_off + b.y][x_off + b.x] - black_level) / white_level,
                        (rawImage[y_off + g2.y][x_off + g2.x] - black_level) / white_level
                    };

                    green_channel[(patch.y + y)/2][(patch.x + x)/2] = (0xffff / white_level) *  rawImage[y_off + g.y][x_off + g.x];

                    varR += square(p[0] - avgR);
                    varG1 += square(p[1] - avgG1);
                    varB += square(p[2] - avgB);
                    varG2 += square(p[3] - avgG2);

//                    (*zapMama)[y_off + r.y][x_off + r.x] = 0;
//                    (*zapMama)[y_off + g.y][x_off + g.x] = 0;
//                    (*zapMama)[y_off + b.y][x_off + b.x] = 0;
//                    (*zapMama)[y_off + g2.y][x_off + g2.x] = 0;
                }
            }

            varR /= patchSamples;
            varG1 /= patchSamples;
            varB /= patchSamples;
            varG2 /= patchSamples;

            (*stats)[patchIdx] = {{avgR, avgG1, avgB, avgG2}, {varR, varG1, varB, varG2}};
        }
    }

    if (rotate_180) {
        std::reverse(stats->begin(), stats->end());
    }

//    for (int patchIdx = 0; patchIdx < 24; patchIdx++) {
//        std::cout << std::setw(12) << std::setfill(' ') << GMBColorNames[patchIdx] << " - avg: {"
//                  << std::setprecision(2) << (*stats)[patchIdx].mean[0] << ", " << (*stats)[patchIdx].mean[1] << ", " << (*stats)[patchIdx].mean[2] << ", " << (*stats)[patchIdx].mean[3] << "}, var: {"
//                                          << (*stats)[patchIdx].variance[0] << ", " << (*stats)[patchIdx].variance[1] << ", " << (*stats)[patchIdx].variance[2] << ", " << (*stats)[patchIdx].variance[3] << "}" << std::endl;
//
//    }

    green_channel.write_png_file("/Users/fabio/green_channel.png", false);
}

// Collect mean and variance of ColorChecker patches
void colorCheckerStats(gls::image<gls::rgba_pixel_float>* image, const gls::rectangle& gmb_position, bool rotate_180, std::array<PatchStats, 24>* stats) {
    // std::cout << "colorCheckerStats rectangle: " << gmb_position.x << ", " << gmb_position.y << ", " << gmb_position.width << ", " << gmb_position.height << std::endl;

    int patch_width = gmb_position.width / 6;
    int patch_height = gmb_position.height / 4;

    int patchIdx = 0;
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 6; col++, patchIdx++) {
            gls::rectangle patch = {
                gmb_position.x + col * patch_width + (int) (0.25 * patch_width),
                gmb_position.y + row * patch_height + (int) (0.25 * patch_height),
                (int) (0.5 * patch_width),
                (int) (0.5 * patch_height) };

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

            (*stats)[patchIdx] = {{avgY, avgCb, avgCr}, {varY, varCb, varCr}};
        }
    }

    if (rotate_180) {
        std::reverse(stats->begin(), stats->end());
    }

//    for (int patchIdx = 0; patchIdx < 24; patchIdx++) {
//        std::cout << std::setw(12) << std::setfill(' ') << GMBColorNames[patchIdx] << " - avg: {"
//                  << std::setprecision(2) << (*stats)[patchIdx].mean[0] << ", " << (*stats)[patchIdx].mean[1] << ", " << (*stats)[patchIdx].mean[2] << "}, var: {"
//                                          << (*stats)[patchIdx].variance[0] << ", " << (*stats)[patchIdx].variance[1] << ", " << (*stats)[patchIdx].variance[2] << "}" << std::endl;
//    }
}

// Slope Regression of a set of points
template <size_t N>
std::pair<float, float> linear_regression(const gls::Vector<N>& x, const gls::Vector<N>& y, float *errorSquare = nullptr) {
    const auto s_x  = std::accumulate(x.begin(), x.end(), 0.0);
    const auto s_y  = std::accumulate(y.begin(), y.end(), 0.0);
    const auto s_xx = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
    const auto s_xy = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
    const auto b    = (N * s_xy - s_x * s_y) / (N * s_xx - s_x * s_x);
    const auto a    = (s_y - b * s_x) / N;

    if (errorSquare) {
        *errorSquare = 0;
        for (int i = 0; i < x.size(); i++) {
            float p = a + b * x[i];
            float diff = p - y[i];
            *errorSquare += diff * diff;
        }
    }

    return { a, b };
}

// Estimate the Sensor's Noise Level Function (NLF: variance vs intensity), which is linear going through zero
gls::Vector<3> estimateNlfParameters(gls::image<gls::rgba_pixel_float>* image, const gls::rectangle& gmb_position, bool rotate_180) {
    std::array<PatchStats, 24> stats;
    colorCheckerStats(image, gmb_position, rotate_180, &stats);

    gls::Vector<6> y_intensity = {
        stats[Black].mean[0],
        stats[Neutral_3_5].mean[0],
        stats[Neutral_5].mean[0],
        stats[Neutral_6_5].mean[0],
        stats[Neutral_8].mean[0],
        stats[White].mean[0]
    };

    gls::Vector<6> y_variance = {
        stats[Black].variance[0],
        stats[Neutral_3_5].variance[0],
        stats[Neutral_5].variance[0],
        stats[Neutral_6_5].variance[0],
        stats[Neutral_8].variance[0],
        stats[White].variance[0]
    };

    gls::Vector<6> cb_variance = {
        stats[Black].variance[1],
        stats[Neutral_3_5].variance[1],
        stats[Neutral_5].variance[1],
        stats[Neutral_6_5].variance[1],
        stats[Neutral_8].variance[1],
        stats[White].variance[1]
    };

    gls::Vector<6> cr_variance = {
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

    float y_err2;
    auto nlf_y = linear_regression(y_intensity, y_variance, &y_err2);
    auto nlf_cb = std::accumulate(cb_variance.begin(), cb_variance.end(), 0.0f) / cb_variance.size();
    auto nlf_cr = std::accumulate(cr_variance.begin(), cr_variance.end(), 0.0f) / cr_variance.size();

//    std::cout << std::setprecision(4) << std::setw(4)
//              << "nlf_y: " << nlf_y.first << ":" << nlf_y.second << " (" << y_err2 << ")"
//              << ", nlf_cb: " << nlf_cb << ", nlf_cr: " << nlf_cr << std::endl;

    // NFL for Y passes by 0, just use the slope, NFL for Cb and and Cr is mostly flat, just return the average
    return {nlf_y.second, nlf_cb, nlf_cr};
}

gls::Vector<4> estimateRawParameters(const gls::image<gls::luma_pixel_16>& rawImage, gls::Matrix<3, 3>* cam_xyz, gls::Vector<3>* pre_mul,
                                     float black_level, float white_level, BayerPattern bayerPattern, const gls::rectangle& gmb_position, bool rotate_180) {
    std::array<RawPatchStats, 24> rawStats;
    colorCheckerRawStats(rawImage, black_level, white_level, bayerPattern, gmb_position, rotate_180, &rawStats);

    gls::Vector<6> red_intensity = {
        rawStats[Black].mean[0],
        rawStats[Neutral_3_5].mean[0],
        rawStats[Neutral_5].mean[0],
        rawStats[Neutral_6_5].mean[0],
        rawStats[Neutral_8].mean[0],
        rawStats[White].mean[0]
    };

    gls::Vector<6> green_intensity = {
        rawStats[Black].mean[1],
        rawStats[Neutral_3_5].mean[1],
        rawStats[Neutral_5].mean[1],
        rawStats[Neutral_6_5].mean[1],
        rawStats[Neutral_8].mean[1],
        rawStats[White].mean[1]
    };

    gls::Vector<6> blue_intensity = {
        rawStats[Black].mean[2],
        rawStats[Neutral_3_5].mean[2],
        rawStats[Neutral_5].mean[2],
        rawStats[Neutral_6_5].mean[2],
        rawStats[Neutral_8].mean[2],
        rawStats[White].mean[2]
    };

    gls::Vector<6> green2_intensity = {
        rawStats[Black].mean[3],
        rawStats[Neutral_3_5].mean[3],
        rawStats[Neutral_5].mean[3],
        rawStats[Neutral_6_5].mean[3],
        rawStats[Neutral_8].mean[3],
        rawStats[White].mean[3]
    };

    gls::Vector<6> red_variance = {
        rawStats[Black].variance[0],
        rawStats[Neutral_3_5].variance[0],
        rawStats[Neutral_5].variance[0],
        rawStats[Neutral_6_5].variance[0],
        rawStats[Neutral_8].variance[0],
        rawStats[White].variance[0]
    };

    gls::Vector<6> green_variance = {
        rawStats[Black].variance[1],
        rawStats[Neutral_3_5].variance[1],
        rawStats[Neutral_5].variance[1],
        rawStats[Neutral_6_5].variance[1],
        rawStats[Neutral_8].variance[1],
        rawStats[White].variance[1]
    };

    gls::Vector<6> blue_variance = {
        rawStats[Black].variance[2],
        rawStats[Neutral_3_5].variance[2],
        rawStats[Neutral_5].variance[2],
        rawStats[Neutral_6_5].variance[2],
        rawStats[Neutral_8].variance[2],
        rawStats[White].variance[2]
    };

    gls::Vector<6> green2_variance = {
        rawStats[Black].variance[3],
        rawStats[Neutral_3_5].variance[3],
        rawStats[Neutral_5].variance[3],
        rawStats[Neutral_6_5].variance[3],
        rawStats[Neutral_8].variance[3],
        rawStats[White].variance[3]
    };

    // Derive color matrix
    matrixFromColorChecker(rawStats, cam_xyz, pre_mul);

    float r_err2, g_err2, b_err2, g2_err2;
    auto nlf_r = linear_regression(red_intensity, red_variance, &r_err2);
    auto nlf_g = linear_regression(green_intensity, green_variance, &g_err2);
    auto nlf_b = linear_regression(blue_intensity, blue_variance, &b_err2);
    auto nlf_g2 = linear_regression(green2_intensity, green2_variance, &g2_err2);

//    std::cout << std::setprecision(4) << std::setw(4)
//              << "raw nlf_r: " << nlf_r.first << ":" << nlf_r.second << " (" << r_err2 << "), "
//              << "raw nlf_g: " << nlf_g.first << ":" << nlf_g.second << " (" << g_err2 << "), "
//              << "raw nlf_b: " << nlf_b.first << ":" << nlf_b.second << " (" << b_err2 << "), "
//              << "raw nlf_g2: " << nlf_g2.first << ":" << nlf_g2.second << " (" << g2_err2 << ")" << std::endl;

    std::cout << std::setprecision(4) << std::scientific << "raw nlf (r g b g2): "
              << nlf_r.second << ", "
              << nlf_g.second << ", "
              << nlf_b.second << ", "
              << nlf_g2.second << std::endl;

    return { nlf_r.second, nlf_g.second, nlf_b.second, nlf_g2.second };
}

gls::Vector<3> extractNlfFromColorChecker(gls::image<gls::rgba_pixel_float>* yCbCrImage, const gls::rectangle gmb_position, bool rotate_180, int scale) {
    const gls::rectangle position = {
        (int) round(gmb_position.x / (float) scale),
        (int) round(gmb_position.y / (float) scale),
        (int) round(gmb_position.width / (float) scale),
        (int) round(gmb_position.height / (float) scale)
    };
    gls::Vector<3> nlf_parameters = estimateNlfParameters(yCbCrImage, position, rotate_180);
    std::cout << "Scale " << scale << " nlf parameters: " << std::setprecision(4) << std::scientific << nlf_parameters[0] << ", " << nlf_parameters[1] << ", " << nlf_parameters[2] << std::endl;

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

struct WhiteBalanceStats {
    gls::Vector<3> wbGain;
    float diffAverage;
    int whitePixelsCount;
};

WhiteBalanceStats autoWhiteBalanceKernel(const gls::image<gls::luma_pixel_16>& rawImage, const gls::Matrix<3, 3>& rgb_ycbcr,
                                         float white, float black, BayerPattern bayerPattern, float highlightsFraction) {
    const auto& offsets = bayerOffsets[bayerPattern];
    const auto& Or  = offsets[0];
    const auto& Og1 = offsets[1];
    const auto& Ob  = offsets[2];
    const auto& Og2 = offsets[3];

    // Compute the average ycbcr values
    gls::Vector<3> ycbcrAverage = { 0, 0, 0 };
    gls::image<gls::rgb_pixel_fp32> ycbcrImage(rawImage.width / 2, rawImage.height / 2);
    for (int y = 0; y < rawImage.height; y += 2) {
        for (int x = 0; x < rawImage.width; x += 2) {
            const auto rgb = gls::Vector<3> {
                (float) rawImage[y + Or.y][x + Or.x],
                ((float) rawImage[y + Og1.y][x + Og1.x] +
                 (float) rawImage[y + Og2.y][x + Og2.x]) / 2,
                (float) rawImage[y + Ob.y][x + Ob.x]
            };
            const auto ycbcr = ((rgb - black) / white) * rgb_ycbcr;
            ycbcrImage[y / 2][x / 2] = ycbcr;
            ycbcrAverage += ycbcr;
        }
    }
    ycbcrAverage /= ycbcrImage.width * ycbcrImage.height;

    // Compute the ycbcr average absolute differences
    gls::Vector<3> ycbcrDiffAverage = { 0, 0, 0 };
    for (int y = 0; y < ycbcrImage.height; y++) {
        for (int x = 0; x < ycbcrImage.width; x++) {
            ycbcrDiffAverage += abs(gls::Vector<3>(ycbcrImage[y][x].v) - ycbcrAverage);
        }
    }
    ycbcrDiffAverage /= ycbcrImage.width * ycbcrImage.height;

    std::array<std::pair<gls::Vector<3>, int>, 128> rgbWhiteAverageHist = { };

    int whitePixelsCount = 0;
    float YMax = 0;
    for (int y = 0; y < ycbcrImage.height; y++) {
        for (int x = 0; x < ycbcrImage.width; x++) {
            const auto& p = ycbcrImage[y][x];

            // Near white region pixels
            if (fabs(p[1] - (ycbcrAverage[1] + copysign(ycbcrDiffAverage[1], ycbcrAverage[1]))) < 1.5 * ycbcrDiffAverage[1] &&
                fabs(p[2] - (1.5 * ycbcrAverage[2] + copysign(ycbcrDiffAverage[2], ycbcrAverage[2]))) < 1.5 * ycbcrDiffAverage[2]) {
                const auto rgb = gls::Vector<3> {
                    (float) rawImage[2 * y + Or.y][2 * x + Or.x],
                    ((float) rawImage[2 * y + Og1.y][2 * x + Og1.x] +
                     (float) rawImage[2 * y + Og2.y][2 * x + Og2.x]) / 2,
                    (float) rawImage[2 * y + Ob.y][2 * x + Ob.x]
                };

                float Y = p[0];
                if (Y > YMax) {
                    YMax = Y;
                }

                size_t histEntry = std::clamp((size_t) round((rgbWhiteAverageHist.size() - 1) * Y), 0UL, rgbWhiteAverageHist.size() - 1);
                auto& entry = rgbWhiteAverageHist[histEntry];
                entry.first += rgb;
                entry.second++;

                whitePixelsCount++;
            }
        }
    }

    int histMaxEntry = (int) std::clamp((size_t) round((rgbWhiteAverageHist.size() - 1) * YMax), 0UL, rgbWhiteAverageHist.size() - 1);

    if (histMaxEntry == 0) {
        return {
            .wbGain = { 1, 1, 1 },
            .diffAverage = 0,
            .whitePixelsCount = 1
        };
    }

    int white90PixelsCount = 0;
    gls::Vector<3> rgbWhite90Average = { 0, 0, 0 };

    for (int i = histMaxEntry; i >= 0; i--) {
        auto& entry = rgbWhiteAverageHist[i];
        rgbWhite90Average += entry.first;
        white90PixelsCount += entry.second;

        if (white90PixelsCount > highlightsFraction * whitePixelsCount) {
            break;
        }
    }
    rgbWhite90Average /= white90PixelsCount;

    std::cout << "histMaxEntry: " << histMaxEntry << ", white90PixelsCount: " << white90PixelsCount << ", whitePixelsCount: " << whitePixelsCount << std::endl;

    auto wbGain = YMax / rgbWhite90Average;

    return {
        .wbGain = wbGain / wbGain[1],
        .diffAverage = std::max(ycbcrDiffAverage[1], ycbcrDiffAverage[2]),
        .whitePixelsCount = white90PixelsCount
    };
}

gls::Vector<3> autoWhiteBalance(const gls::image<gls::luma_pixel_16>& rawImage, const gls::Matrix<3, 3>& rgb_ycbcr,
                                float white, float black, BayerPattern bayerPattern) {
    const int hTiles = 8;
    const int vTiles = 6;

    const int tileWidth = rawImage.width / hTiles;
    const int tileHeight = rawImage.height / vTiles;

    ThreadPool threadPool(8);

    gls::Vector<3> wbGain = { 0, 0, 0 };
    int wbCount = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    std::array<std::array<std::future<WhiteBalanceStats>, hTiles>, vTiles> wbGains;
    for (int y = 0; y < vTiles; y++) {
        for (int x = 0; x < hTiles; x++) {
            wbGains[y][x] = threadPool.enqueue([x, y, tileWidth, tileHeight, rgb_ycbcr, white, black, bayerPattern, &rawImage]() -> WhiteBalanceStats {
                const auto rawTile = gls::image<gls::luma_pixel_16>(rawImage, x * tileWidth, y * tileHeight, tileWidth, tileHeight);
                return autoWhiteBalanceKernel(rawTile, rgb_ycbcr, white, black, bayerPattern, /*highlightsFraction=*/ 0.1);
            });
        }
    }
    for (int y = 0; y < vTiles; y++) {
        for (int x = 0; x < hTiles; x++) {
            const auto wb = wbGains[y][x].get();
            std::cout << "wbGain(" << x << ":" << y << "): " << wb.wbGain << ", diffAverage: " << wb.diffAverage << std::endl;

            if (wb.diffAverage > 0.001) {
                wbGain += wb.wbGain;
                wbCount++;
            }
        }
    }
    wbGain /= wbCount;
    wbGain /= wbGain[0];

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    std::cout << "wbGain: " << wbGain << ", execution time: " << elapsed_time_ms << "ms." << std::endl;

    return wbGain;
}
