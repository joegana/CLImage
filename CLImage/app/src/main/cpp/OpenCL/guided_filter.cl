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

kernel void boxBlurX(read_only image2d_t inputImage,
                     int filterSize,
                     write_only image2d_t outputImage) {
    const int2 imageCoordinate = (int2)(get_global_id(0), get_global_id(1));

    float4 sum = 0;
    for (int x = -filterSize / 2; x <= filterSize / 2; x++) {
        sum += read_imagef(inputImage, imageCoordinate + (int2)(x, 0));
    }
    sum /= filterSize;

    write_imagef(outputImage, (int2)(imageCoordinate.y, imageCoordinate.x), sum);
}

kernel void covMatrixProducts(read_only image2d_t inputImage,
                              write_only image2d_t products1,
                              write_only image2d_t products2) {
    const int2 imageCoordinate = (int2)(get_global_id(0), get_global_id(1));

    float3 sample = read_imagef(inputImage, imageCoordinate).xyz;

    write_imagef(products1, imageCoordinate, (float4)(sample.x * sample.x,
                                                      sample.x * sample.y,
                                                      sample.x * sample.z,
                                                      sample.y * sample.y));
    write_imagef(products2, imageCoordinate, (float4)(sample.y * sample.z,
                                                      sample.z * sample.z,
                                                      0, 0));
}

kernel void invSigma(read_only image2d_t meanImage,
                     read_only image2d_t products1Image,
                     read_only image2d_t products2Image,
                     float3 eps,
                     write_only image2d_t invSigma1Image,
                     write_only image2d_t invSigma2Image) {
    const int2 imageCoordinate = (int2)(get_global_id(0), get_global_id(1));

    float3 mean_I = read_imagef(meanImage, imageCoordinate).xyz;
    float4 products1 = read_imagef(products1Image, imageCoordinate);
    float2 products2 = read_imagef(products2Image, imageCoordinate).xy;

    float var_I_rr = products1.x - mean_I.x * mean_I.x + eps.x * mean_I.x;
    float var_I_rg = products1.y - mean_I.x * mean_I.y;
    float var_I_rb = products1.z - mean_I.x * mean_I.z;
    float var_I_gg = products1.w - mean_I.y * mean_I.y + eps.y;
    float var_I_gb = products2.x - mean_I.y * mean_I.z;
    float var_I_bb = products2.y - mean_I.z * mean_I.z + eps.z;

    float invrr = var_I_gg * var_I_bb - var_I_gb * var_I_gb;
    float invrg = var_I_gb * var_I_rb - var_I_rg * var_I_bb;
    float invrb = var_I_rg * var_I_gb - var_I_gg * var_I_rb;
    float invgg = var_I_rr * var_I_bb - var_I_rb * var_I_rb;
    float invgb = var_I_rb * var_I_rg - var_I_rr * var_I_gb;
    float invbb = var_I_rr * var_I_gg - var_I_rg * var_I_rg;

    float covDet = invrr * var_I_rr + invrg * var_I_rg + invrb * var_I_rb;

    invrr /= covDet;
    invrg /= covDet;
    invrb /= covDet;
    invgg /= covDet;
    invgb /= covDet;
    invbb /= covDet;

    write_imagef(invSigma1Image, imageCoordinate, (float4)(invrr, invrg, invrb, invgg));
    write_imagef(invSigma2Image, imageCoordinate, (float4)(invgb, invbb, 0, 0));
}

kernel void meanIpProducts(read_only image2d_t inputImage,
                           write_only image2d_t mean_Ip_r,
                           write_only image2d_t mean_Ip_g,
                           write_only image2d_t mean_Ip_b) {
    const int2 imageCoordinate = (int2)(get_global_id(0), get_global_id(1));

    float3 p = read_imagef(inputImage, imageCoordinate).xyz;
    write_imagef(mean_Ip_r, imageCoordinate, (float4)(p.x * p, 0));
    write_imagef(mean_Ip_g, imageCoordinate, (float4)(p.y * p, 0));
    write_imagef(mean_Ip_b, imageCoordinate, (float4)(p.z * p, 0));
}

kernel void computeAb(read_only image2d_t mean_pImage,
                      read_only image2d_t mean_Ip_rImage,
                      read_only image2d_t mean_Ip_gImage,
                      read_only image2d_t mean_Ip_bImage,
                      read_only image2d_t invSigma1Image,
                      read_only image2d_t invSigma2Image,
                      write_only image2d_t ab_rImage,
                      write_only image2d_t ab_gImage,
                      write_only image2d_t ab_bImage) {
    const int2 imageCoordinate = (int2)(get_global_id(0), get_global_id(1));

    float3 mean_p = read_imagef(mean_pImage, imageCoordinate).xyz;
    float3 mean_I = mean_p;

    float3 mean_Ip_r = read_imagef(mean_Ip_rImage, imageCoordinate).xyz;
    float3 mean_Ip_g = read_imagef(mean_Ip_gImage, imageCoordinate).xyz;
    float3 mean_Ip_b = read_imagef(mean_Ip_bImage, imageCoordinate).xyz;

    float3 cov_Ip_r = mean_Ip_r - mean_I.x * mean_p;
    float3 cov_Ip_g = mean_Ip_g - mean_I.y * mean_p;
    float3 cov_Ip_b = mean_Ip_b - mean_I.z * mean_p;

    float4 invSigma1 = read_imagef(invSigma1Image, imageCoordinate);
    float2 invSigma2 = read_imagef(invSigma2Image, imageCoordinate).xy;
    float invrr = invSigma1.x;
    float invrg = invSigma1.y;
    float invrb = invSigma1.z;
    float invgg = invSigma1.w;
    float invgb = invSigma2.x;
    float invbb = invSigma2.y;

    float3 a_r = invrr * cov_Ip_r + invrg * cov_Ip_g + invrb * cov_Ip_b;
    float3 a_g = invrg * cov_Ip_r + invgg * cov_Ip_g + invgb * cov_Ip_b;
    float3 a_b = invrb * cov_Ip_r + invgb * cov_Ip_g + invbb * cov_Ip_b;

    float3 b = mean_p - a_r * mean_I.x - a_g * mean_I.y - a_b * mean_I.z; // Eqn. (15) in the paper;

    write_imagef(ab_rImage, imageCoordinate, (float4)(a_r, b.x));
    write_imagef(ab_gImage, imageCoordinate, (float4)(a_g, b.y));
    write_imagef(ab_bImage, imageCoordinate, (float4)(a_b, b.z));
}

kernel void computeResult(read_only image2d_t inputImage,
                          read_only image2d_t ab_rImage,
                          read_only image2d_t ab_gImage,
                          read_only image2d_t ab_bImage,
                          write_only image2d_t resultImage) {
    const int2 imageCoordinate = (int2)(get_global_id(0), get_global_id(1));

    float3 input = read_imagef(inputImage, imageCoordinate).xyz;
    float4 ab_r = read_imagef(ab_rImage, imageCoordinate);
    float4 ab_g = read_imagef(ab_gImage, imageCoordinate);
    float4 ab_b = read_imagef(ab_bImage, imageCoordinate);

    float3 result = ab_r.xyz * input.x + ab_g.xyz * input.y + ab_b.xyz * input.z + (float3)(ab_r.w, ab_g.w, ab_b.w);

    write_imagef(resultImage, imageCoordinate, (float4)(result, 0));
}
