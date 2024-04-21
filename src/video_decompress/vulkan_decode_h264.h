/**
 * @file   video_decompress/vulkan_decode_h264.h
 * @author Ondřej Richtr     <524885@mail.muni.cz>
 */

// This file is a (modified) portion taken from h264 header from WickedEngine project:
// https://github.com/turanszkij/WickedEngine/blob/master/WickedEngine/Utility/h264.h
// and from utils/h264_stream.c file (nal_to_rbsp function)

//TODO useless check?
#ifndef VULKAN_DECODE_H264_H
#define VULKAN_DECODE_H264_H

#include "utils/bs.h"

//constants borrowed from NVidia example
#define MAX_VPS_IDS 16
#define MAX_SPS_IDS 32
#define MAX_PPS_IDS 256

typedef struct
{
    int profile_idc;
    int constraint_set0_flag;
    int constraint_set1_flag;
    int constraint_set2_flag;
    int constraint_set3_flag;
    int constraint_set4_flag;
    int constraint_set5_flag;
    int reserved_zero_2bits;
    int level_idc;
    int seq_parameter_set_id;
    int chroma_format_idc;
    int separate_colour_plane_flag;
    int bit_depth_luma_minus8;
    int bit_depth_chroma_minus8;
    int qpprime_y_zero_transform_bypass_flag;
    int seq_scaling_matrix_present_flag;
    int seq_scaling_list_present_flag[8];
    int ScalingList4x4[6][16];
    int UseDefaultScalingMatrix4x4Flag[6];
    int ScalingList8x8[2][64];
    int UseDefaultScalingMatrix8x8Flag[2];
    int log2_max_frame_num_minus4;
    int pic_order_cnt_type;
    int log2_max_pic_order_cnt_lsb_minus4;
    int delta_pic_order_always_zero_flag;
    int offset_for_non_ref_pic;
    int offset_for_top_to_bottom_field;
    int num_ref_frames_in_pic_order_cnt_cycle;
    int offset_for_ref_frame[256];
    int num_ref_frames;
    int gaps_in_frame_num_value_allowed_flag;
    int pic_width_in_mbs_minus1;
    int pic_height_in_map_units_minus1;
    int frame_mbs_only_flag;
    int mb_adaptive_frame_field_flag;
    int direct_8x8_inference_flag;
    int frame_cropping_flag;
    int frame_crop_left_offset;
    int frame_crop_right_offset;
    int frame_crop_top_offset;
    int frame_crop_bottom_offset;
    int vui_parameters_present_flag;

    struct
    {
        int aspect_ratio_info_present_flag;
        int aspect_ratio_idc;
        int sar_width;
        int sar_height;
        int overscan_info_present_flag;
        int overscan_appropriate_flag;
        int video_signal_type_present_flag;
        int video_format;
        int video_full_range_flag;
        int colour_description_present_flag;
        int colour_primaries;
        int transfer_characteristics;
        int matrix_coefficients;
        int chroma_loc_info_present_flag;
        int chroma_sample_loc_type_top_field;
        int chroma_sample_loc_type_bottom_field;
        int timing_info_present_flag;
        int num_units_in_tick;
        int time_scale;
        int fixed_frame_rate_flag;
        int nal_hrd_parameters_present_flag;
        int vcl_hrd_parameters_present_flag;
        int low_delay_hrd_flag;
        int pic_struct_present_flag;
        int bitstream_restriction_flag;
        int motion_vectors_over_pic_boundaries_flag;
        int max_bytes_per_pic_denom;
        int max_bits_per_mb_denom;
        int log2_max_mv_length_horizontal;
        int log2_max_mv_length_vertical;
        int num_reorder_frames;
        int max_dec_frame_buffering;
    } vui;

    struct
    {
        int cpb_cnt_minus1;
        int bit_rate_scale;
        int cpb_size_scale;
        int bit_rate_value_minus1[32];
        int cpb_size_value_minus1[32];
        int cbr_flag[32];
        int initial_cpb_removal_delay_length_minus1;
        int cpb_removal_delay_length_minus1;
        int dpb_output_delay_length_minus1;
        int time_offset_length;
    } hrd;
} sps_t;

typedef struct
{
    int pic_parameter_set_id; //v
    int seq_parameter_set_id; //v
    int entropy_coding_mode_flag; //v
    int pic_order_present_flag;
    int num_slice_groups_minus1;
    int slice_group_map_type;
    int run_length_minus1[8];
    int top_left[8];
    int bottom_right[8];
    int slice_group_change_direction_flag;
    int slice_group_change_rate_minus1;
    int pic_size_in_map_units_minus1;
    int slice_group_id[256];
    int num_ref_idx_l0_active_minus1; //v
    int num_ref_idx_l1_active_minus1; //v
    int weighted_pred_flag; //v
    int weighted_bipred_idc; //v
    int pic_init_qp_minus26; //v
    int pic_init_qs_minus26; //v
    int chroma_qp_index_offset; //v
    int deblocking_filter_control_present_flag; //v
    int constrained_intra_pred_flag; //v
    int redundant_pic_cnt_present_flag; //v

    int _more_rbsp_data_present;

    int transform_8x8_mode_flag; //v
    int pic_scaling_matrix_present_flag; //v
    int pic_scaling_list_present_flag[8];
    int ScalingList4x4[6][16];
    int UseDefaultScalingMatrix4x4Flag[6];
    int ScalingList8x8[2][64];
    int UseDefaultScalingMatrix8x8Flag[2];
    int second_chroma_qp_index_offset; //v
} pps_t;

enum sh_slice_type_t
{
    SH_SLICE_TYPE_P = 0,
    SH_SLICE_TYPE_B = 1,
    SH_SLICE_TYPE_I = 2,
    SH_SLICE_TYPE_SP = 3,
    SH_SLICE_TYPE_SI = 4,

    // The *_ONLY slice types indicate that all other slices in that picture are of the same type
    SH_SLICE_TYPE_P_ONLY = 5,
    SH_SLICE_TYPE_B_ONLY = 6,
    SH_SLICE_TYPE_I_ONLY = 7,
    SH_SLICE_TYPE_SP_ONLY = 8,
    SH_SLICE_TYPE_SI_ONLY = 9,
};

typedef struct
{
    int first_mb_in_slice;
    int slice_type;
    int pic_parameter_set_id;
    int frame_num;
    int field_pic_flag;
    int bottom_field_flag;
    int idr_pic_id;
    int pic_order_cnt_lsb;
    int delta_pic_order_cnt_bottom;
    int delta_pic_order_cnt[2];
    int redundant_pic_cnt;
    int direct_spatial_mv_pred_flag;
    int num_ref_idx_active_override_flag;
    int num_ref_idx_l0_active_minus1;
    int num_ref_idx_l1_active_minus1;
    int cabac_init_idc;
    int slice_qp_delta;
    int sp_for_switch_flag;
    int slice_qs_delta;
    int disable_deblocking_filter_idc;
    int slice_alpha_c0_offset_div2;
    int slice_beta_offset_div2;
    int slice_group_change_cycle;

    struct
    {
        int luma_log2_weight_denom;
        int chroma_log2_weight_denom;
        int luma_weight_l0_flag[64];
        int luma_weight_l0[64];
        int luma_offset_l0[64];
        int chroma_weight_l0_flag[64];
        int chroma_weight_l0[64][2];
        int chroma_offset_l0[64][2];
        int luma_weight_l1_flag[64];
        int luma_weight_l1[64];
        int luma_offset_l1[64];
        int chroma_weight_l1_flag[64];
        int chroma_weight_l1[64][2];
        int chroma_offset_l1[64][2];
    } pwt; // predictive weight table

    struct
    {
        int ref_pic_list_reordering_flag_l0;
        struct
        {
            int reordering_of_pic_nums_idc[64];
            int abs_diff_pic_num_minus1[64];
            int long_term_pic_num[64];
        } reorder_l0;
        int ref_pic_list_reordering_flag_l1;
        struct
        {
            int reordering_of_pic_nums_idc[64];
            int abs_diff_pic_num_minus1[64];
            int long_term_pic_num[64];
        } reorder_l1;
    } rplr; // ref pic list reorder

    struct
    {
        int no_output_of_prior_pics_flag;
        int long_term_reference_flag;
        int adaptive_ref_pic_marking_mode_flag;
        int memory_management_control_operation[64];
        int difference_of_pic_nums_minus1[64];
        int long_term_pic_num[64];
        int long_term_frame_idx[64];
        int max_long_term_frame_idx_plus1[64];
    } drpm; // decoded ref pic marking
} slice_header_t;

static void print_pps(const pps_t *pps) //DEBUG
{
	printf("pic_parameter_set_id: %d, seq_parameter_set_id: %d, num_slice_groups_minus1 %d, num_ref_idx_l0_active_minus1 %d, num_ref_idx_l1_active_minus1 %d\n",
			pps->pic_parameter_set_id, pps->seq_parameter_set_id, pps->num_slice_groups_minus1, pps->num_ref_idx_l0_active_minus1, pps->num_ref_idx_l1_active_minus1);
}

static void print_sps(sps_t *sps) //DEBUG
{
    printf("======= SPS =======\n");
    printf(" profile_idc : %d \n", sps->profile_idc );
    printf(" constraint_set0_flag : %d \n", sps->constraint_set0_flag );
    printf(" constraint_set1_flag : %d \n", sps->constraint_set1_flag );
    printf(" constraint_set2_flag : %d \n", sps->constraint_set2_flag );
    printf(" constraint_set3_flag : %d \n", sps->constraint_set3_flag );
    printf(" constraint_set4_flag : %d \n", sps->constraint_set4_flag );
    printf(" constraint_set5_flag : %d \n", sps->constraint_set5_flag );
    printf(" reserved_zero_2bits : %d \n", sps->reserved_zero_2bits );
    printf(" level_idc : %d \n", sps->level_idc );
    printf(" seq_parameter_set_id : %d \n", sps->seq_parameter_set_id );
    printf(" chroma_format_idc : %d \n", sps->chroma_format_idc );
    //printf(" residual_colour_transform_flag : %d \n", sps->residual_colour_transform_flag );
    printf(" bit_depth_luma_minus8 : %d \n", sps->bit_depth_luma_minus8 );
    printf(" bit_depth_chroma_minus8 : %d \n", sps->bit_depth_chroma_minus8 );
    printf(" qpprime_y_zero_transform_bypass_flag : %d \n", sps->qpprime_y_zero_transform_bypass_flag );
    printf(" seq_scaling_matrix_present_flag : %d \n", sps->seq_scaling_matrix_present_flag );
    //  int seq_scaling_list_present_flag[8];
    //  void* ScalingList4x4[6];
    //  int UseDefaultScalingMatrix4x4Flag[6];
    //  void* ScalingList8x8[2];
    //  int UseDefaultScalingMatrix8x8Flag[2];
    printf(" log2_max_frame_num_minus4 : %d \n", sps->log2_max_frame_num_minus4 );
    printf(" pic_order_cnt_type : %d \n", sps->pic_order_cnt_type );
      printf("   log2_max_pic_order_cnt_lsb_minus4 : %d \n", sps->log2_max_pic_order_cnt_lsb_minus4 );
      printf("   delta_pic_order_always_zero_flag : %d \n", sps->delta_pic_order_always_zero_flag );
      printf("   offset_for_non_ref_pic : %d \n", sps->offset_for_non_ref_pic );
      printf("   offset_for_top_to_bottom_field : %d \n", sps->offset_for_top_to_bottom_field );
      printf("   num_ref_frames_in_pic_order_cnt_cycle : %d \n", sps->num_ref_frames_in_pic_order_cnt_cycle );
    //  int offset_for_ref_frame[256];
    printf(" num_ref_frames : %d \n", sps->num_ref_frames );
    printf(" gaps_in_frame_num_value_allowed_flag : %d \n", sps->gaps_in_frame_num_value_allowed_flag );
    printf(" pic_width_in_mbs_minus1 : %d \n", sps->pic_width_in_mbs_minus1 );
    printf(" pic_height_in_map_units_minus1 : %d \n", sps->pic_height_in_map_units_minus1 );
    printf(" frame_mbs_only_flag : %d \n", sps->frame_mbs_only_flag );
    printf(" mb_adaptive_frame_field_flag : %d \n", sps->mb_adaptive_frame_field_flag );
    printf(" direct_8x8_inference_flag : %d \n", sps->direct_8x8_inference_flag );
    printf(" frame_cropping_flag : %d \n", sps->frame_cropping_flag );
      printf("   frame_crop_left_offset : %d \n", sps->frame_crop_left_offset );
      printf("   frame_crop_right_offset : %d \n", sps->frame_crop_right_offset );
      printf("   frame_crop_top_offset : %d \n", sps->frame_crop_top_offset );
      printf("   frame_crop_bottom_offset : %d \n", sps->frame_crop_bottom_offset );
    printf(" vui_parameters_present_flag : %d \n", sps->vui_parameters_present_flag );

    printf("=== VUI ===\n");
    printf(" aspect_ratio_info_present_flag : %d \n", sps->vui.aspect_ratio_info_present_flag );
      printf("   aspect_ratio_idc : %d \n", sps->vui.aspect_ratio_idc );
        printf("     sar_width : %d \n", sps->vui.sar_width );
        printf("     sar_height : %d \n", sps->vui.sar_height );
    printf(" overscan_info_present_flag : %d \n", sps->vui.overscan_info_present_flag );
      printf("   overscan_appropriate_flag : %d \n", sps->vui.overscan_appropriate_flag );
    printf(" video_signal_type_present_flag : %d \n", sps->vui.video_signal_type_present_flag );
      printf("   video_format : %d \n", sps->vui.video_format );
      printf("   video_full_range_flag : %d \n", sps->vui.video_full_range_flag );
      printf("   colour_description_present_flag : %d \n", sps->vui.colour_description_present_flag );
        printf("     colour_primaries : %d \n", sps->vui.colour_primaries );
        printf("   transfer_characteristics : %d \n", sps->vui.transfer_characteristics );
        printf("   matrix_coefficients : %d \n", sps->vui.matrix_coefficients );
    printf(" chroma_loc_info_present_flag : %d \n", sps->vui.chroma_loc_info_present_flag );
      printf("   chroma_sample_loc_type_top_field : %d \n", sps->vui.chroma_sample_loc_type_top_field );
      printf("   chroma_sample_loc_type_bottom_field : %d \n", sps->vui.chroma_sample_loc_type_bottom_field );
    printf(" timing_info_present_flag : %d \n", sps->vui.timing_info_present_flag );
      printf("   num_units_in_tick : %d \n", sps->vui.num_units_in_tick );
      printf("   time_scale : %d \n", sps->vui.time_scale );
      printf("   fixed_frame_rate_flag : %d \n", sps->vui.fixed_frame_rate_flag );
    printf(" nal_hrd_parameters_present_flag : %d \n", sps->vui.nal_hrd_parameters_present_flag );
    printf(" vcl_hrd_parameters_present_flag : %d \n", sps->vui.vcl_hrd_parameters_present_flag );
      printf("   low_delay_hrd_flag : %d \n", sps->vui.low_delay_hrd_flag );
    printf(" pic_struct_present_flag : %d \n", sps->vui.pic_struct_present_flag );
    printf(" bitstream_restriction_flag : %d \n", sps->vui.bitstream_restriction_flag );
      printf("   motion_vectors_over_pic_boundaries_flag : %d \n", sps->vui.motion_vectors_over_pic_boundaries_flag );
      printf("   max_bytes_per_pic_denom : %d \n", sps->vui.max_bytes_per_pic_denom );
      printf("   max_bits_per_mb_denom : %d \n", sps->vui.max_bits_per_mb_denom );
      printf("   log2_max_mv_length_horizontal : %d \n", sps->vui.log2_max_mv_length_horizontal );
      printf("   log2_max_mv_length_vertical : %d \n", sps->vui.log2_max_mv_length_vertical );
      printf("   num_reorder_frames : %d \n", sps->vui.num_reorder_frames );
      printf("   max_dec_frame_buffering : %d \n", sps->vui.max_dec_frame_buffering );

    printf("=== HRD ===\n");
    printf(" cpb_cnt_minus1 : %d \n", sps->hrd.cpb_cnt_minus1 );
    printf(" bit_rate_scale : %d \n", sps->hrd.bit_rate_scale );
    printf(" cpb_size_scale : %d \n", sps->hrd.cpb_size_scale );
    int SchedSelIdx;
    for( SchedSelIdx = 0; SchedSelIdx <= sps->hrd.cpb_cnt_minus1; SchedSelIdx++ )
    {
        printf("   bit_rate_value_minus1[%d] : %d \n", SchedSelIdx, sps->hrd.bit_rate_value_minus1[SchedSelIdx] ); // up to cpb_cnt_minus1, which is <= 31
        printf("   cpb_size_value_minus1[%d] : %d \n", SchedSelIdx, sps->hrd.cpb_size_value_minus1[SchedSelIdx] );
        printf("   cbr_flag[%d] : %d \n", SchedSelIdx, sps->hrd.cbr_flag[SchedSelIdx] );
    }
    printf(" initial_cpb_removal_delay_length_minus1 : %d \n", sps->hrd.initial_cpb_removal_delay_length_minus1 );
    printf(" cpb_removal_delay_length_minus1 : %d \n", sps->hrd.cpb_removal_delay_length_minus1 );
    printf(" dpb_output_delay_length_minus1 : %d \n", sps->hrd.dpb_output_delay_length_minus1 );
    printf(" time_offset_length : %d \n", sps->hrd.time_offset_length );
}

static void print_sh(slice_header_t *sh) //DEBUG
{
	printf("frame_num: %d, idr_pic_id: %d, slice_type: %d, pic_parameter_set_id: %d, poc_lsb: %d\n",
			sh->frame_num, sh->idr_pic_id, sh->slice_type, sh->pic_parameter_set_id, sh->pic_order_cnt_lsb);
}

static int intlog2(int x) //TODO check if its not already provided
{
    int log = 0;
    if (x < 0) { x = 0; }
    while ((x >> log) > 0)
    {
        log++;
    }
    if (log > 0 && x == 1 << (log - 1)) { log--; }
    return log;
}

// NOTE: this function was copied from utils/h264_stream.c,
// however we dont include this file as some functions would collide with functions from Wicked engine
/**
   Convert NAL data (Annex B format) to RBSP data.
   The size of rbsp_buf must be the same as size of the nal_buf to guarantee the output will fit.
   If that is not true, output may be truncated and an error will be returned.
   Additionally, certain byte sequences in the input nal_buf are not allowed in the spec and also cause the conversion to fail and an error to be returned.
   @param[in] nal_buf   the nal data
   @param[in,out] nal_size  as input, pointer to the size of the nal data; as output, filled in with the actual size of the nal data
   @param[in,out] rbsp_buf   allocated memory in which to put the rbsp data
   @param[in,out] rbsp_size  as input, pointer to the maximum size of the rbsp data; as output, filled in with the actual size of rbsp data
   @return  actual size of rbsp data, or -1 on error
 */
// 7.3.1 NAL unit syntax
// 7.4.1.1 Encapsulation of an SODB within an RBSP
static int nal_to_rbsp(const uint8_t *nal_buf, int *nal_size, uint8_t *rbsp_buf, int *rbsp_size)
{
    int i;
    int j     = 0;
    int count = 0;

    for( i = 1; i < *nal_size; i++ )
    {
        // in NAL unit, 0x000000, 0x000001 or 0x000002 shall not occur at any byte-aligned position
        if( ( count == 2 ) && ( nal_buf[i] < 0x03) )
        {
            return -1;
        }

        if( ( count == 2 ) && ( nal_buf[i] == 0x03) )
        {
            // check the 4th byte after 0x000003, except when cabac_zero_word is used, in which case the last three bytes of this NAL unit must be 0x000003
            if((i < *nal_size - 1) && (nal_buf[i+1] > 0x03))
            {
                return -1;
            }

            // if cabac_zero_word is used, the final byte of this NAL unit(0x03) is discarded, and the last two bytes of RBSP must be 0x0000
            if(i == *nal_size - 1)
            {
                break;
            }

            i++;
            count = 0;
        }

        if ( j >= *rbsp_size )
        {
            // error, not enough space
            return -1;
        }

        rbsp_buf[j] = nal_buf[i];
        if(nal_buf[i] == 0x00)
        {
            count++;
        }
        else
        {
            count = 0;
        }
        j++;
    }

    *nal_size = i;
    *rbsp_size = j;
    return j;
}

static int more_rbsp_data(bs_t *b) //TODO rewrite
{
    // no more data
    if (bs_eof(b)) { return 0; }

    bs_t bs_tmp = *b; // make copy

    // no rbsp_stop_bit yet
    if (bs_read_u1(&bs_tmp) == 0) { return 1; }

    while (bs_eof(&bs_tmp))
    {
        // A later bit was 1, it wasn't the rsbp_stop_bit
        if (bs_read_u1(&bs_tmp) == 1) { return 1; }
    }

    // All following bits were 0, it was the rsbp_stop_bit
    return 0;
}

static void read_rbsp_trailing_bits(bs_t* b)
{
    /* rbsp_stop_one_bit */ bs_read_u(b, 1);

    while (!bs_byte_aligned(b))
    {
        /* rbsp_alignment_zero_bit */ bs_read_u(b, 1);
    }
}

static void read_scaling_list(bs_t *b, int *scalingList, int sizeOfScalingList, int *useDefaultScalingMatrixFlag)
{
    int lastScale = 8;
    int nextScale = 8;
    int delta_scale;
    for (int j = 0; j < sizeOfScalingList; j++)
    {
        if (nextScale != 0)
        {
            if (0)
            {
                nextScale = scalingList[j];
                if (*useDefaultScalingMatrixFlag) { nextScale = 0; }
                delta_scale = (nextScale - lastScale) % 256;
            }

            delta_scale = bs_read_se(b);

            if (1)
            {
                nextScale = (lastScale + delta_scale + 256) % 256;
                *useDefaultScalingMatrixFlag = (j == 0 && nextScale == 0);
            }
        }
        if (1)
        {
            scalingList[j] = (nextScale == 0) ? lastScale : nextScale;
        }
        lastScale = scalingList[j];
    }
}

static void read_hrd_parameters(sps_t *sps, bs_t *b)
{
	sps->hrd.cpb_cnt_minus1 = bs_read_ue(b);
	sps->hrd.bit_rate_scale = bs_read_u(b, 4);
	sps->hrd.cpb_size_scale = bs_read_u(b, 4);
	for (int SchedSelIdx = 0; SchedSelIdx <= sps->hrd.cpb_cnt_minus1; SchedSelIdx++)
	{
		sps->hrd.bit_rate_value_minus1[SchedSelIdx] = bs_read_ue(b);
		sps->hrd.cpb_size_value_minus1[SchedSelIdx] = bs_read_ue(b);
		sps->hrd.cbr_flag[SchedSelIdx] = bs_read_u1(b);
	}
	sps->hrd.initial_cpb_removal_delay_length_minus1 = bs_read_u(b, 5);
	sps->hrd.cpb_removal_delay_length_minus1 = bs_read_u(b, 5);
	sps->hrd.dpb_output_delay_length_minus1 = bs_read_u(b, 5);
	sps->hrd.time_offset_length = bs_read_u(b, 5);
}

static void read_vui_parameters(sps_t *sps, bs_t *b)
{
	sps->vui.aspect_ratio_info_present_flag = bs_read_u1(b);
	if (sps->vui.aspect_ratio_info_present_flag)
	{
		sps->vui.aspect_ratio_idc = bs_read_u(b, 8);
		if (sps->vui.aspect_ratio_idc == 255) // Extended_SAR
		{
			sps->vui.sar_width = bs_read_u(b, 16);
			sps->vui.sar_height = bs_read_u(b, 16);
		}
	}
	sps->vui.overscan_info_present_flag = bs_read_u1(b);
	if (sps->vui.overscan_info_present_flag)
	{
		sps->vui.overscan_appropriate_flag = bs_read_u1(b);
	}
	sps->vui.video_signal_type_present_flag = bs_read_u1(b);
	if (sps->vui.video_signal_type_present_flag)
	{
		sps->vui.video_format = bs_read_u(b, 3);
		sps->vui.video_full_range_flag = bs_read_u1(b);
		sps->vui.colour_description_present_flag = bs_read_u1(b);
		if (sps->vui.colour_description_present_flag)
		{
			sps->vui.colour_primaries = bs_read_u(b, 8);
			sps->vui.transfer_characteristics = bs_read_u(b, 8);
			sps->vui.matrix_coefficients = bs_read_u(b, 8);
		}
	}
	sps->vui.chroma_loc_info_present_flag = bs_read_u1(b);
	if (sps->vui.chroma_loc_info_present_flag)
	{
		sps->vui.chroma_sample_loc_type_top_field = bs_read_ue(b);
		sps->vui.chroma_sample_loc_type_bottom_field = bs_read_ue(b);
	}
	sps->vui.timing_info_present_flag = bs_read_u1(b);
	if (sps->vui.timing_info_present_flag)
	{
		sps->vui.num_units_in_tick = bs_read_u(b, 32);
		sps->vui.time_scale = bs_read_u(b, 32);
		sps->vui.fixed_frame_rate_flag = bs_read_u1(b);
	}
	sps->vui.nal_hrd_parameters_present_flag = bs_read_u1(b);
	if (sps->vui.nal_hrd_parameters_present_flag)
	{
		read_hrd_parameters(sps, b);
	}
	sps->vui.vcl_hrd_parameters_present_flag = bs_read_u1(b);
	if (sps->vui.vcl_hrd_parameters_present_flag)
	{
		read_hrd_parameters(sps, b);
	}
	if (sps->vui.nal_hrd_parameters_present_flag || sps->vui.vcl_hrd_parameters_present_flag)
	{
		sps->vui.low_delay_hrd_flag = bs_read_u1(b);
	}
	sps->vui.pic_struct_present_flag = bs_read_u1(b);
	sps->vui.bitstream_restriction_flag = bs_read_u1(b);
	if (sps->vui.bitstream_restriction_flag)
	{
		sps->vui.motion_vectors_over_pic_boundaries_flag = bs_read_u1(b);
		sps->vui.max_bytes_per_pic_denom = bs_read_ue(b);
		sps->vui.max_bits_per_mb_denom = bs_read_ue(b);
		sps->vui.log2_max_mv_length_horizontal = bs_read_ue(b);
		sps->vui.log2_max_mv_length_vertical = bs_read_ue(b);
		sps->vui.num_reorder_frames = bs_read_ue(b);
		sps->vui.max_dec_frame_buffering = bs_read_ue(b);
	}
}

static int is_slice_type(int slice_type, int cmp_type)
{
	if (slice_type >= 5) { slice_type -= 5; }
	if (cmp_type >= 5) { cmp_type -= 5; }
	if (slice_type == cmp_type) { return 1; }
	else { return 0; }
}

static void read_ref_pic_list_reordering(slice_header_t *sh, bs_t *b)
{
	if (!is_slice_type(sh->slice_type, SH_SLICE_TYPE_I) &&
		!is_slice_type(sh->slice_type, SH_SLICE_TYPE_SI))
	{
		sh->rplr.ref_pic_list_reordering_flag_l0 = bs_read_u1(b);
		if (sh->rplr.ref_pic_list_reordering_flag_l0)
		{
			int n = -1;
			do
			{
				n++;
				sh->rplr.reorder_l0.reordering_of_pic_nums_idc[n] = bs_read_ue(b);
				if (sh->rplr.reorder_l0.reordering_of_pic_nums_idc[n] == 0 ||
					sh->rplr.reorder_l0.reordering_of_pic_nums_idc[n] == 1)
				{
					sh->rplr.reorder_l0.abs_diff_pic_num_minus1[n] = bs_read_ue(b);
				}
				else if (sh->rplr.reorder_l0.reordering_of_pic_nums_idc[n] == 2)
				{
					sh->rplr.reorder_l0.long_term_pic_num[n] = bs_read_ue(b);
				}
			} while (sh->rplr.reorder_l0.reordering_of_pic_nums_idc[n] != 3 && !bs_eof(b));
		}
	}
	if (is_slice_type(sh->slice_type, SH_SLICE_TYPE_B))
	{
		sh->rplr.ref_pic_list_reordering_flag_l1 = bs_read_u1(b);
		if (sh->rplr.ref_pic_list_reordering_flag_l1)
		{
			int n = -1;
			do
			{
				n++;
				sh->rplr.reorder_l1.reordering_of_pic_nums_idc[n] = bs_read_ue(b);
				if (sh->rplr.reorder_l1.reordering_of_pic_nums_idc[n] == 0 ||
					sh->rplr.reorder_l1.reordering_of_pic_nums_idc[n] == 1)
				{
					sh->rplr.reorder_l1.abs_diff_pic_num_minus1[n] = bs_read_ue(b);
				}
				else if (sh->rplr.reorder_l1.reordering_of_pic_nums_idc[n] == 2)
				{
					sh->rplr.reorder_l1.long_term_pic_num[n] = bs_read_ue(b);
				}
			} while (sh->rplr.reorder_l1.reordering_of_pic_nums_idc[n] != 3 && !bs_eof(b));
		}
	}
}

static void read_pred_weight_table(slice_header_t *sh, const sps_t *sps, const pps_t *pps, bs_t *b)
{
	int i, j;

	sh->pwt.luma_log2_weight_denom = bs_read_ue(b);
	if (sps->chroma_format_idc != 0)
	{
		sh->pwt.chroma_log2_weight_denom = bs_read_ue(b);
	}
	for (i = 0; i <= pps->num_ref_idx_l0_active_minus1; i++)
	{
		sh->pwt.luma_weight_l0_flag[i] = bs_read_u1(b);
		if (sh->pwt.luma_weight_l0_flag[i])
		{
			sh->pwt.luma_weight_l0[i] = bs_read_se(b);
			sh->pwt.luma_offset_l0[i] = bs_read_se(b);
		}
		if (sps->chroma_format_idc != 0)
		{
			sh->pwt.chroma_weight_l0_flag[i] = bs_read_u1(b);
			if (sh->pwt.chroma_weight_l0_flag[i])
			{
				for (j = 0; j < 2; j++)
				{
					sh->pwt.chroma_weight_l0[i][j] = bs_read_se(b);
					sh->pwt.chroma_offset_l0[i][j] = bs_read_se(b);
				}
			}
		}
	}
	if (is_slice_type(sh->slice_type, SH_SLICE_TYPE_B))
	{
		for (i = 0; i <= pps->num_ref_idx_l1_active_minus1; i++)
		{
			sh->pwt.luma_weight_l1_flag[i] = bs_read_u1(b);
			if (sh->pwt.luma_weight_l1_flag[i])
			{
				sh->pwt.luma_weight_l1[i] = bs_read_se(b);
				sh->pwt.luma_offset_l1[i] = bs_read_se(b);
			}
			if (sps->chroma_format_idc != 0)
			{
				sh->pwt.chroma_weight_l1_flag[i] = bs_read_u1(b);
				if (sh->pwt.chroma_weight_l1_flag[i])
				{
					for (j = 0; j < 2; j++)
					{
						sh->pwt.chroma_weight_l1[i][j] = bs_read_se(b);
						sh->pwt.chroma_offset_l1[i][j] = bs_read_se(b);
					}
				}
			}
		}
	}
}

static void read_dec_ref_pic_marking(slice_header_t *sh, int nal_type, bs_t *b)
{
	if (nal_type == NAL_H264_IDR)
	{
		sh->drpm.no_output_of_prior_pics_flag = bs_read_u1(b);
		sh->drpm.long_term_reference_flag = bs_read_u1(b);
	}
	else
	{
		sh->drpm.adaptive_ref_pic_marking_mode_flag = bs_read_u1(b);
		if (sh->drpm.adaptive_ref_pic_marking_mode_flag)
		{
			int n = -1;
			do
			{
				n++;
				sh->drpm.memory_management_control_operation[n] = bs_read_ue(b);
				if (sh->drpm.memory_management_control_operation[n] == 1 ||
					sh->drpm.memory_management_control_operation[n] == 3)
				{
					sh->drpm.difference_of_pic_nums_minus1[n] = bs_read_ue(b);
				}
				if (sh->drpm.memory_management_control_operation[n] == 2)
				{
					sh->drpm.long_term_pic_num[n] = bs_read_ue(b);
				}
				if (sh->drpm.memory_management_control_operation[n] == 3 ||
					sh->drpm.memory_management_control_operation[n] == 6)
				{
					sh->drpm.long_term_frame_idx[n] = bs_read_ue(b);
				}
				if (sh->drpm.memory_management_control_operation[n] == 4)
				{
					sh->drpm.max_long_term_frame_idx_plus1[n] = bs_read_ue(b);
				}
			} while (sh->drpm.memory_management_control_operation[n] != 0 && !bs_eof(b));
		}
	}
}

static void read_pps(pps_t *pps, bs_t *b)
{
    pps->pic_parameter_set_id = bs_read_ue(b);
    pps->seq_parameter_set_id = bs_read_ue(b);
    pps->entropy_coding_mode_flag = bs_read_u1(b);
    pps->pic_order_present_flag = bs_read_u1(b);
    pps->num_slice_groups_minus1 = bs_read_ue(b);

    if (pps->num_slice_groups_minus1 > 0)
    {
        pps->slice_group_map_type = bs_read_ue(b);
        if (pps->slice_group_map_type == 0)
        {
            for (int i_group = 0; i_group <= pps->num_slice_groups_minus1; i_group++)
            {
                pps->run_length_minus1[i_group] = bs_read_ue(b);
            }
        }
        else if (pps->slice_group_map_type == 2)
        {
            for (int i_group = 0; i_group < pps->num_slice_groups_minus1; i_group++)
            {
                pps->top_left[i_group] = bs_read_ue(b);
                pps->bottom_right[i_group] = bs_read_ue(b);
            }
        }
        else if (pps->slice_group_map_type == 3 ||
            pps->slice_group_map_type == 4 ||
            pps->slice_group_map_type == 5)
        {
            pps->slice_group_change_direction_flag = bs_read_u1(b);
            pps->slice_group_change_rate_minus1 = bs_read_ue(b);
        }
        else if (pps->slice_group_map_type == 6)
        {
            pps->pic_size_in_map_units_minus1 = bs_read_ue(b);
            for (int i = 0; i <= pps->pic_size_in_map_units_minus1; i++)
            {
                int v = intlog2(pps->num_slice_groups_minus1 + 1);
                pps->slice_group_id[i] = bs_read_u(b, v);
            }
        }
    }
    pps->num_ref_idx_l0_active_minus1 = bs_read_ue(b);
    pps->num_ref_idx_l1_active_minus1 = bs_read_ue(b);
    pps->weighted_pred_flag = bs_read_u1(b);
    pps->weighted_bipred_idc = bs_read_u(b, 2);
    pps->pic_init_qp_minus26 = bs_read_se(b);
    pps->pic_init_qs_minus26 = bs_read_se(b);
    pps->chroma_qp_index_offset = bs_read_se(b);
    pps->deblocking_filter_control_present_flag = bs_read_u1(b);
    pps->constrained_intra_pred_flag = bs_read_u1(b);
    pps->redundant_pic_cnt_present_flag = bs_read_u1(b);

    int have_more_data = 0;
    if (1) { have_more_data = more_rbsp_data(b); }
    if (0)
    {
        have_more_data = (pps->transform_8x8_mode_flag |
                          pps->pic_scaling_matrix_present_flag |
                          pps->second_chroma_qp_index_offset) != 0;
    }

    if (have_more_data)
    {
        pps->transform_8x8_mode_flag = bs_read_u1(b);
        pps->pic_scaling_matrix_present_flag = bs_read_u1(b);
        if (pps->pic_scaling_matrix_present_flag)
        {
            for (int i = 0; i < 6 + 2 * pps->transform_8x8_mode_flag; i++)
            {
                pps->pic_scaling_list_present_flag[i] = bs_read_u1(b);
                if (pps->pic_scaling_list_present_flag[i])
                {
                    if (i < 6)
                    {
                        read_scaling_list(b, pps->ScalingList4x4[i], 16,
                            &(pps->UseDefaultScalingMatrix4x4Flag[i]));
                    }
                    else
                    {
                        read_scaling_list(b, pps->ScalingList8x8[i - 6], 64,
                            &(pps->UseDefaultScalingMatrix8x8Flag[i - 6]));
                    }
                }
            }
        }
        pps->second_chroma_qp_index_offset = bs_read_se(b);
    }
    read_rbsp_trailing_bits(b);
}

static void read_sps(sps_t *sps, bs_t *b)
{
	sps->profile_idc = bs_read_u(b, 8);
	sps->constraint_set0_flag = bs_read_u1(b);
	sps->constraint_set1_flag = bs_read_u1(b);
	sps->constraint_set2_flag = bs_read_u1(b);
	sps->constraint_set3_flag = bs_read_u1(b);
	sps->constraint_set4_flag = bs_read_u1(b);
	sps->constraint_set5_flag = bs_read_u1(b);
	/* reserved_zero_2bits */ bs_read_u(b, 2);
	sps->level_idc = bs_read_u(b, 8);
	sps->seq_parameter_set_id = bs_read_ue(b);

	if (sps->profile_idc == 100 || sps->profile_idc == 110 ||
		sps->profile_idc == 122 || sps->profile_idc == 144)
	{
		sps->chroma_format_idc = bs_read_ue(b);
		if (sps->chroma_format_idc == 3)
		{
			sps->separate_colour_plane_flag = bs_read_u1(b);
		}
		sps->bit_depth_luma_minus8 = bs_read_ue(b);
		sps->bit_depth_chroma_minus8 = bs_read_ue(b);
		sps->qpprime_y_zero_transform_bypass_flag = bs_read_u1(b);
		sps->seq_scaling_matrix_present_flag = bs_read_u1(b);
		if (sps->seq_scaling_matrix_present_flag)
		{
			for (int i = 0; i < 8; i++)
			{
				sps->seq_scaling_list_present_flag[i] = bs_read_u1(b);
				if (sps->seq_scaling_list_present_flag[i])
				{
					if (i < 6)
					{
						read_scaling_list(b, sps->ScalingList4x4[i], 16,
							&(sps->UseDefaultScalingMatrix4x4Flag[i]));
					}
					else
					{
						read_scaling_list(b, sps->ScalingList8x8[i - 6], 64,
							&(sps->UseDefaultScalingMatrix8x8Flag[i - 6]));
					}
				}
			}
		}
	}
	sps->log2_max_frame_num_minus4 = bs_read_ue(b);
	sps->pic_order_cnt_type = bs_read_ue(b);
	if (sps->pic_order_cnt_type == 0)
	{
		sps->log2_max_pic_order_cnt_lsb_minus4 = bs_read_ue(b);
	}
	else if (sps->pic_order_cnt_type == 1)
	{
		sps->delta_pic_order_always_zero_flag = bs_read_u1(b);
		sps->offset_for_non_ref_pic = bs_read_se(b);
		sps->offset_for_top_to_bottom_field = bs_read_se(b);
		sps->num_ref_frames_in_pic_order_cnt_cycle = bs_read_ue(b);
		for (int i = 0; i < sps->num_ref_frames_in_pic_order_cnt_cycle; i++)
		{
			sps->offset_for_ref_frame[i] = bs_read_se(b);
		}
	}
	sps->num_ref_frames = bs_read_ue(b);
	sps->gaps_in_frame_num_value_allowed_flag = bs_read_u1(b);
	sps->pic_width_in_mbs_minus1 = bs_read_ue(b);
	sps->pic_height_in_map_units_minus1 = bs_read_ue(b);
	sps->frame_mbs_only_flag = bs_read_u1(b);
	if (!sps->frame_mbs_only_flag)
	{
		sps->mb_adaptive_frame_field_flag = bs_read_u1(b);
	}
	sps->direct_8x8_inference_flag = bs_read_u1(b);
	sps->frame_cropping_flag = bs_read_u1(b);
	if (sps->frame_cropping_flag)
	{
		sps->frame_crop_left_offset = bs_read_ue(b);
		sps->frame_crop_right_offset = bs_read_ue(b);
		sps->frame_crop_top_offset = bs_read_ue(b);
		sps->frame_crop_bottom_offset = bs_read_ue(b);
	}
	sps->vui_parameters_present_flag = bs_read_u1(b);
	if (sps->vui_parameters_present_flag)
	{
		read_vui_parameters(sps, b);
	}
	read_rbsp_trailing_bits(b);
}

static bool read_slice_header(slice_header_t *sh, int nal_type, int nal_idc,
					   		  const pps_t *pps_array, const sps_t *sps_array, bs_t *b)
{
	// modification: returns whether indices are in bounds
	sh->first_mb_in_slice = bs_read_ue(b);
	sh->slice_type = bs_read_ue(b);
	sh->pic_parameter_set_id = bs_read_ue(b);

	int pps_id = sh->pic_parameter_set_id;
	if (pps_id < 0 || pps_id >= MAX_PPS_IDS) return false;
	const pps_t* pps = pps_array + pps_id;
	
	int sps_id = pps->seq_parameter_set_id;
	if (sps_id < 0 || sps_id >= MAX_SPS_IDS) return false;
	const sps_t* sps = sps_array + pps->seq_parameter_set_id;

	sh->frame_num = bs_read_u(b, sps->log2_max_frame_num_minus4 + 4); // was u(v)
	if (!sps->frame_mbs_only_flag)
	{
		sh->field_pic_flag = bs_read_u1(b);
		if (sh->field_pic_flag)
		{
			sh->bottom_field_flag = bs_read_u1(b);
		}
	}
	if (nal_type == NAL_H264_IDR)
	{
		sh->idr_pic_id = bs_read_ue(b);
	}
	if (sps->pic_order_cnt_type == 0)
	{
		sh->pic_order_cnt_lsb = bs_read_u(b, sps->log2_max_pic_order_cnt_lsb_minus4 + 4); // was u(v)
		if (pps->pic_order_present_flag && !sh->field_pic_flag)
		{
			sh->delta_pic_order_cnt_bottom = bs_read_se(b);
		}
	}
	if (sps->pic_order_cnt_type == 1 && !sps->delta_pic_order_always_zero_flag)
	{
		sh->delta_pic_order_cnt[0] = bs_read_se(b);
		if (pps->pic_order_present_flag && !sh->field_pic_flag)
		{
			sh->delta_pic_order_cnt[1] = bs_read_se(b);
		}
	}
	if (pps->redundant_pic_cnt_present_flag)
	{
		sh->redundant_pic_cnt = bs_read_ue(b);
	}
	if (is_slice_type(sh->slice_type, SH_SLICE_TYPE_B))
	{
		sh->direct_spatial_mv_pred_flag = bs_read_u1(b);
	}
	if (is_slice_type(sh->slice_type, SH_SLICE_TYPE_P) ||
		is_slice_type(sh->slice_type, SH_SLICE_TYPE_SP) ||
		is_slice_type(sh->slice_type, SH_SLICE_TYPE_B))
	{
		sh->num_ref_idx_active_override_flag = bs_read_u1(b);
		if (sh->num_ref_idx_active_override_flag)
		{
			sh->num_ref_idx_l0_active_minus1 = bs_read_ue(b);
			if (is_slice_type(sh->slice_type, SH_SLICE_TYPE_B))
			{
				sh->num_ref_idx_l1_active_minus1 = bs_read_ue(b);
			}
		}
	}
	read_ref_pic_list_reordering(sh, b);
	if ((pps->weighted_pred_flag &&
		(is_slice_type(sh->slice_type, SH_SLICE_TYPE_P) || is_slice_type(sh->slice_type, SH_SLICE_TYPE_SP))) ||
		(pps->weighted_bipred_idc == 1 && is_slice_type(sh->slice_type, SH_SLICE_TYPE_B)))
	{
		read_pred_weight_table(sh, sps, pps, b);
	}
	if (nal_idc != 0)
	{
		read_dec_ref_pic_marking(sh, nal_type, b);
	}
	if (pps->entropy_coding_mode_flag &&
		!is_slice_type(sh->slice_type, SH_SLICE_TYPE_I) &&
		!is_slice_type(sh->slice_type, SH_SLICE_TYPE_SI))
	{
		sh->cabac_init_idc = bs_read_ue(b);
	}
	sh->slice_qp_delta = bs_read_se(b);
	if (is_slice_type(sh->slice_type, SH_SLICE_TYPE_SP) || is_slice_type(sh->slice_type, SH_SLICE_TYPE_SI))
	{
		if (is_slice_type(sh->slice_type, SH_SLICE_TYPE_SP))
		{
			sh->sp_for_switch_flag = bs_read_u1(b);
		}
		sh->slice_qs_delta = bs_read_se(b);
	}
	if (pps->deblocking_filter_control_present_flag)
	{
		sh->disable_deblocking_filter_idc = bs_read_ue(b);
		if (sh->disable_deblocking_filter_idc != 1)
		{
			sh->slice_alpha_c0_offset_div2 = bs_read_se(b);
			sh->slice_beta_offset_div2 = bs_read_se(b);
		}
	}
	if (pps->num_slice_groups_minus1 > 0 &&
		pps->slice_group_map_type >= 3 && pps->slice_group_map_type <= 5)
	{
		int v = intlog2(pps->pic_size_in_map_units_minus1 + pps->slice_group_change_rate_minus1 + 1);
		sh->slice_group_change_cycle = bs_read_u(b, v);
	}

	return true;
}

static int32_t get_picture_order_count(int *prev_poc_msb, int *prev_poc_lsb,
									   int sps_log2_max_pic_order_cnt_lsb_minus4, int sh_pic_order_cnt_lsb)
{
	// Type 0 POC value calculation

	// For each slice, do the following:
	int prev_pic_order_cnt_msb = *prev_poc_msb;
	int prev_pic_order_cnt_lsb = *prev_poc_lsb;

	// Rec. ITU-T H.264 (08/2021) page 77
	int max_pic_order_cnt_lsb = 1 << (sps_log2_max_pic_order_cnt_lsb_minus4 + 4);
	int pic_order_cnt_lsb = sh_pic_order_cnt_lsb;

	// Rec. ITU-T H.264 (08/2021) page 115
	// from https://videonerd.website/frames-with-negative-pocs/
	int pic_order_cnt_msb = 0;
	if (pic_order_cnt_lsb < prev_pic_order_cnt_lsb && (prev_pic_order_cnt_lsb - pic_order_cnt_lsb) >= max_pic_order_cnt_lsb / 2)
	{
		pic_order_cnt_msb = prev_pic_order_cnt_msb + max_pic_order_cnt_lsb; // pic_order_cnt_lsb wrapped around
	}
	else if (pic_order_cnt_lsb > prev_pic_order_cnt_lsb && (pic_order_cnt_lsb - prev_pic_order_cnt_lsb) > max_pic_order_cnt_lsb / 2)
	{
		pic_order_cnt_msb = prev_pic_order_cnt_msb - max_pic_order_cnt_lsb; // here negative POC might occur
	}
	else
	{
		pic_order_cnt_msb = prev_pic_order_cnt_msb;
	}
	*prev_poc_lsb = pic_order_cnt_lsb;
	*prev_poc_msb = pic_order_cnt_msb;

	// final value of PicOrderCnt:
	return (int32_t)(pic_order_cnt_msb + pic_order_cnt_lsb);
}


// ---Convert functions---
static StdVideoH264ProfileIdc profile_idc_to_h264_flag(int profile_idc)
{
	//TODO switch instead?

	if (profile_idc == 66) return STD_VIDEO_H264_PROFILE_IDC_BASELINE;

	if (profile_idc == 77) return STD_VIDEO_H264_PROFILE_IDC_MAIN;
	// Extended profile (A.2.3), there's no h264 flag for it however => returning main profile
	if (profile_idc == 88) return STD_VIDEO_H264_PROFILE_IDC_MAIN;

	if (profile_idc == 100) return STD_VIDEO_H264_PROFILE_IDC_HIGH;
	// High 10 profile (A.2.5), there's no h264 flag for it however => returning high profile
	if (profile_idc == 110) return STD_VIDEO_H264_PROFILE_IDC_HIGH;
	// High 4:2:2 profile (A.2.6), there's no h264 flag for it however => returning high profile
	if (profile_idc == 122) return STD_VIDEO_H264_PROFILE_IDC_HIGH;

	if (profile_idc == 244) return STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE;

	return STD_VIDEO_H264_PROFILE_IDC_INVALID;
}

// Following convert functions sps_to_vk_sps and pps_to_vk_pps were copied and modified also from WickedEngine project:
// https://github.com/turanszkij/WickedEngine/blob/87e9870b500951f38448c158421818cb9036c583/WickedEngine/wiGraphicsDevice_Vulkan.cpp

static void sps_to_vk_sps(const sps_t *sps, StdVideoH264SequenceParameterSet *vk_sps,
						  StdVideoH264SequenceParameterSetVui *vk_vui, StdVideoH264HrdParameters *vk_hrd)
{
	vk_sps->flags.constraint_set0_flag = sps->constraint_set0_flag;
	vk_sps->flags.constraint_set1_flag = sps->constraint_set1_flag;
	vk_sps->flags.constraint_set2_flag = sps->constraint_set2_flag;
	vk_sps->flags.constraint_set3_flag = sps->constraint_set3_flag;
	vk_sps->flags.constraint_set4_flag = sps->constraint_set4_flag;
	vk_sps->flags.constraint_set5_flag = sps->constraint_set5_flag;
	vk_sps->flags.direct_8x8_inference_flag = sps->direct_8x8_inference_flag;
	vk_sps->flags.mb_adaptive_frame_field_flag = sps->mb_adaptive_frame_field_flag;
	vk_sps->flags.frame_mbs_only_flag = sps->frame_mbs_only_flag;
	vk_sps->flags.delta_pic_order_always_zero_flag = sps->delta_pic_order_always_zero_flag;
	vk_sps->flags.separate_colour_plane_flag = sps->separate_colour_plane_flag;
	vk_sps->flags.gaps_in_frame_num_value_allowed_flag = sps->gaps_in_frame_num_value_allowed_flag;
	vk_sps->flags.qpprime_y_zero_transform_bypass_flag = sps->qpprime_y_zero_transform_bypass_flag;
	vk_sps->flags.frame_cropping_flag = sps->frame_cropping_flag;
	vk_sps->flags.seq_scaling_matrix_present_flag = sps->seq_scaling_matrix_present_flag;
	vk_sps->flags.vui_parameters_present_flag = sps->vui_parameters_present_flag;

	if (vk_sps->flags.vui_parameters_present_flag)
	{
		vk_sps->pSequenceParameterSetVui = vk_vui;
		vk_vui->flags.aspect_ratio_info_present_flag = sps->vui.aspect_ratio_info_present_flag;
		vk_vui->flags.overscan_info_present_flag = sps->vui.overscan_info_present_flag;
		vk_vui->flags.overscan_appropriate_flag = sps->vui.overscan_appropriate_flag;
		vk_vui->flags.video_signal_type_present_flag = sps->vui.video_signal_type_present_flag;
		vk_vui->flags.video_full_range_flag = sps->vui.video_full_range_flag;
		vk_vui->flags.color_description_present_flag = sps->vui.colour_description_present_flag;
		vk_vui->flags.chroma_loc_info_present_flag = sps->vui.chroma_loc_info_present_flag;
		vk_vui->flags.timing_info_present_flag = sps->vui.timing_info_present_flag;
		vk_vui->flags.fixed_frame_rate_flag = sps->vui.fixed_frame_rate_flag;
		vk_vui->flags.bitstream_restriction_flag = sps->vui.bitstream_restriction_flag;
		vk_vui->flags.nal_hrd_parameters_present_flag = sps->vui.nal_hrd_parameters_present_flag;
		vk_vui->flags.vcl_hrd_parameters_present_flag = sps->vui.vcl_hrd_parameters_present_flag;

		vk_vui->aspect_ratio_idc = (StdVideoH264AspectRatioIdc)sps->vui.aspect_ratio_idc;
		vk_vui->sar_width = sps->vui.sar_width;
		vk_vui->sar_height = sps->vui.sar_height;
		vk_vui->video_format = sps->vui.video_format;
		vk_vui->colour_primaries = sps->vui.colour_primaries;
		vk_vui->transfer_characteristics = sps->vui.transfer_characteristics;
		vk_vui->matrix_coefficients = sps->vui.matrix_coefficients;
		vk_vui->num_units_in_tick = sps->vui.num_units_in_tick;
		vk_vui->time_scale = sps->vui.time_scale;
		vk_vui->max_num_reorder_frames = sps->vui.num_reorder_frames;
		vk_vui->max_dec_frame_buffering = sps->vui.max_dec_frame_buffering;
		vk_vui->chroma_sample_loc_type_top_field = sps->vui.chroma_sample_loc_type_top_field;
		vk_vui->chroma_sample_loc_type_bottom_field = sps->vui.chroma_sample_loc_type_bottom_field;

		vk_vui->pHrdParameters = vk_hrd;
		vk_hrd->cpb_cnt_minus1 = sps->hrd.cpb_cnt_minus1;
		vk_hrd->bit_rate_scale = sps->hrd.bit_rate_scale;
		vk_hrd->cpb_size_scale = sps->hrd.cpb_size_scale;
		for (size_t j = 0; j < sizeof(sps->hrd.bit_rate_value_minus1) / sizeof(sps->hrd.bit_rate_value_minus1[0]); ++j)
		{
			vk_hrd->bit_rate_value_minus1[j] = sps->hrd.bit_rate_value_minus1[j];
			vk_hrd->cpb_size_value_minus1[j] = sps->hrd.cpb_size_value_minus1[j];
			vk_hrd->cbr_flag[j] = sps->hrd.cbr_flag[j];
		}
		vk_hrd->initial_cpb_removal_delay_length_minus1 = sps->hrd.initial_cpb_removal_delay_length_minus1;
		vk_hrd->cpb_removal_delay_length_minus1 = sps->hrd.cpb_removal_delay_length_minus1;
		vk_hrd->dpb_output_delay_length_minus1 = sps->hrd.dpb_output_delay_length_minus1;
		vk_hrd->time_offset_length = sps->hrd.time_offset_length;
	}

	vk_sps->profile_idc = profile_idc_to_h264_flag(sps->profile_idc);
	switch (sps->level_idc)
	{
	case 0:
		vk_sps->level_idc = STD_VIDEO_H264_LEVEL_IDC_1_0;
		break;
	case 11:
		vk_sps->level_idc = STD_VIDEO_H264_LEVEL_IDC_1_1;
		break;
	case 12:
		vk_sps->level_idc = STD_VIDEO_H264_LEVEL_IDC_1_2;
		break;
	case 13:
		vk_sps->level_idc = STD_VIDEO_H264_LEVEL_IDC_1_3;
		break;
	case 20:
		vk_sps->level_idc = STD_VIDEO_H264_LEVEL_IDC_2_0;
		break;
	case 21:
		vk_sps->level_idc = STD_VIDEO_H264_LEVEL_IDC_2_1;
		break;
	case 22:
		vk_sps->level_idc = STD_VIDEO_H264_LEVEL_IDC_2_2;
		break;
	case 30:
		vk_sps->level_idc = STD_VIDEO_H264_LEVEL_IDC_3_0;
		break;
	case 31:
		vk_sps->level_idc = STD_VIDEO_H264_LEVEL_IDC_3_1;
		break;
	case 32:
		vk_sps->level_idc = STD_VIDEO_H264_LEVEL_IDC_3_2;
		break;
	case 40:
		vk_sps->level_idc = STD_VIDEO_H264_LEVEL_IDC_4_0;
		break;
	case 41:
		vk_sps->level_idc = STD_VIDEO_H264_LEVEL_IDC_4_1;
		break;
	case 42:
		vk_sps->level_idc = STD_VIDEO_H264_LEVEL_IDC_4_2;
		break;
	case 50:
		vk_sps->level_idc = STD_VIDEO_H264_LEVEL_IDC_5_0;
		break;
	case 51:
		vk_sps->level_idc = STD_VIDEO_H264_LEVEL_IDC_5_1;
		break;
	case 52:
		vk_sps->level_idc = STD_VIDEO_H264_LEVEL_IDC_5_2;
		break;
	case 60:
		vk_sps->level_idc = STD_VIDEO_H264_LEVEL_IDC_6_0;
		break;
	case 61:
		vk_sps->level_idc = STD_VIDEO_H264_LEVEL_IDC_6_1;
		break;
	case 62:
		vk_sps->level_idc = STD_VIDEO_H264_LEVEL_IDC_6_2;
		break;
	default:
		assert(0); //wrong value for SPS level_idc
		break;
	}

	//vk_sps->chroma_format_idc = STD_VIDEO_H264_CHROMA_FORMAT_IDC_420; // only one we support currently
	//NOTE change for vulkan_decode
	vk_sps->chroma_format_idc = (StdVideoH264ChromaFormatIdc)sps->chroma_format_idc;
	vk_sps->seq_parameter_set_id = sps->seq_parameter_set_id;
	vk_sps->bit_depth_luma_minus8 = sps->bit_depth_luma_minus8;
	vk_sps->bit_depth_chroma_minus8 = sps->bit_depth_chroma_minus8;
	vk_sps->log2_max_frame_num_minus4 = sps->log2_max_frame_num_minus4;
	vk_sps->pic_order_cnt_type = (StdVideoH264PocType)sps->pic_order_cnt_type;
	vk_sps->offset_for_non_ref_pic = sps->offset_for_non_ref_pic;
	vk_sps->offset_for_top_to_bottom_field = sps->offset_for_top_to_bottom_field;
	vk_sps->log2_max_pic_order_cnt_lsb_minus4 = sps->log2_max_pic_order_cnt_lsb_minus4;
	vk_sps->num_ref_frames_in_pic_order_cnt_cycle = sps->num_ref_frames_in_pic_order_cnt_cycle;
	vk_sps->max_num_ref_frames = sps->num_ref_frames;
	vk_sps->pic_width_in_mbs_minus1 = sps->pic_width_in_mbs_minus1;
	vk_sps->pic_height_in_map_units_minus1 = sps->pic_height_in_map_units_minus1;
	vk_sps->frame_crop_left_offset = sps->frame_crop_left_offset;
	vk_sps->frame_crop_right_offset = sps->frame_crop_right_offset;
	vk_sps->frame_crop_top_offset = sps->frame_crop_top_offset;
	vk_sps->frame_crop_bottom_offset = sps->frame_crop_bottom_offset;
	vk_sps->pOffsetForRefFrame = sps->offset_for_ref_frame;
}

static void pps_to_vk_pps(const pps_t *pps, StdVideoH264PictureParameterSet *vk_pps,
						  StdVideoH264ScalingLists *vk_scalinglist)
{
	vk_pps->flags.transform_8x8_mode_flag = pps->transform_8x8_mode_flag;
	vk_pps->flags.redundant_pic_cnt_present_flag = pps->redundant_pic_cnt_present_flag;
	vk_pps->flags.constrained_intra_pred_flag = pps->constrained_intra_pred_flag;
	vk_pps->flags.deblocking_filter_control_present_flag = pps->deblocking_filter_control_present_flag;
	vk_pps->flags.weighted_pred_flag = pps->weighted_pred_flag;
	vk_pps->flags.bottom_field_pic_order_in_frame_present_flag = pps->pic_order_present_flag;
	vk_pps->flags.entropy_coding_mode_flag = pps->entropy_coding_mode_flag;
	vk_pps->flags.pic_scaling_matrix_present_flag = pps->pic_scaling_matrix_present_flag;

	vk_pps->seq_parameter_set_id = pps->seq_parameter_set_id;
	vk_pps->pic_parameter_set_id = pps->pic_parameter_set_id;
	vk_pps->num_ref_idx_l0_default_active_minus1 = pps->num_ref_idx_l0_active_minus1;
	vk_pps->num_ref_idx_l1_default_active_minus1 = pps->num_ref_idx_l1_active_minus1;
	vk_pps->weighted_bipred_idc = (StdVideoH264WeightedBipredIdc)pps->weighted_bipred_idc;
	vk_pps->pic_init_qp_minus26 = pps->pic_init_qp_minus26;
	vk_pps->pic_init_qs_minus26 = pps->pic_init_qs_minus26;
	vk_pps->chroma_qp_index_offset = pps->chroma_qp_index_offset;
	vk_pps->second_chroma_qp_index_offset = pps->second_chroma_qp_index_offset;

	vk_pps->pScalingLists = vk_scalinglist;
	for (size_t j = 0; j < sizeof(pps->pic_scaling_list_present_flag) / sizeof(pps->pic_scaling_list_present_flag[0]); ++j)
	{
		vk_scalinglist->scaling_list_present_mask |= pps->pic_scaling_list_present_flag[j] << j;
	}
	for (size_t j = 0; j < sizeof(pps->UseDefaultScalingMatrix4x4Flag) / sizeof(pps->UseDefaultScalingMatrix4x4Flag[0]); ++j)
	{
		vk_scalinglist->use_default_scaling_matrix_mask |= pps->UseDefaultScalingMatrix4x4Flag[j] << j;
	}
	for (size_t j = 0; j < sizeof(pps->ScalingList4x4) / sizeof(pps->ScalingList4x4[0]); ++j)
	{
		for (size_t k = 0; k < sizeof(pps->ScalingList4x4[j]) / sizeof(pps->ScalingList4x4[j][0]); ++k)
		{
			vk_scalinglist->ScalingList4x4[j][k] = (uint8_t)pps->ScalingList4x4[j][k];
		}
	}
	for (size_t j = 0; j < sizeof(pps->ScalingList8x8) / sizeof(pps->ScalingList8x8[0]); ++j)
	{
		for (size_t k = 0; k < sizeof(pps->ScalingList8x8[j]) / sizeof(pps->ScalingList8x8[j][0]); ++k)
		{
			vk_scalinglist->ScalingList8x8[j][k] = (uint8_t)pps->ScalingList8x8[j][k];
		}
	}
}

#endif // VULKAN_DECODE_H264_H
