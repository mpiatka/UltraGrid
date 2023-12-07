/**
 * @file   video_decompress/vulkan_decode.c
 * @author Ondřej Richtr     <524885@mail.muni.cz>
 */
 
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "debug.h"
#include "host.h"				//?
#include "lib_common.h"
#include "video.h"				//?
#include "video_decompress.h"


struct state_vulkan_decompress
{
	int test;
};

static void * vulkan_decompress_init(void)
{
	struct state_vulkan_decompress *s = calloc(1, sizeof(struct state_vulkan_decompress));
	if (!s) return NULL;

	s->test = 13;

	return s;
}

static void vulkan_decompress_done(void *state)
{
	struct state_vulkan_decompress *s = (struct state_vulkan_decompress *)state;
	if (s) debug_msg("[vulkan_decode] Vulkan_decompress_done freeing state '%d'\n", s->test);
	
	free(s);
}

static int vulkan_decompress_reconfigure(void *state, struct video_desc desc,
											 int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
	UNUSED(state);
	UNUSED(desc);
	UNUSED(rshift);
	UNUSED(gshift);
	UNUSED(bshift);
	UNUSED(pitch);
	UNUSED(out_codec);
	//TODO
	return 0;
}

static int vulkan_decompress_get_property(void *state, int property, void *val, size_t *len)
{
	UNUSED(property);
	UNUSED(val);
	UNUSED(len);
	struct state_vulkan_decompress *s = (struct state_vulkan_decompress *)state;
    UNUSED(s);
	//TODO
	return 0;
}


static int vulkan_decompress_get_priority(codec_t compression, struct pixfmt_desc internal, codec_t ugc)
{
	UNUSED(compression);
	UNUSED(internal);
	UNUSED(ugc);
	//TODO
	return -1;
}

static decompress_status vulkan_decompress(void *state, unsigned char *dst, unsigned char *src,
                unsigned int src_len, int frame_seq, struct video_frame_callbacks *callbacks,
                struct pixfmt_desc *internal_prop)
{
	UNUSED(dst);
	UNUSED(src);
	UNUSED(src_len);
	UNUSED(frame_seq);
	UNUSED(callbacks);
	UNUSED(internal_prop);
	struct state_vulkan_decompress *s = (struct state_vulkan_decompress *)state;
    UNUSED(s);
    //TODO
    decompress_status res = DECODER_NO_FRAME;
    
    return res;
};

static const struct video_decompress_info vulkan_info = {
        vulkan_decompress_init,
        vulkan_decompress_reconfigure,
        vulkan_decompress,
        vulkan_decompress_get_property,
        vulkan_decompress_done,
        vulkan_decompress_get_priority,
};


REGISTER_MODULE(vulkan_decode, &vulkan_info, LIBRARY_CLASS_VIDEO_DECOMPRESS, VIDEO_DECOMPRESS_ABI_VERSION);
