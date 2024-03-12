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

//#include <windows.h> //LoadLibrary

#include "vulkan/vulkan.h"
//#include "vk_video/vulkan_video_codecs_common.h" //?


struct state_vulkan_decompress
{
	int test;
	HMODULE vulkanLib;
	VkInstance instance;
	PFN_vkGetInstanceProcAddr loader;
	VkPhysicalDevice physicalDevice;
};

static void * vulkan_decompress_init(void)
{
	//validation layers, queue, nvidia examples
	printf("vulkan_decode - init\n");

	struct state_vulkan_decompress *s = calloc(1, sizeof(struct state_vulkan_decompress));
	if (!s) return NULL;
	s->test = 13;

	const char vulkan_lib_filename[] = "vulkan-1.dll";
    HMODULE vulkanLib = LoadLibrary(vulkan_lib_filename);
	if (vulkanLib == NULL)
	{
		printf("[vulkan_decode] Vulkan file '%s' not found!\n", vulkan_lib_filename);
		free(s);
		return NULL;
	}
	//printf("[vulkan_decode] Vulkan file '%s' loaded.\n", vulkan_lib_filename);
	s->vulkanLib = vulkanLib;

	const char vulkan_proc_name[] = "vkGetInstanceProcAddr";
	PFN_vkGetInstanceProcAddr getInstanceProcAddr = (PFN_vkGetInstanceProcAddr)GetProcAddress(vulkanLib, vulkan_proc_name);
    if (getInstanceProcAddr == NULL) {

		printf("[vulkan_decode] Vulkan procedure '%s' not found!\n", vulkan_proc_name);
        FreeLibrary(vulkanLib);
		free(s);
        return NULL;
    }
	s->loader = getInstanceProcAddr;

	VkApplicationInfo appInfo = { 0 };
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "UltraGrid vulkan_decode";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

	VkInstanceCreateInfo createInfo = { 0 };
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo = &appInfo;
	//createInfo.enabledExtensionCount = 
	//createInfo.ppEnabledExtensionNames
	createInfo.enabledLayerCount = 0;

	PFN_vkCreateInstance vkCreateInstance = (PFN_vkCreateInstance)getInstanceProcAddr(NULL, "vkCreateInstance");
	assert(vkCreateInstance != NULL);
	int result = vkCreateInstance(&createInfo, NULL, &s->instance);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to create vulkan instance! Error: %d\n", result);
        FreeLibrary(vulkanLib);
		free(s);
		return NULL;
	}

	// checking for extensions
	PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties =
					(PFN_vkEnumerateInstanceExtensionProperties)getInstanceProcAddr(NULL, "vkEnumerateInstanceExtensionProperties");
	assert(vkEnumerateInstanceExtensionProperties != NULL);
	uint32_t extensions_count = 0;

	vkEnumerateInstanceExtensionProperties(NULL, &extensions_count, NULL);
	printf("extensions_count: %d\n", extensions_count);

	VkExtensionProperties *extensions = (VkExtensionProperties*)calloc(extensions_count, sizeof(VkExtensionProperties));
	if (extensions == NULL)
	{
		printf("[vulkan_decode] Failed to allocate array for extensions!\n");
	}
	else
	{
		vkEnumerateInstanceExtensionProperties(NULL, &extensions_count, extensions);

		for (uint32_t i = 0; i < extensions_count; ++i)
		{
			printf("\tExtension: '%s'\n", extensions[i].extensionName);
		}
	}
	free(extensions);

	//TODO device enumeration

	return s;
}

static void vulkan_decompress_done(void *state)
{
	printf("vulkan_decode - done\n");
	struct state_vulkan_decompress *s = (struct state_vulkan_decompress *)state;
	if (!s) return;

	printf("[vulkan_decode] Vulkan_decompress_done freeing state '%d'\n", s->test);
	
	PFN_vkDestroyInstance vkDestroyInstance = (PFN_vkDestroyInstance)s->loader(s->instance, "vkDestroyInstance");
	if (vkDestroyInstance != NULL) vkDestroyInstance(s->instance, NULL);

	FreeLibrary(s->vulkanLib);
	free(s);
}

static int vulkan_decompress_reconfigure(void *state, struct video_desc desc,
											 int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
	printf("vulkan_decode - reconfigure\n");
	UNUSED(state);
	UNUSED(desc);
	UNUSED(rshift);
	UNUSED(gshift);
	UNUSED(bshift);
	UNUSED(pitch);
	UNUSED(out_codec);
	//TODO
	return true;
}

static int vulkan_decompress_get_property(void *state, int property, void *val, size_t *len)
{
	printf("vulkan_decode - get_property\n");
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
	printf("vulkan_decode - get_priority\n");
	UNUSED(compression);
	UNUSED(internal);
	UNUSED(ugc);
	
	return 3; //TODO magic value
}

static decompress_status vulkan_decompress(void *state, unsigned char *dst, unsigned char *src,
                unsigned int src_len, int frame_seq, struct video_frame_callbacks *callbacks,
                struct pixfmt_desc *internal_prop)
{
	printf("vulkan_decode - decompress\n");
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
