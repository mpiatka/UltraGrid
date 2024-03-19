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

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "vulkan/vulkan.h"
//#include "vk_video/vulkan_video_codecs_common.h" //?

static PFN_vkDestroyInstance vkDestroyInstance = NULL;
static PFN_vkDestroyDevice vkDestroyDevice = NULL;

static bool load_vulkan_functions_with_instance(PFN_vkGetInstanceProcAddr loader, VkInstance instance)
{
	vkDestroyInstance = (PFN_vkDestroyInstance)loader(instance, "vkDestroyInstance");
	vkDestroyDevice = (PFN_vkDestroyDevice)loader(instance, "vkDestroyDevice");

	return vkDestroyInstance && vkDestroyDevice; 
}

struct state_vulkan_decompress
{
	HMODULE vulkanLib;					// needs to be destroyed if valid
	VkInstance instance; 				// needs to be destroyed if valid
	PFN_vkGetInstanceProcAddr loader;
	VkPhysicalDevice physicalDevice;
	VkDevice device;					// needs to be destroyed if valid
	VkQueue queue;

	// UltraGrid
	//video_desc desc;
	int rshift;
	int gshift;
	int bshift;
	int pitch;
	codec_t out_codec;
};

static VkPhysicalDevice choose_physical_device(struct state_vulkan_decompress *s,
										VkPhysicalDevice devices[], uint32_t devices_count,
										uint32_t *queue_family_idx, VkQueueFamilyProperties2 *queue_family)
{
	// chooses the preferred physical device from array of given devices (of length atleast 1)
	// if queue_family and queue_family_idx pointers are non-NULL it will fill it with queue family properties
	// of preferred queue of the chosen device and the family index
	// returns VK_NULL_HANDLE and does not set queue_family if no suitable physical device found
	assert(devices_count > 0);
	assert((queue_family && queue_family_idx) || (!queue_family && !queue_family_idx)); //both must be NULL or both non-NULL

	PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties =
					(PFN_vkGetPhysicalDeviceProperties)s->loader(s->instance, "vkGetPhysicalDeviceProperties");
	assert(vkGetPhysicalDeviceProperties != NULL);
	PFN_vkGetPhysicalDeviceFeatures vkGetPhysicalDeviceFeatures =
					(PFN_vkGetPhysicalDeviceFeatures)s->loader(s->instance, "vkGetPhysicalDeviceFeatures");
	assert(vkGetPhysicalDeviceFeatures != NULL);
	PFN_vkGetPhysicalDeviceQueueFamilyProperties2 vkGetPhysicalDeviceQueueFamilyProperties2 =
					(PFN_vkGetPhysicalDeviceQueueFamilyProperties2)s->loader(s->instance, "vkGetPhysicalDeviceQueueFamilyProperties2");
	assert(vkGetPhysicalDeviceQueueFamilyProperties2 != NULL);

	VkPhysicalDevice chosen = VK_NULL_HANDLE;

	for (uint32_t i = 0; i < devices_count; ++i)
	{
		VkPhysicalDeviceProperties deviceProperties;
		vkGetPhysicalDeviceProperties(devices[i], &deviceProperties);
		printf("Device %d: '%s'\n", deviceProperties.deviceID, deviceProperties.deviceName);

		//TODO deviceFeatures
		//VkPhysicalDeviceFeatures deviceFeatures;
		//vkGetPhysicalDeviceFeatures(devices[i], &deviceFeatures);

		uint32_t queues_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties2(devices[i], &queues_count, NULL);

		if (queues_count == 0)
		{
			printf("\t No queue families found for this device.\n");
			continue;
		}
		
		VkQueueFamilyProperties2 *properties = (VkQueueFamilyProperties2*)calloc(queues_count, sizeof(VkQueueFamilyProperties2));
		VkQueueFamilyVideoPropertiesKHR *video_properties = (VkQueueFamilyVideoPropertiesKHR*)calloc(queues_count, sizeof(VkQueueFamilyVideoPropertiesKHR));
		if (properties == NULL || video_properties == NULL) //TODO probably return error?
		{
			printf("[vulkan_decode] Failed to allocate propertie and/or video_properties arrays!\n");
			free(properties);
			free(video_properties);
			break;
		}

		for (uint32_t j = 0; j < queues_count; ++j)
		{
			video_properties[j] = (VkQueueFamilyVideoPropertiesKHR){ .sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR,
																	 .pNext = NULL };
			properties[j] = (VkQueueFamilyProperties2){ .sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2,
														.pNext = (void*)&video_properties[j] };
		}

		vkGetPhysicalDeviceQueueFamilyProperties2(devices[i], &queues_count, properties);

		int hasDecode = 0;
		uint32_t preferred_queue_family = 0;
		for (uint32_t j = 0; j < queues_count; ++j)
		{
			VkQueueFlags flags = properties[j].queueFamilyProperties.queueFlags;
			int encode = flags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR ? 1 : 0;
			int decode = flags & VK_QUEUE_VIDEO_DECODE_BIT_KHR ? 1 : 0;
			printf("\tflags: %d, encode: %d, decode: %d\n", flags, encode, decode);

			if (decode) preferred_queue_family = j;
			hasDecode = hasDecode || decode;
		}

		for (uint32_t j = 0; j < queues_count; ++j)
		{
			VkVideoCodecOperationFlagsKHR flags = video_properties[j].videoCodecOperations;
			int h264_en = flags & VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR ? 1 : 0;
			int h265_en = flags & VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR ? 1 : 0;
			int h264_de = flags & VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR ? 1 : 0;
			int h265_de = flags & VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR ? 1 : 0;
			//int av1_de = flags & VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR ? 1 : 0;
			printf("\tvideo flags: %d, h264: %d %d, h265: %d %d\n", flags, h264_en, h264_de, h265_en, h265_de);
		}
		
		free(properties);
		free(video_properties);
		if (hasDecode)
		{
			chosen = devices[i];
			if (queue_family != NULL && queue_family_idx != NULL)
			{
				*queue_family_idx = preferred_queue_family;
				*queue_family = properties[preferred_queue_family];
			}
		}
	}

	return chosen;
}

static void * vulkan_decompress_init(void)
{
	//validation layers, queue, nvidia examples
	printf("vulkan_decode - init\n");

	// ---Allocation of the vulkan_decompress state---
	struct state_vulkan_decompress *s = calloc(1, sizeof(struct state_vulkan_decompress));
	if (!s)
	{
		printf("[vulkan_decode] Couldn't allocate memory for state struct!\n");
		return NULL;
	}

	// ---Dynamic loading of the vulkan loader library---
	const char vulkan_lib_filename[] = "vulkan-1.dll"; //TODO .so
    HMODULE vulkanLib = LoadLibrary(vulkan_lib_filename);
	if (vulkanLib == NULL)
	{
		printf("[vulkan_decode] Vulkan loader file '%s' not found!\n", vulkan_lib_filename);
		free(s);
		return NULL;
	}
	//printf("[vulkan_decode] Vulkan file '%s' loaded.\n", vulkan_lib_filename);
	s->vulkanLib = vulkanLib;

	// ---Getting the loader function---
	const char vulkan_proc_name[] = "vkGetInstanceProcAddr";
	PFN_vkGetInstanceProcAddr getInstanceProcAddr = (PFN_vkGetInstanceProcAddr)GetProcAddress(vulkanLib, vulkan_proc_name);
    if (getInstanceProcAddr == NULL) {

		printf("[vulkan_decode] Vulkan function '%s' not found!\n", vulkan_proc_name);
        FreeLibrary(vulkanLib);
		free(s);
        return NULL;
    }
	s->loader = getInstanceProcAddr;
	
	// ---Creating the vulkan instance---
	VkApplicationInfo appInfo = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
								  .pApplicationName = "UltraGrid vulkan_decode",
								  .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
								  .pEngineName = "No Engine",
								  .engineVersion = VK_MAKE_VERSION(1, 0, 0),
								  .apiVersion = VK_API_VERSION_1_0 };
	VkInstanceCreateInfo createInfo = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
										.pApplicationInfo = &appInfo,
										.enabledExtensionCount = 0,
										.enabledLayerCount = 0 };

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

	if (!load_vulkan_functions_with_instance(getInstanceProcAddr, s->instance))
	{
		printf("[vulkan_decode] Failed to load all vulkan functions!\n");
		if (vkDestroyInstance != NULL) vkDestroyInstance(s->instance, NULL);
		FreeLibrary(vulkanLib);
		free(s);
        return NULL;
	}

	// ---Checking for extensions---
	PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties =
					(PFN_vkEnumerateInstanceExtensionProperties)getInstanceProcAddr(s->instance, "vkEnumerateInstanceExtensionProperties");
	assert(vkEnumerateInstanceExtensionProperties != NULL);
	uint32_t extensions_count = 0;

	vkEnumerateInstanceExtensionProperties(NULL, &extensions_count, NULL);
	printf("extensions_count: %d\n", extensions_count);
	//TODO count == 0

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

	// ---Choosing of physical device---
	PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices =
						(PFN_vkEnumeratePhysicalDevices)getInstanceProcAddr(s->instance, "vkEnumeratePhysicalDevices");
	assert(vkEnumeratePhysicalDevices != NULL);
	PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties =
						(PFN_vkGetPhysicalDeviceProperties)getInstanceProcAddr(s->instance, "vkGetPhysicalDeviceProperties");
	assert(vkGetPhysicalDeviceProperties != NULL);
	uint32_t phys_devices_count = 0;
	//TODO count == 0

	vkEnumeratePhysicalDevices(s->instance, &phys_devices_count, NULL);
	printf("phys_devices_count: %d\n", phys_devices_count);

	VkPhysicalDevice *devices = (VkPhysicalDevice*)calloc(phys_devices_count, sizeof(VkPhysicalDevice));
	if (devices == NULL)
	{
		printf("[vulkan_decode] Failed to allocate array for devices!\n");
		vkDestroyInstance(s->instance, NULL);
        FreeLibrary(vulkanLib);
		free(s);
		return NULL;
	}

	vkEnumeratePhysicalDevices(s->instance, &phys_devices_count, devices);
	uint32_t chosen_queue_family_idx = 0;
	VkQueueFamilyProperties2 chosen_queue_family; //TODO use this
	s->physicalDevice = choose_physical_device(s, devices, phys_devices_count, &chosen_queue_family_idx, &chosen_queue_family);
	free(devices);
	if (s->physicalDevice == VK_NULL_HANDLE)
	{
		printf("[vulkan_decode] Failed to choose a appropriate physical device!\n");
		vkDestroyInstance(s->instance, NULL);
        FreeLibrary(vulkanLib);
		free(s);
		return NULL;
	}
	assert(chosen_queue_family.pNext != NULL &&
		   ((VkQueueFamilyVideoPropertiesKHR*)chosen_queue_family.pNext)->sType == VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR);

	VkPhysicalDeviceProperties deviceProperties;
	vkGetPhysicalDeviceProperties(s->physicalDevice, &deviceProperties);
	printf("Chosen physical device is: '%s' and chosen queue family index is: %d\n", 
				deviceProperties.deviceName, chosen_queue_family_idx);

	// ---Creating a logical device---
	PFN_vkCreateDevice vkCreateDevice = (PFN_vkCreateDevice)getInstanceProcAddr(s->instance, "vkCreateDevice");
	assert(vkCreateDevice != NULL);

	float queue_priorities = 1.0f;
	VkDeviceQueueCreateInfo queueCreateInfo = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
												.queueFamilyIndex = chosen_queue_family_idx,
												.queueCount = 1,
												.pQueuePriorities = &queue_priorities };
	VkPhysicalDeviceFeatures deviceFeatures = { 0 }; //TODO deviceFeatures
	VkDeviceCreateInfo createDeviceInfo = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
											.queueCreateInfoCount = 1,
											.pQueueCreateInfos = &queueCreateInfo,
											.pEnabledFeatures = &deviceFeatures,
											.enabledLayerCount = 0, .enabledExtensionCount = 0 };
	
	result = vkCreateDevice(s->physicalDevice, &createDeviceInfo, NULL, &s->device);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to create a appropriate vulkan device!\n");
		vkDestroyInstance(s->instance, NULL);
		free(s);
        FreeLibrary(vulkanLib);
		return NULL;
	}

	PFN_vkGetDeviceQueue vkGetDeviceQueue = (PFN_vkGetDeviceQueue)getInstanceProcAddr(s->instance, "vkGetDeviceQueue");
	assert(vkGetDeviceQueue != NULL);

	vkGetDeviceQueue(s->device, queueCreateInfo.queueFamilyIndex, 0, &s->queue);

	printf("[vulkan_decode] Initialization finished successfully.\n");
	return s;
}

static void vulkan_decompress_done(void *state)
{
	printf("vulkan_decode - done\n");
	struct state_vulkan_decompress *s = (struct state_vulkan_decompress *)state;
	if (!s) return;

	if (vkDestroyDevice != NULL) vkDestroyDevice(s->device, NULL);

	if (vkDestroyInstance != NULL) vkDestroyInstance(s->instance, NULL);

	FreeLibrary(s->vulkanLib);

	free(s);
}

static bool configure_with(struct state_vulkan_decompress *s, struct video_desc desc)
{
	printf("vulkan_decode - configure_with\n");
	UNUSED(s);
	UNUSED(desc);
	//TODO
	return true;
}

static int vulkan_decompress_reconfigure(void *state, struct video_desc desc,
											 int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
	printf("vulkan_decode - reconfigure\n");
	UNUSED(desc);
	UNUSED(rshift);
	UNUSED(gshift);
	UNUSED(bshift);
	UNUSED(pitch);
	UNUSED(out_codec);
	struct state_vulkan_decompress *s = (struct state_vulkan_decompress *)state;
	if (!s) return false;

	//TODO
	const char *spec_name = get_codec_name(desc.color_spec);
	const char *out_name = get_codec_name(out_codec);
	printf("\tcodec color_spec: '%s', out_codec: '%s'\n", spec_name, out_name);

	//s->desc = desc;
	s->rshift = rshift;
	s->gshift = gshift;
	s->bshift = bshift;
	s->pitch = pitch;
	s->out_codec = out_codec;

	return configure_with(s, desc);
}

static int vulkan_decompress_get_property(void *state, int property, void *val, size_t *len)
{
	printf("vulkan_decode - get_property\n");
	struct state_vulkan_decompress *s = (struct state_vulkan_decompress *)state;
    UNUSED(s);
	//printf("\tproperty index: %d, val: %p, len: %p\n", property, val, len);
	
	int ret = FALSE;
	switch(property) {
		case DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME:
			if(*len >= sizeof(int))
			{
				*(int *) val = FALSE;
				*len = sizeof(int);
				ret = TRUE;
			}
			break;
		default:
				ret = FALSE;
	}

	return ret;
}


static int vulkan_decompress_get_priority(codec_t compression, struct pixfmt_desc internal, codec_t ugc)
{
	printf("vulkan_decode - get_priority\n");
	UNUSED(compression);
	UNUSED(internal);
	UNUSED(ugc);
	const char *compression_name = get_codec_name(compression);
	const char *ugc_name = get_codec_name(ugc);
	printf("\tcompression: '%s', ugc: '%s'\n", compression_name, ugc_name);
	
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
