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

#include "utils/bs.h"
#include "rtp/rtpdec_h264.h"	//?
#include "rtp/rtpenc_h264.h"	//?

//#include <windows.h> //LoadLibrary

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
//TODO add into configure.ac
#include "vulkan/vulkan.h"
#include "vk_video/vulkan_video_codec_h264std_decode.h"
#include "vulkan_decode_h264.h"

// activates vulkan validation layers if defined
// if defined your vulkan loader needs to know where to find the validation layer manifest
// (for example through VK_LAYER_PATH or VK_ADD_LAYER_PATH env. variables)
#define VULKAN_VALIDATE

// one of value from enum VkDebugUtilsMessageSeverityFlagBitsEXT included from vulkan.h
// (definition in vulkan_core.h)
#define VULKAN_VALIDATE_SHOW_SEVERITY VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
//#define VULKAN_VALIDATE_SHOW_SEVERITY VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT
//#define VULKAN_VALIDATE_SHOW_SEVERITY VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT

#define MAX_REF_FRAMES 16
#define MAX_SLICES 128

static PFN_vkCreateInstance vkCreateInstance = NULL;
static PFN_vkDestroyInstance vkDestroyInstance = NULL;
#ifdef VULKAN_VALIDATE
static PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT = NULL;
static PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT = NULL;
#endif
static PFN_vkGetPhysicalDeviceFeatures vkGetPhysicalDeviceFeatures = NULL;
static PFN_vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFeatures2 = NULL;
static PFN_vkGetPhysicalDeviceQueueFamilyProperties2 vkGetPhysicalDeviceQueueFamilyProperties2 = NULL;
static PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = NULL;
static PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = NULL;
static PFN_vkGetPhysicalDeviceProperties2KHR vkGetPhysicalDeviceProperties2KHR = NULL;
static PFN_vkCreateDevice vkCreateDevice = NULL;
static PFN_vkDestroyDevice vkDestroyDevice = NULL;
static PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties = NULL;
static PFN_vkEnumerateInstanceLayerProperties vkEnumerateInstanceLayerProperties = NULL;
static PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties = NULL;
static PFN_vkGetDeviceQueue vkGetDeviceQueue = NULL;
static PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = NULL;
static PFN_vkAllocateMemory vkAllocateMemory = NULL;
static PFN_vkFreeMemory vkFreeMemory = NULL;
static PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR vkGetPhysicalDeviceVideoCapabilitiesKHR = NULL;
static PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR vkGetPhysicalDeviceVideoFormatPropertiesKHR = NULL;
static PFN_vkCreateBuffer vkCreateBuffer = NULL;
static PFN_vkDestroyBuffer vkDestroyBuffer = NULL;
static PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = NULL;
static PFN_vkBindBufferMemory vkBindBufferMemory = NULL;
static PFN_vkCreateCommandPool vkCreateCommandPool = NULL;
static PFN_vkDestroyCommandPool vkDestroyCommandPool = NULL;
static PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = NULL;
static PFN_vkBeginCommandBuffer vkBeginCommandBuffer = NULL;
static PFN_vkEndCommandBuffer vkEndCommandBuffer = NULL;
static PFN_vkResetCommandBuffer vkResetCommandBuffer = NULL;
static PFN_vkCreateVideoSessionKHR vkCreateVideoSessionKHR = NULL;
static PFN_vkDestroyVideoSessionKHR vkDestroyVideoSessionKHR = NULL;
static PFN_vkGetVideoSessionMemoryRequirementsKHR vkGetVideoSessionMemoryRequirementsKHR = NULL;
static PFN_vkBindVideoSessionMemoryKHR vkBindVideoSessionMemoryKHR = NULL;
static PFN_vkCreateVideoSessionParametersKHR vkCreateVideoSessionParametersKHR = NULL;
static PFN_vkDestroyVideoSessionParametersKHR vkDestroyVideoSessionParametersKHR = NULL;
static PFN_vkUpdateVideoSessionParametersKHR vkUpdateVideoSessionParametersKHR = NULL;
static PFN_vkCmdBeginVideoCodingKHR vkCmdBeginVideoCodingKHR = NULL;
static PFN_vkCmdEndVideoCodingKHR vkCmdEndVideoCodingKHR = NULL;
static PFN_vkCmdControlVideoCodingKHR vkCmdControlVideoCodingKHR = NULL;
static PFN_vkMapMemory vkMapMemory = NULL;
static PFN_vkUnmapMemory vkUnmapMemory = NULL;
static PFN_vkQueueSubmit vkQueueSubmit = NULL;
static PFN_vkCreateFence vkCreateFence = NULL;
static PFN_vkDestroyFence vkDestroyFence = NULL;
static PFN_vkGetFenceStatus vkGetFenceStatus = NULL;
static PFN_vkResetFences vkResetFences = NULL;
static PFN_vkWaitForFences vkWaitForFences = NULL;
static PFN_vkCreateImage vkCreateImage = NULL;
static PFN_vkDestroyImage vkDestroyImage = NULL;
static PFN_vkGetImageMemoryRequirements vkGetImageMemoryRequirements = NULL;
static PFN_vkBindImageMemory vkBindImageMemory = NULL;
static PFN_vkCreateImageView vkCreateImageView = NULL;
static PFN_vkDestroyImageView vkDestroyImageView = NULL;
static PFN_vkCmdDecodeVideoKHR vkCmdDecodeVideoKHR = NULL;
static PFN_vkCmdPipelineBarrier2KHR vkCmdPipelineBarrier2KHR = NULL;
static PFN_vkCmdCopyImageToBuffer vkCmdCopyImageToBuffer = NULL;
static PFN_vkGetPhysicalDeviceFormatProperties vkGetPhysicalDeviceFormatProperties = NULL;

static bool load_vulkan_functions_globals(PFN_vkGetInstanceProcAddr loader)
{
	vkCreateInstance = (PFN_vkCreateInstance)loader(NULL, "vkCreateInstance");
	vkEnumerateInstanceExtensionProperties = (PFN_vkEnumerateInstanceExtensionProperties)
												loader(NULL, "vkEnumerateInstanceExtensionProperties");
	vkEnumerateInstanceLayerProperties = (PFN_vkEnumerateInstanceLayerProperties)
												loader(NULL, "vkEnumerateInstanceLayerProperties");
	
	return vkCreateInstance &&
		   vkEnumerateInstanceExtensionProperties && vkEnumerateInstanceLayerProperties;
}

static bool load_vulkan_functions_with_instance(PFN_vkGetInstanceProcAddr loader, VkInstance instance)
{
	vkDestroyInstance = (PFN_vkDestroyInstance)loader(instance, "vkDestroyInstance");
	#ifdef VULKAN_VALIDATE
	vkCreateDebugUtilsMessengerEXT = (PFN_vkCreateDebugUtilsMessengerEXT)
										loader(instance, "vkCreateDebugUtilsMessengerEXT");
	vkDestroyDebugUtilsMessengerEXT = (PFN_vkDestroyDebugUtilsMessengerEXT)
										loader(instance, "vkDestroyDebugUtilsMessengerEXT");
	#endif
	vkGetPhysicalDeviceFeatures = (PFN_vkGetPhysicalDeviceFeatures)
										loader(instance, "vkGetPhysicalDeviceFeatures");
	vkGetPhysicalDeviceFeatures2 = (PFN_vkGetPhysicalDeviceFeatures2)
										loader(instance, "vkGetPhysicalDeviceFeatures2");
	vkGetPhysicalDeviceQueueFamilyProperties2 = (PFN_vkGetPhysicalDeviceQueueFamilyProperties2)
										loader(instance, "vkGetPhysicalDeviceQueueFamilyProperties2");
	vkEnumeratePhysicalDevices = (PFN_vkEnumeratePhysicalDevices)
										loader(instance, "vkEnumeratePhysicalDevices");
	vkGetPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)
										loader(instance, "vkGetPhysicalDeviceProperties");
	vkGetPhysicalDeviceProperties2KHR = (PFN_vkGetPhysicalDeviceProperties2KHR)
										loader(instance, "vkGetPhysicalDeviceProperties2KHR");
	vkCreateDevice = (PFN_vkCreateDevice)loader(instance, "vkCreateDevice");
	vkDestroyDevice = (PFN_vkDestroyDevice)loader(instance, "vkDestroyDevice");
	vkEnumerateDeviceExtensionProperties = (PFN_vkEnumerateDeviceExtensionProperties)
												loader(instance, "vkEnumerateDeviceExtensionProperties");
	vkGetDeviceQueue = (PFN_vkGetDeviceQueue)loader(instance, "vkGetDeviceQueue"); //USELESS
	vkGetPhysicalDeviceMemoryProperties = (PFN_vkGetPhysicalDeviceMemoryProperties)
												loader(instance, "vkGetPhysicalDeviceMemoryProperties");
	vkAllocateMemory = (PFN_vkAllocateMemory)loader(instance, "vkAllocateMemory");
	vkFreeMemory = (PFN_vkFreeMemory)loader(instance, "vkFreeMemory");
	vkGetPhysicalDeviceVideoCapabilitiesKHR = (PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR)
												loader(instance, "vkGetPhysicalDeviceVideoCapabilitiesKHR");
	vkGetPhysicalDeviceVideoFormatPropertiesKHR = (PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR)
												loader(instance, "vkGetPhysicalDeviceVideoFormatPropertiesKHR");
	vkCreateBuffer = (PFN_vkCreateBuffer)loader(instance, "vkCreateBuffer");
	vkDestroyBuffer = (PFN_vkDestroyBuffer)loader(instance, "vkDestroyBuffer");
	vkGetBufferMemoryRequirements = (PFN_vkGetBufferMemoryRequirements)
												loader(instance, "vkGetBufferMemoryRequirements");
	vkBindBufferMemory = (PFN_vkBindBufferMemory)loader(instance, "vkBindBufferMemory");
	vkCreateCommandPool = (PFN_vkCreateCommandPool)loader(instance, "vkCreateCommandPool");
	vkDestroyCommandPool = (PFN_vkDestroyCommandPool)loader(instance, "vkDestroyCommandPool");
	vkAllocateCommandBuffers = (PFN_vkAllocateCommandBuffers)loader(instance, "vkAllocateCommandBuffers");
	vkBeginCommandBuffer = (PFN_vkBeginCommandBuffer)loader(instance, "vkBeginCommandBuffer");
	vkEndCommandBuffer = (PFN_vkEndCommandBuffer)loader(instance, "vkEndCommandBuffer");
	vkResetCommandBuffer = (PFN_vkResetCommandBuffer)loader(instance, "vkResetCommandBuffer");
	vkCreateVideoSessionKHR = (PFN_vkCreateVideoSessionKHR)loader(instance, "vkCreateVideoSessionKHR");
	vkDestroyVideoSessionKHR = (PFN_vkDestroyVideoSessionKHR)loader(instance, "vkDestroyVideoSessionKHR");
	vkGetVideoSessionMemoryRequirementsKHR = (PFN_vkGetVideoSessionMemoryRequirementsKHR)
												loader(instance, "vkGetVideoSessionMemoryRequirementsKHR");
	vkBindVideoSessionMemoryKHR = (PFN_vkBindVideoSessionMemoryKHR)
												loader(instance, "vkBindVideoSessionMemoryKHR");
	vkCreateVideoSessionParametersKHR = (PFN_vkCreateVideoSessionParametersKHR)
												loader(instance, "vkCreateVideoSessionParametersKHR");
	vkDestroyVideoSessionParametersKHR = (PFN_vkDestroyVideoSessionParametersKHR)
												loader(instance, "vkDestroyVideoSessionParametersKHR");
	vkUpdateVideoSessionParametersKHR = (PFN_vkUpdateVideoSessionParametersKHR)
												loader(instance, "vkUpdateVideoSessionParametersKHR");
	vkCmdBeginVideoCodingKHR = (PFN_vkCmdBeginVideoCodingKHR)loader(instance, "vkCmdBeginVideoCodingKHR");
	vkCmdEndVideoCodingKHR = (PFN_vkCmdEndVideoCodingKHR)loader(instance, "vkCmdEndVideoCodingKHR");
	vkCmdControlVideoCodingKHR = (PFN_vkCmdControlVideoCodingKHR)
												loader(instance, "vkCmdControlVideoCodingKHR");
	vkMapMemory = (PFN_vkMapMemory)loader(instance, "vkMapMemory");
	vkUnmapMemory = (PFN_vkUnmapMemory)loader(instance, "vkUnmapMemory");
	vkQueueSubmit = (PFN_vkQueueSubmit)loader(instance, "vkQueueSubmit");
	vkCreateFence = (PFN_vkCreateFence)loader(instance, "vkCreateFence");
	vkDestroyFence = (PFN_vkDestroyFence)loader(instance, "vkDestroyFence");
	vkGetFenceStatus = (PFN_vkGetFenceStatus)loader(instance, "vkGetFenceStatus");
	vkResetFences = (PFN_vkResetFences)loader(instance, "vkResetFences");
	vkWaitForFences = (PFN_vkWaitForFences)loader(instance, "vkWaitForFences");
	vkCreateImage = (PFN_vkCreateImage)loader(instance, "vkCreateImage");
	vkDestroyImage = (PFN_vkDestroyImage)loader(instance, "vkDestroyImage");
	vkGetImageMemoryRequirements = (PFN_vkGetImageMemoryRequirements)
												loader(instance, "vkGetImageMemoryRequirements");
	vkBindImageMemory = (PFN_vkBindImageMemory)loader(instance, "vkBindImageMemory");
	vkCreateImageView = (PFN_vkCreateImageView)loader(instance, "vkCreateImageView");
	vkDestroyImageView = (PFN_vkDestroyImageView)loader(instance, "vkDestroyImageView");
	vkCmdDecodeVideoKHR = (PFN_vkCmdDecodeVideoKHR)loader(instance, "vkCmdDecodeVideoKHR");
	vkCmdCopyImageToBuffer = (PFN_vkCmdCopyImageToBuffer)loader(instance, "vkCmdCopyImageToBuffer");
	vkCmdPipelineBarrier2KHR = (PFN_vkCmdPipelineBarrier2)loader(instance, "vkCmdPipelineBarrier2KHR");
	vkGetPhysicalDeviceFormatProperties = (PFN_vkGetPhysicalDeviceFormatProperties)
												loader(instance, "vkGetPhysicalDeviceFormatProperties");

	return vkDestroyInstance &&
	#ifdef VULKAN_VALIDATE
		   vkCreateDebugUtilsMessengerEXT && vkDestroyDebugUtilsMessengerEXT &&
	#endif
		   vkGetPhysicalDeviceFeatures && vkGetPhysicalDeviceFeatures2 &&
		   vkGetPhysicalDeviceQueueFamilyProperties2 && vkEnumeratePhysicalDevices &&
		   vkGetPhysicalDeviceProperties && vkGetPhysicalDeviceProperties2KHR &&
		   vkCreateDevice && vkDestroyDevice &&
		   vkEnumerateDeviceExtensionProperties &&
		   vkGetDeviceQueue &&
		   vkGetPhysicalDeviceMemoryProperties &&
		   vkAllocateMemory && vkFreeMemory &&
		   vkGetPhysicalDeviceVideoCapabilitiesKHR &&
		   vkGetPhysicalDeviceVideoFormatPropertiesKHR &&
		   vkCreateBuffer && vkDestroyBuffer &&
		   vkGetBufferMemoryRequirements && vkBindBufferMemory &&
		   vkCreateCommandPool && vkDestroyCommandPool &&
		   vkAllocateCommandBuffers &&
		   vkBeginCommandBuffer && vkEndCommandBuffer &&
		   vkResetCommandBuffer &&
		   vkCreateVideoSessionKHR && vkDestroyVideoSessionKHR &&
		   vkGetVideoSessionMemoryRequirementsKHR && vkBindVideoSessionMemoryKHR &&
		   vkCreateVideoSessionParametersKHR && vkDestroyVideoSessionParametersKHR &&
		   vkUpdateVideoSessionParametersKHR &&
		   vkCmdBeginVideoCodingKHR && vkCmdEndVideoCodingKHR &&
		   vkCmdControlVideoCodingKHR &&
		   vkMapMemory && vkUnmapMemory &&
		   vkQueueSubmit &&
		   vkCreateFence && vkDestroyFence&&
		   vkGetFenceStatus && vkResetFences &&
		   vkWaitForFences &&
		   vkCreateImage && vkDestroyImage && 
		   vkGetImageMemoryRequirements && vkBindImageMemory &&
		   vkCreateImageView && vkDestroyImageView &&
		   vkCmdDecodeVideoKHR &&
		   vkCmdCopyImageToBuffer && vkCmdPipelineBarrier2KHR &&
		   vkGetPhysicalDeviceFormatProperties;
}

typedef struct	// structure used to pass around variables related to currently decoded frame (slice)
{
	bool is_intra, is_reference;
	int idr_pic_id;
	int sps_id;
	int pps_id;
	int frame_num;
	int frame_seq; // Ultragrid's frame_seq parameter in decompress function
	int poc, poc_lsb;
	uint32_t dpbIndex;

} slice_info_t;

struct state_vulkan_decompress
{
	HMODULE vulkanLib;							// needs to be destroyed if valid
	VkInstance instance; 						// needs to be destroyed if valid
	PFN_vkGetInstanceProcAddr loader;
	//maybe this could be present only when VULKAN_VALIDATE is defined?
	VkDebugUtilsMessengerEXT debugMessenger;	// needs to be destroyed if valid
	VkDeviceSize physDeviceMaxBuffSize;
	VkPhysicalDevice physicalDevice;
	VkDevice device;							// needs to be destroyed if valid
	uint32_t queueFamilyIdx;
	VkQueue decodeQueue;
	VkVideoCodecOperationFlagsKHR queueVideoFlags;
	VkVideoCodecOperationFlagsKHR codecOperation;
	bool prepared, sps_vps_found, resetVideoCoding;
	VkFence fence;
	VkBuffer decodeBuffer;						// needs to be destroyed if valid
	VkDeviceSize decodeBufferSize;
	VkDeviceSize decodeBufferOffsetAlignment;
	VkBuffer dstPicBuffer;						// needs to be destroyed if valid
	VkDeviceSize dstPicBufferSize;
	VkDeviceSize dstPicBufferMemoryOffset;		// offset of dstPicBuffer in the bufferMemory
	VkDeviceMemory bufferMemory;				// allocated memory fot both decodeBuffer and dstPicBuffer, needs to be freed if valid
	VkCommandPool commandPool;					// needs to be destroyed if valid
	VkCommandBuffer cmdBuffer;
	VkVideoSessionKHR videoSession;				// needs to be destroyed if valid
	uint32_t videoSessionMemory_count;
	// array of size videoSessionMemory_count, needs to be freed and VkvideoSessionMemory deallocated
	VkDeviceMemory *videoSessionMemory;

	// Parameters of the video:
	int depth_chroma, depth_luma;
	int subsampling; // in the Ultragrid format
	StdVideoH264ProfileIdc profileIdc; //TODO H.265
	VkVideoSessionParametersKHR videoSessionParams; // needs to be destroyed if valid
	//uint32_t videoSessionParams_update_count;
	// pointers to arrays of sps (length MAX_SPS_IDS), pps (length MAX_PPS_IDS)
	// could be static arrays but that would take too much memory of this struct
	sps_t *sps_array; //TODO maybe better to store already converted vulkan version?
	pps_t *pps_array;

	// Memory related to decode picture buffer and picture queue
	bool dpbHasDefinedLayout;					// indicates that VkImages in 'dpb' array are not in undefined layout
	VkImage dpb[MAX_REF_FRAMES + 1];			// decoded picture buffer (aka dpb)
	VkImageView dpbViews[MAX_REF_FRAMES + 1];	// dpb image views
	VkDeviceMemory dpbMemory;					// backing memory for dpb - needs to be freed if valid (destroyed in destroy_dpb)
	VkFormat dpbFormat;							// format of VkImages in dpb
	//uint32_t dpbDstPictureIdx;					// index (into dpb and dpbViews) of the slot for next to be decoded frame
	slice_info_t referenceSlotsQueue[MAX_REF_FRAMES];	// queue containing slice infos of the current reference frames 
	uint32_t referenceSlotsQueue_start;			  		// index into referenceSlotsQueue where the queue starts
	uint32_t referenceSlotsQueue_count;			  		// the current length of the reference slots queue

	int prev_poc_lsb, prev_poc_msb, idr_frame_seq, current_frame_seq;

	// UltraGrid data
	int width, height;
	int pitch; //USELESS ?
	codec_t out_codec;
};

static void free_buffers(struct state_vulkan_decompress *s);
static void destroy_dpb(struct state_vulkan_decompress *s);

static bool check_for_instance_extensions(const char * const requiredInstanceextensions[])
{
	/*const char* const requiredInstanceLayerExtensions[] = {
								"VK_LAYER_KHRONOS_validation",
								NULL };

    const char* const requiredWsiInstanceExtensions[] = {
								// Required generic WSI extensions
								VK_KHR_SURFACE_EXTENSION_NAME,
								NULL };*/
	
	/*const char* const requiredWsiDeviceExtension[] = {
							// Add the WSI required device extensions
							VK_KHR_SWAPCHAIN_EXTENSION_NAME,
							NULL };

    static const char* const optinalDeviceExtension[] = {
								VK_EXT_YCBCR_2PLANE_444_FORMATS_EXTENSION_NAME,
								VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME,
								VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
								VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
								NULL };*/

	uint32_t extensions_count = 0;
	vkEnumerateInstanceExtensionProperties(NULL, &extensions_count, NULL);
	printf("instance extensions_count: %d\n", extensions_count);
	if (extensions_count == 0)
	{
		if (requiredInstanceextensions[0] != NULL)
		{
			printf("[vulkan_decode] No instance extensions supported.\n");
			return false;
		}
		
		printf("No instance extensions found and none required.\n");
		return true;
	}

	VkExtensionProperties *extensions = (VkExtensionProperties*)calloc(extensions_count, sizeof(VkExtensionProperties));
	if (extensions == NULL)
	{
		printf("[vulkan_decode] Failed to allocate array for instance extensions!\n");
		return false;
	}

	vkEnumerateInstanceExtensionProperties(NULL, &extensions_count, extensions);

	//IDEA verbose setting
	// Printing of all found instance extentions
	//for (uint32_t i = 0; i < extensions_count; ++i) printf("\tExtension: '%s'\n", extensions[i].extensionName);

	// Checking for required ones
	for (size_t i = 0; requiredInstanceextensions[i] != NULL; ++i)
	{
		bool found = false;
		for (uint32_t j = 0; j < extensions_count; ++j)
		{
			if (!strcmp(requiredInstanceextensions[i], extensions[j].extensionName))
			{
				found = true;
				break;
			}
		}

		if (!found)
		{
			printf("[vulkan_decode] Required instance extension: '%s' was not found!\n",
						requiredInstanceextensions[i]);
			free(extensions);
			return false;
		}
	}

	free(extensions);
	return true;
}

#ifdef VULKAN_VALIDATE
static bool check_for_validation_layers(const char * const validationLayers[])
{
	uint32_t properties_count = 0;
	VkResult result = vkEnumerateInstanceLayerProperties(&properties_count, NULL);
	if (result != VK_SUCCESS) return false;

	VkLayerProperties *properties = (VkLayerProperties*)calloc(properties_count, sizeof(VkLayerProperties));
	if (!properties) return false;

	result = vkEnumerateInstanceLayerProperties(&properties_count, properties);
	if (result != VK_SUCCESS) return false;

	//IDEA verbose setting
	// Printing of all found layers extentions
	//for (uint32_t i = 0; i < properties_count; ++i) printf("\tLayer: '%s' desc: '%s'\n",
	//		properties[i].layerName, properties[i].description);

	for (size_t i = 0; validationLayers[i]; ++i)
	{
		bool found = false;

		for (uint32_t j = 0; j < properties_count; ++j)
		{
			if (!strcmp(validationLayers[i], properties[j].layerName))
			{
				found = true;
				break;
			}
		}

		if (!found)
		{
			free(properties);
			return false;
		}
	}

	free(properties);
	return true;
}
#endif

static bool check_for_device_extensions(VkPhysicalDevice physDevice, const char* const requiredDeviceExtensions[])
{
	uint32_t extensions_count = 0;
	vkEnumerateDeviceExtensionProperties(physDevice, NULL, &extensions_count, NULL);
	printf("device extensions_count: %d\n", extensions_count);
	if (extensions_count == 0)
	{
		if (requiredDeviceExtensions[0] != NULL)
		{
			printf("[vulkan_decode] No device extensions supported.\n");
			return false;
		}
		
		printf("No device extensions found and none required.\n");
		return true;
	}

	VkExtensionProperties *extensions = (VkExtensionProperties*)calloc(extensions_count, sizeof(VkExtensionProperties));
	if (extensions == NULL)
	{
		printf("[vulkan_decode] Failed to allocate array for device extensions!\n");
		return false;
	}

	vkEnumerateDeviceExtensionProperties(physDevice, NULL, &extensions_count, extensions);

	//IDEA verbose setting
	// Printing of all found device extentions
	//for (uint32_t i = 0; i < extensions_count; ++i) printf("\tExtension: '%s'\n", extensions[i].extensionName);

	// Checking for required ones
	for (size_t i = 0; requiredDeviceExtensions[i] != NULL; ++i)
	{
		bool found = false;
		for (uint32_t j = 0; j < extensions_count; ++j)
		{
			if (!strcmp(requiredDeviceExtensions[i], extensions[j].extensionName))
			{
				found = true;
				break;
			}
		}

		if (!found)
		{
			printf("[vulkan_decode] Required device extension: '%s' was not found!\n",
						requiredDeviceExtensions[i]);
			free(extensions);
			return false;
		}
	}

	free(extensions);
	return true;
}

#ifdef VULKAN_VALIDATE
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
								VkDebugUtilsMessageSeverityFlagBitsEXT           messageSeverity,
								VkDebugUtilsMessageTypeFlagsEXT                  messageTypes,
								const VkDebugUtilsMessengerCallbackDataEXT*      pCallbackData,
								void*                                            pUserData)
{
	UNUSED(messageTypes);
	UNUSED(pUserData);

	if (messageSeverity >= VULKAN_VALIDATE_SHOW_SEVERITY)
	{
		printf("VULKAN VALIDATION: '%s'\n", pCallbackData->pMessage);
	}

	return VK_FALSE;
}

static VkResult create_debug_messenger(VkInstance instance, VkDebugUtilsMessengerEXT *debugMessenger_ptr)
{
	VkDebugUtilsMessengerCreateInfoEXT messengerCreateInfo = { .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
												   			   .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
								 				   			 	VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
															 	VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
															 	VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
								 				   			   .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
												    		 	VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
															 	VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
												   			    .pfnUserCallback = debug_callback };

	return vkCreateDebugUtilsMessengerEXT(instance, &messengerCreateInfo, NULL, debugMessenger_ptr);
}
#endif

static void destroy_debug_messenger(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger)
{
	//destroys debug messenger when VULKAN_VALIDATE is defined and destructor exists
	//this function is defined even when VULKAN_VALIDATE is not just to avoid too many
	//ifdefs everywhere where destroying needs to happen
	#ifdef VULKAN_VALIDATE
	if (vkDestroyDebugUtilsMessengerEXT != NULL && instance != VK_NULL_HANDLE)
				vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, NULL);
	#else
	UNUSED(instance);
	UNUSED(debugMessenger);
	#endif
}

static VkPhysicalDevice choose_physical_device(VkPhysicalDevice devices[], uint32_t devices_count,
											   VkQueueFlags requestedQueueFamilyFlags,
											   const char* const requiredDeviceExtensions[],
											   uint32_t *queue_family_idx, VkQueueFamilyProperties2 *queue_family,
											   VkQueueFamilyVideoPropertiesKHR *queue_video_props)
{
	// chooses the preferred physical device from array of given devices (of length atleast 1)
	// queue_family_idx and if queue_family and queue_video_props pointers are non-NULL it will fill
	// them with queue family properties and video properties of preferred queue of the chosen device and the family index
	// returns VK_NULL_HANDLE and does not set queue_family if no suitable physical device found
	assert(devices_count > 0);

	VkPhysicalDevice chosen = VK_NULL_HANDLE;

	for (uint32_t i = 0; i < devices_count; ++i)
	{
		VkPhysicalDeviceProperties deviceProperties;
		vkGetPhysicalDeviceProperties(devices[i], &deviceProperties);
		printf("Device %d: '%s'\n", deviceProperties.deviceID, deviceProperties.deviceName);

		if (!check_for_device_extensions(devices[i], requiredDeviceExtensions))
		{
			printf("\tDevice does not have required extensions.\n");
			continue;
		}

		VkPhysicalDeviceVulkan13Features deviceFeatures13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
		VkPhysicalDeviceFeatures2 deviceFeatures = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
													 .pNext = (void*)&deviceFeatures13 };
		vkGetPhysicalDeviceFeatures2(devices[i], &deviceFeatures);

		//TODO better check, maybe make it a parameter?
		if (!deviceFeatures13.synchronization2 || !deviceFeatures13.maintenance4)
		{
			printf("\tDevice does not have required features (synchronization2 and maintenance4)!\n");
			continue;
		}

		uint32_t queues_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties2(devices[i], &queues_count, NULL);

		if (queues_count == 0)
		{
			printf("\tNo queue families found for this device.\n");
			continue;
		}
		
		//IDEA maybe we can use MAX_QUEUE_FAMILIES (6?) instead of dynamic allocation
		VkQueueFamilyProperties2 *properties = (VkQueueFamilyProperties2*)calloc(queues_count, sizeof(VkQueueFamilyProperties2));
		VkQueueFamilyVideoPropertiesKHR *video_properties = (VkQueueFamilyVideoPropertiesKHR*)calloc(queues_count, sizeof(VkQueueFamilyVideoPropertiesKHR));
		if (properties == NULL || video_properties == NULL) //TODO probably return error?
		{
			printf("[vulkan_decode] Failed to allocate properties and/or video_properties arrays!\n");
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

		//printf("\tQueue families of the device:\n");
		// the only important queue flags for us
		const VkQueueFlags queueFlagsFilter = (VK_QUEUE_GRAPHICS_BIT |
                                               VK_QUEUE_COMPUTE_BIT |
                                               VK_QUEUE_TRANSFER_BIT |
                                               VK_QUEUE_VIDEO_DECODE_BIT_KHR |
                                               VK_QUEUE_VIDEO_ENCODE_BIT_KHR);
		bool approved = false;
		uint32_t preferred_queue_family = 0;
		for (uint32_t j = 0; j < queues_count; ++j)
		{
			VkQueueFlags flags = properties[j].queueFamilyProperties.queueFlags & queueFlagsFilter;
			VkVideoCodecOperationFlagsKHR videoFlags = video_properties[j].videoCodecOperations;

			/*int encode = flags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR ? 1 : 0;
			int decode = flags & VK_QUEUE_VIDEO_DECODE_BIT_KHR ? 1 : 0;
			int compute = flags & VK_QUEUE_COMPUTE_BIT ? 1 : 0;
			int transfer = flags & VK_QUEUE_TRANSFER_BIT ? 1 : 0;
			int graphics = flags & VK_QUEUE_GRAPHICS_BIT ? 1 : 0;
			printf("\tflags: %d, encode: %d, decode: %d, compute: %d, transfer: %d, graphics: %d\n",
					  flags, encode, decode, compute, transfer, graphics);

			int h264_en = videoFlags & VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR ? 1 : 0;
			int h265_en = videoFlags & VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR ? 1 : 0;
			int h264_de = videoFlags & VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR ? 1 : 0;
			int h265_de = videoFlags & VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR ? 1 : 0;
			//int av1_de = videoFlags & VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR ? 1 : 0;
			printf("\tvideo flags: %d, h264: %d %d, h265: %d %d\n",
					  videoFlags, h264_en, h264_de, h265_en, h265_de);*/

			if (requestedQueueFamilyFlags == (flags & requestedQueueFamilyFlags) && videoFlags)
			{
				preferred_queue_family = j;
				approved = true;
				break; // the first suitable queue family should be the preferred one
			}
		}
		
		if (approved)
		{
			chosen = devices[i];
			if (queue_family_idx) *queue_family_idx = preferred_queue_family;
			if (queue_family)
			{
				*queue_family = properties[preferred_queue_family];
				queue_family->pNext = NULL;
			}
			if (queue_video_props) *queue_video_props = video_properties[preferred_queue_family];	
		}

		free(properties);
		free(video_properties);
	}

	return chosen;
}

static void * vulkan_decompress_init(void)
{
	printf("vulkan_decode - init\n");

	// ---Allocation of the vulkan_decompress state and sps/pps arrays---
	struct state_vulkan_decompress *s = calloc(1, sizeof(struct state_vulkan_decompress));
	if (s == NULL)
	{
		printf("[vulkan_decode] Couldn't allocate memory for state struct!\n");
		return NULL;
	}

	sps_t *sps_array = calloc(MAX_SPS_IDS, sizeof(sps_t));
	if (sps_array == NULL)
	{
		printf("[vulkan_decode] Couldn't allocate memory for SPS array (num of members: %u)!\n", MAX_SPS_IDS);
		free(s);
		return NULL;
	}

	pps_t *pps_array = calloc(MAX_PPS_IDS, sizeof(pps_t));
	if (pps_array == NULL)
	{
		printf("[vulkan_decode] Couldn't allocate memory for PPS array (num of members: %u)!\n", MAX_PPS_IDS);
		free(sps_array);
		free(s);
		return NULL;
	}

	s->sps_array = sps_array;
	s->pps_array = pps_array;

	// ---Dynamic loading of the vulkan loader library---
	const char vulkan_lib_filename[] = "vulkan-1.dll"; //TODO .so
    s->vulkanLib = LoadLibrary(vulkan_lib_filename);
	if (s->vulkanLib == NULL)
	{
		printf("[vulkan_decode] Vulkan loader file '%s' not found!\n", vulkan_lib_filename);
		free(pps_array);
		free(sps_array);
		free(s);
		return NULL;
	}

	// ---Getting the loader function---
	const char vulkan_proc_name[] = "vkGetInstanceProcAddr";
	PFN_vkGetInstanceProcAddr getInstanceProcAddr = (PFN_vkGetInstanceProcAddr)GetProcAddress(s->vulkanLib, vulkan_proc_name);
    if (getInstanceProcAddr == NULL) {

		printf("[vulkan_decode] Vulkan function '%s' not found!\n", vulkan_proc_name);
        FreeLibrary(s->vulkanLib);
		free(pps_array);
		free(sps_array);
		free(s);
        return NULL;
    }
	s->loader = getInstanceProcAddr;

	// ---Loading function pointers where the instance is not needed---
	if (!load_vulkan_functions_globals(getInstanceProcAddr))
	{
		printf("[vulkan_decode] Failed to load all vulkan functions!\n");
		FreeLibrary(s->vulkanLib);
		free(pps_array);
		free(sps_array);
		free(s);
        return NULL;
	}

	// ---Enabling validation layers---
	#ifdef VULKAN_VALIDATE
	const char* const validationLayers[] = {
						"VK_LAYER_KHRONOS_validation",
						NULL };
	
	//printf("Checking for validation layers.\n");
	if (!check_for_validation_layers(validationLayers))
	{
		printf("[vulkan_deconde] Required vulkan validation layers not found!\n");
		FreeLibrary(s->vulkanLib);
		free(pps_array);
		free(sps_array);
		free(s);
        return NULL;
	}
	#endif

	// ---Checking for extensions---
	const char* const requiredInstanceExtensions[] = {
						VK_EXT_DEBUG_REPORT_EXTENSION_NAME, //?
	#ifdef VULKAN_VALIDATE
						VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
	#endif
						VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME, //maxBuffSize
						NULL };
	
	if (!check_for_instance_extensions(requiredInstanceExtensions))
	{
		//error msg should be printed inside of check_for_extensions
		FreeLibrary(s->vulkanLib);
		free(pps_array);
		free(sps_array);
		free(s);
        return NULL;
	}

	// ---Creating the vulkan instance---
	VkApplicationInfo appInfo = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
								  .pApplicationName = "UltraGrid vulkan_decode",
								  .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
								  .pEngineName = "No Engine",
								  .engineVersion = VK_MAKE_VERSION(1, 0, 0),
								  .apiVersion = VK_API_VERSION_1_3 };
	VkInstanceCreateInfo createInfo = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
										.pApplicationInfo = &appInfo,
										//-1 for not counting the NULL ptr
										.enabledExtensionCount = sizeof(requiredInstanceExtensions) / sizeof(requiredInstanceExtensions[0]) -1,
										.ppEnabledExtensionNames = requiredInstanceExtensions,
	#ifdef VULKAN_VALIDATE
										.enabledLayerCount = sizeof(validationLayers) / sizeof(validationLayers[0]) - 1,
										.ppEnabledLayerNames = validationLayers };
	#else
										.enabledLayerCount = 0 };
	#endif
	VkResult result = vkCreateInstance(&createInfo, NULL, &s->instance);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to create vulkan instance! Error: %d\n", result);
        FreeLibrary(s->vulkanLib);
		free(pps_array);
		free(sps_array);
		free(s);
		return NULL;
	}

	if (!load_vulkan_functions_with_instance(getInstanceProcAddr, s->instance))
	{
		printf("[vulkan_decode] Failed to load all instance related vulkan functions!\n");
		if (vkDestroyInstance != NULL) vkDestroyInstance(s->instance, NULL);
		FreeLibrary(s->vulkanLib);
		free(pps_array);
		free(sps_array);
		free(s);
        return NULL;
	}

	// ---Setting up Debug messenger---
	#ifdef VULKAN_VALIDATE
	result = create_debug_messenger(s->instance, &s->debugMessenger);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to setup debug messenger!\n");
		vkDestroyInstance(s->instance, NULL);
        FreeLibrary(s->vulkanLib);
		free(pps_array);
		free(sps_array);
		free(s);
		return NULL;
	}
	#endif

	// ---Choosing of physical device---
	VkQueueFlags requestedFamilyQueueFlags = VK_QUEUE_VIDEO_DECODE_BIT_KHR | VK_QUEUE_TRANSFER_BIT;
	const char* const requiredDeviceExtensions[] = {
	//#if defined(__linux) || defined(__linux__) || defined(linux)
    //    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,	//?
    //    VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME,	//?
	//#endif
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,	//required by VK_KHR_VIDEO_QUEUE extension
        VK_KHR_VIDEO_QUEUE_EXTENSION_NAME,
        VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME,
		VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME,
		VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME,
		VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME,
		VK_KHR_MAINTENANCE_4_EXTENSION_NAME,		//maxBuffSize
        NULL };

	uint32_t phys_devices_count = 0;
	vkEnumeratePhysicalDevices(s->instance, &phys_devices_count, NULL);
	printf("phys_devices_count: %d\n", phys_devices_count);
	if (phys_devices_count == 0)
	{
		printf("[vulkan_decode] No physical devices found!\n");
		destroy_debug_messenger(s->instance, s->debugMessenger);
		vkDestroyInstance(s->instance, NULL);
        FreeLibrary(s->vulkanLib);
		free(pps_array);
		free(sps_array);
		free(s);
		return NULL;
	}

	VkPhysicalDevice *devices = (VkPhysicalDevice*)calloc(phys_devices_count, sizeof(VkPhysicalDevice));
	if (devices == NULL)
	{
		printf("[vulkan_decode] Failed to allocate array for devices!\n");
		destroy_debug_messenger(s->instance, s->debugMessenger);
		vkDestroyInstance(s->instance, NULL);
        FreeLibrary(s->vulkanLib);
		free(pps_array);
		free(sps_array);
		free(s);
		return NULL;
	}

	vkEnumeratePhysicalDevices(s->instance, &phys_devices_count, devices);
	VkQueueFamilyProperties2 chosen_queue_family; //TODO use this?
	VkQueueFamilyVideoPropertiesKHR chosen_queue_video_props; //TODO use this more?
	s->physicalDevice = choose_physical_device(devices, phys_devices_count, requestedFamilyQueueFlags,
											   requiredDeviceExtensions,
											   &s->queueFamilyIdx, &chosen_queue_family, &chosen_queue_video_props);
	free(devices);
	if (s->physicalDevice == VK_NULL_HANDLE)
	{
		printf("[vulkan_decode] Failed to choose a appropriate physical device!\n");
		destroy_debug_messenger(s->instance, s->debugMessenger);
		vkDestroyInstance(s->instance, NULL);
        FreeLibrary(s->vulkanLib);
		free(pps_array);
		free(sps_array);
		free(s);
		return NULL;
	}

	s->queueVideoFlags = chosen_queue_video_props.videoCodecOperations;
	assert(chosen_queue_family.pNext == NULL && chosen_queue_video_props.pNext == NULL);
	assert(s->queueVideoFlags != 0);

	VkPhysicalDeviceMaintenance4PropertiesKHR physDevicePropertiesExtra =
												{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES };
	VkPhysicalDeviceProperties2KHR physDeviceProperties = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
														 	.pNext = &physDevicePropertiesExtra };
	printf("before: maxBufferSize: %u\n", physDevicePropertiesExtra.maxBufferSize);
	vkGetPhysicalDeviceProperties2KHR(s->physicalDevice, &physDeviceProperties);
	//TODO this?
	printf("after: maxBufferSize: %u\n", physDevicePropertiesExtra.maxBufferSize);
	//void *phys_pnext = physDeviceProperties.pNext;
	//printf("after2: maxBufferSize: %u\n", ((VkPhysicalDeviceMaintenance4PropertiesKHR*)phys_pnext)->maxBufferSize);

	s->physDeviceMaxBuffSize = physDevicePropertiesExtra.maxBufferSize; //TODO this is wrong for some unknown reason
	printf("Chosen physical device is: '%s', maxBuffSize: %u and chosen queue family index is: %d\n", 
				physDeviceProperties.properties.deviceName, s->physDeviceMaxBuffSize, s->queueFamilyIdx);

	// ---Creating a logical device---
	float queue_priorities = 1.0f;
	VkDeviceQueueCreateInfo queueCreateInfo = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
												.queueFamilyIndex = s->queueFamilyIdx,
												.queueCount = 1,
												.pQueuePriorities = &queue_priorities };
	VkPhysicalDeviceVulkan13Features enabledDeviceFeatures13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
																 .synchronization2 = 1, .maintenance4 = 1 };
	VkDeviceCreateInfo createDeviceInfo = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
											.pNext = (void*)&enabledDeviceFeatures13,
											.flags = 0,
											.queueCreateInfoCount = 1,
											.pQueueCreateInfos = &queueCreateInfo,
											.pEnabledFeatures = NULL,
											//-1 because of NULL at the end of the array
											.enabledExtensionCount = sizeof(requiredDeviceExtensions) / sizeof(requiredDeviceExtensions[0]) - 1,
											.ppEnabledExtensionNames = requiredDeviceExtensions };
	
	result = vkCreateDevice(s->physicalDevice, &createDeviceInfo, NULL, &s->device);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to create a appropriate vulkan logical device!\n");
		destroy_debug_messenger(s->instance, s->debugMessenger);
		vkDestroyInstance(s->instance, NULL);
        FreeLibrary(s->vulkanLib);
		free(pps_array);
		free(sps_array);
		free(s);
		return NULL;
	}

	vkGetDeviceQueue(s->device, s->queueFamilyIdx, 0, &s->decodeQueue);

	s->fence = VK_NULL_HANDLE;
	s->decodeBuffer = VK_NULL_HANDLE; //buffers gets created in allocate_buffers function
	s->dstPicBuffer = VK_NULL_HANDLE;
	s->bufferMemory = VK_NULL_HANDLE;
	s->commandPool = VK_NULL_HANDLE;  //command pool gets created in prepare function
	s->cmdBuffer = VK_NULL_HANDLE;	  //same
	s->videoSession = VK_NULL_HANDLE; //video session gets created in prepare function
	s->videoSessionParams = VK_NULL_HANDLE; //same
	//s->videoSessionParams_update_count = 0;
	s->videoSessionMemory_count = 0;
	s->videoSessionMemory = NULL;

	for (size_t i = 0; i < MAX_REF_FRAMES + 1; ++i)
	{
		s->dpb[i] = VK_NULL_HANDLE;
		s->dpbViews[i] = VK_NULL_HANDLE;
	}
	s->dpbMemory = VK_NULL_HANDLE;
	s->dpbFormat = VK_FORMAT_UNDEFINED;
	for (size_t i = 0; i < MAX_REF_FRAMES; ++i)
	{
		// setting the reference queue members to some unvalid value 
		s->referenceSlotsQueue[i] = (slice_info_t){ .idr_pic_id = -1, .sps_id = -1, .pps_id = -1, .frame_num = -1, .frame_seq = -1,
													.poc = -1, .poc_lsb = -1 };
	}

	printf("[vulkan_decode] Initialization finished successfully.\n");
	return s;
}

static void free_video_session_memory(struct state_vulkan_decompress *s)
{
	if (vkFreeMemory != NULL && s->device != VK_NULL_HANDLE && s->videoSessionMemory != NULL)
	{
		for (uint32_t i = 0; i < s->videoSessionMemory_count; ++i)
		{
			vkFreeMemory(s->device, s->videoSessionMemory[i], NULL);
		}
	}

	free(s->videoSessionMemory);
	s->videoSessionMemory_count = 0;
	s->videoSessionMemory = NULL;
}

static void vulkan_decompress_done(void *state)
{
	printf("vulkan_decode - done\n");
	struct state_vulkan_decompress *s = (struct state_vulkan_decompress *)state;
	if (!s) return;

	if (vkDestroyVideoSessionParametersKHR != NULL && s->device != VK_NULL_HANDLE)
			vkDestroyVideoSessionParametersKHR(s->device, s->videoSessionParams, NULL);

	if (vkDestroyVideoSessionKHR != NULL && s->device != VK_NULL_HANDLE)
			vkDestroyVideoSessionKHR(s->device, s->videoSession, NULL);
	
	free_video_session_memory(s);

	destroy_dpb(s);
	
	if (vkDestroyCommandPool != NULL && s->device != VK_NULL_HANDLE)
			vkDestroyCommandPool(s->device, s->commandPool, NULL);

	free_buffers(s);

	if (vkDestroyFence != NULL && s->device != VK_NULL_HANDLE)
			vkDestroyFence(s->device, s->fence, NULL);
	
	if (vkDestroyDevice != NULL) vkDestroyDevice(s->device, NULL);

	destroy_debug_messenger(s->instance, s->debugMessenger);

	if (vkDestroyInstance != NULL) vkDestroyInstance(s->instance, NULL);

	FreeLibrary(s->vulkanLib);

	free(s->pps_array);
	free(s->sps_array);
	free(s);
}

static VkVideoCodecOperationFlagsKHR codec_to_vulkan_flag(codec_t codec)
{
	switch(codec)
	{
		case H264: return VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
		case H265: return VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR;
		//case AV1: return VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR;
		default: return VK_VIDEO_CODEC_OPERATION_NONE_KHR;
	}
}

static codec_t vulkan_flag_to_codec(VkVideoCodecOperationFlagsKHR flag)
{
	switch(flag)
	{
		case VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR: return H264;
		case VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR: return H265;
		//case VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR: return AV1;
		default: return VIDEO_CODEC_NONE;
	}
}

static bool configure_with(struct state_vulkan_decompress *s, struct video_desc desc)
{
	printf("vulkan_decode - configure_with\n");
	const char *spec_name = get_codec_name_long(desc.color_spec);
	printf("\tw: %u, h: %u, color_spec: '%s', fps: %f, tile_count: %u\n",
			desc.width, desc.height, spec_name, desc.fps, desc.tile_count);

	s->codecOperation = VK_VIDEO_CODEC_OPERATION_NONE_KHR;
	VkVideoCodecOperationFlagsKHR videoCodecOperation = codec_to_vulkan_flag(desc.color_spec);

	if (!(s->queueVideoFlags & videoCodecOperation))
	{
		printf("[vulkan_decode] Wanted color spec: '%s' is not supported by chosen vulkan queue family!\n", spec_name);
		return false;
	}

	assert(videoCodecOperation != VK_VIDEO_CODEC_OPERATION_NONE_KHR);

	s->codecOperation = videoCodecOperation;
	s->width = desc.width;
	s->height = desc.height;

	s->dpbHasDefinedLayout = false;
	s->dpbFormat = VK_FORMAT_UNDEFINED;
	s->referenceSlotsQueue_start = 0;
	s->referenceSlotsQueue_count = 0;

	s->prev_poc_lsb = 0;
	s->prev_poc_msb = 0;
	s->idr_frame_seq = 0;
	s->current_frame_seq = 0;

	return true;
}

static int vulkan_decompress_reconfigure(void *state, struct video_desc desc,
										 int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
	printf("vulkan_decode - reconfigure\n");
	struct state_vulkan_decompress *s = (struct state_vulkan_decompress *)state;
	if (!s) return false;

	const char *spec_name = get_codec_name_long(desc.color_spec);
	const char *out_name = get_codec_name_long(out_codec);
	printf("\tcodec color_spec: '%s', out_codec: '%s', pitch: %d\n", spec_name, out_name, pitch);
	if (out_codec == VIDEO_CODEC_NONE) printf("\tRequested probing.\n");

	if (desc.tile_count != 1)
	{
		//TODO they could be supported
		printf("[vulkan_decode] Tiled video formats are not supported!\n");
		return false;
	}

	s->prepared = false; //TODO - freeing resources probably needed when s->prepared == true
	s->sps_vps_found = false;
	s->resetVideoCoding = true;
	s->depth_chroma = 0;
	s->depth_luma = 0;
	s->subsampling = 0;
	s->profileIdc = STD_VIDEO_H264_PROFILE_IDC_INVALID;

	UNUSED(rshift);
	UNUSED(gshift);
	UNUSED(bshift);
	/*s->rshift = rshift;
	s->gshift = gshift;
	s->bshift = bshift;*/

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
	const char *compression_name = get_codec_name(compression);
	const char *ugc_name = get_codec_name(ugc);
	printf("\tcompression: '%s', ugc: '%s', pixfmt - depth: %d, subsampling: %d, rgb: %d\n",
			compression_name, ugc_name, internal.depth, internal.subsampling, internal.rgb ? 1 : 0);
	
	VkVideoCodecOperationFlagsKHR vulkanFlag = codec_to_vulkan_flag(compression);

	if (vulkanFlag == VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR ||
		vulkanFlag == VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR)
			return 30; //TODO magic value

	return -1; 
}

static VkVideoChromaSubsamplingFlagBitsKHR subsampling_to_vulkan_flag(int subs)
{
	switch((enum subsampling)subs)
	{
		case SUBS_420: return VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR;
		case SUBS_422: return VK_VIDEO_CHROMA_SUBSAMPLING_422_BIT_KHR;
		case SUBS_444: return VK_VIDEO_CHROMA_SUBSAMPLING_444_BIT_KHR;
		case SUBS_4444: return VK_VIDEO_CHROMA_SUBSAMPLING_444_BIT_KHR;
		default: return VK_VIDEO_CHROMA_SUBSAMPLING_INVALID_KHR;
	}
}

static int vulkan_flag_to_subsampling(VkVideoChromaSubsamplingFlagBitsKHR flag)
{
	switch(flag)
	{
		case VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR: return 4200;
		case VK_VIDEO_CHROMA_SUBSAMPLING_422_BIT_KHR: return 4220;
		case VK_VIDEO_CHROMA_SUBSAMPLING_444_BIT_KHR: return 4440;
		default: return 0; // invalid subsampling
	}
}

static int h264_flag_to_subsampling(StdVideoH264ChromaFormatIdc flag)
{
	switch(flag)
	{
		case STD_VIDEO_H264_CHROMA_FORMAT_IDC_420: return 4200;
		case STD_VIDEO_H264_CHROMA_FORMAT_IDC_422: return 4220;
		case STD_VIDEO_H264_CHROMA_FORMAT_IDC_444: return 4440;
		default: return 0; // invalid subsampling
	}
}

static int h265_flag_to_subsampling(StdVideoH265ChromaFormatIdc flag)
{
	switch(flag)
	{
		case STD_VIDEO_H265_CHROMA_FORMAT_IDC_420: return 4200;
		case STD_VIDEO_H265_CHROMA_FORMAT_IDC_422: return 4220;
		case STD_VIDEO_H265_CHROMA_FORMAT_IDC_444: return 4440;
		default: return 0; // invalid subsampling
	}
}

static VkVideoComponentBitDepthFlagBitsKHR depth_to_vulkan_flag(int depth)
{
	switch(depth)
	{
		case 8: return VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
		case 10: return VK_VIDEO_COMPONENT_BIT_DEPTH_10_BIT_KHR;
		case 12: return VK_VIDEO_COMPONENT_BIT_DEPTH_12_BIT_KHR;
		default: return VK_VIDEO_COMPONENT_BIT_DEPTH_INVALID_KHR;
	}
}

static bool does_video_size_fit(VkExtent2D videoSize, VkExtent2D minExtent, VkExtent2D maxExtent)
{
	assert(minExtent.width <= maxExtent.width && minExtent.height <= maxExtent.height);

	return minExtent.width <= videoSize.width && videoSize.width <= maxExtent.width &&
		   minExtent.height <= videoSize.height && videoSize.height <= maxExtent.height;
}

static VkFormat codec_to_vulkan_format(codec_t codec)
{
	switch(codec)
	{
		case I420: return VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
		default: return VK_FORMAT_UNDEFINED;
	}
}

static bool check_format(const VkVideoFormatPropertiesKHR *props, const VkFormatProperties *extraProps)
{
	VkFormatFeatureFlags optFlags = extraProps->optimalTilingFeatures;
	
	return props->format == VK_FORMAT_G8_B8R8_2PLANE_420_UNORM && //TODO allow different formats
		   optFlags & VK_FORMAT_FEATURE_VIDEO_DECODE_DPB_BIT_KHR &&
		   optFlags & VK_FORMAT_FEATURE_VIDEO_DECODE_OUTPUT_BIT_KHR && //only for VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR
		   optFlags & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT_KHR;
}

//TODO delete:
static void print_bits(unsigned char num);

static bool check_for_vulkan_format(VkPhysicalDevice physDevice, VkPhysicalDeviceVideoFormatInfoKHR videoFormatInfo,
									VkVideoFormatPropertiesKHR *formatProperties)
{
	uint32_t properties_count = 0;

	VkResult result = vkGetPhysicalDeviceVideoFormatPropertiesKHR(physDevice, &videoFormatInfo,
																  &properties_count, NULL);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to get video format properties of physical device!\n");
		return false;
	}
	if (properties_count == 0)
	{
		return false;
	}

	VkVideoFormatPropertiesKHR *properties = (VkVideoFormatPropertiesKHR*)
							calloc(properties_count, sizeof(VkVideoFormatPropertiesKHR));
	if (properties == NULL)
	{
		printf("[vulkan_decode] Failed to allocate memory for video format properties!\n");
		return false;
	}
	for (uint32_t i = 0; i < properties_count; ++i)
			properties[i].sType = VK_STRUCTURE_TYPE_VIDEO_FORMAT_PROPERTIES_KHR;

	result = vkGetPhysicalDeviceVideoFormatPropertiesKHR(physDevice, &videoFormatInfo,
														 &properties_count, properties);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to get video format properties of physical device!\n");
		free(properties);
		return false;
	}

	printf("videoFromatProperties_count: %u\n", properties_count);
	//VkFormat wanted = codec_to_vulkan_format(codec);

	for (uint32_t i = 0; i < properties_count; ++i)
	{
		VkFormat format = properties[i].format;
		VkFormatProperties extraProps = { 0 };
		vkGetPhysicalDeviceFormatProperties(physDevice, format, &extraProps);

		printf("\tformat: %d, image_type: %d, imageCreateFlags: %d, imageTiling: %d, imageUsageFlags: %d\n",
				format, properties[i].imageType, properties[i].imageCreateFlags, properties[i].imageTiling, properties[i].imageUsageFlags);
		printf("\tusage flags: ");
		print_bits((unsigned char)(properties[i].imageUsageFlags >> 8)); 
		print_bits((unsigned char)properties[i].imageUsageFlags);
		putchar('\n');
		
		bool blitSrc = VK_FORMAT_FEATURE_BLIT_SRC_BIT & extraProps.optimalTilingFeatures ? 1 : 0;
		bool blitDst = VK_FORMAT_FEATURE_BLIT_DST_BIT & extraProps.optimalTilingFeatures ? 1 : 0;
		bool decodeOutput = VK_FORMAT_FEATURE_VIDEO_DECODE_OUTPUT_BIT_KHR & extraProps.optimalTilingFeatures ? 1 : 0;
		bool decodeDPB = VK_FORMAT_FEATURE_VIDEO_DECODE_DPB_BIT_KHR & extraProps.optimalTilingFeatures ? 1 : 0;
		bool transferSrc = VK_FORMAT_FEATURE_TRANSFER_SRC_BIT_KHR & extraProps.optimalTilingFeatures ? 1 : 0;
		bool transferDst = VK_FORMAT_FEATURE_TRANSFER_DST_BIT_KHR  & extraProps.optimalTilingFeatures ? 1 : 0;
		
		printf("\tblit: %d %d, decode out: %d, decode dpb: %d, transfer: %d %d\n",
				blitSrc, blitDst, decodeOutput, decodeDPB, transferSrc, transferDst);

		if (check_format(properties + i, &extraProps))
		{
			if (formatProperties != NULL) *formatProperties = properties[i];
			
			free(properties);
			return true;
		}
	}

	printf("[vulkan_decode] Wanted output video format is not supported!\n");
	free(properties);
	return false;
}

static bool find_memory_type(struct state_vulkan_decompress *s, uint32_t typeFilter,
								 VkMemoryPropertyFlags reqProperties, uint32_t *idx)
{
	assert(s->physicalDevice != VK_NULL_HANDLE);

	VkPhysicalDeviceMemoryProperties memoryProperties = { 0 };
	vkGetPhysicalDeviceMemoryProperties(s->physicalDevice, &memoryProperties);

	assert(memoryProperties.memoryTypeCount <= 32);

	for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
	{
		if ((typeFilter & (1 << i)) &&
			(memoryProperties.memoryTypes[i].propertyFlags & reqProperties) == reqProperties)
		{
			if (idx != NULL) *idx = i;
			return true;
		}
	}

	return false;
}

static bool allocate_buffers(struct state_vulkan_decompress *s, VkVideoProfileListInfoKHR videoProfileList,
							 const VkVideoCapabilitiesKHR videoCapabilities)
{
	assert(s->decodeBuffer == VK_NULL_HANDLE);
	assert(s->dstPicBuffer == VK_NULL_HANDLE);
	assert(s->bufferMemory == VK_NULL_HANDLE);

	const VkDeviceSize wantedDecodeBufferSize = 1024 * 1024; //TODO magic number, check if smaller than allowed amount
	VkDeviceSize sizeAlignment = videoCapabilities.minBitstreamBufferSizeAlignment;
	s->decodeBufferSize = (wantedDecodeBufferSize + (sizeAlignment - 1)) & ~(sizeAlignment - 1); //alignment bit mask magic
	VkBufferCreateInfo decodeBufferInfo = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
									  		.pNext = (void*)&videoProfileList,
									  		.flags = 0,
									  		.usage = VK_BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR,
									  		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
									  		.size = s->decodeBufferSize,
									  		.queueFamilyIndexCount = 1,
									  		.pQueueFamilyIndices = &s->queueFamilyIdx };
	VkResult result = vkCreateBuffer(s->device, &decodeBufferInfo, NULL, &s->decodeBuffer);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to create vulkan buffer for decoding!\n");
		free_buffers(s);

		return false;
	}
	s->decodeBufferOffsetAlignment = videoCapabilities.minBitstreamBufferOffsetAlignment;

	s->dstPicBufferSize = 10 * 1024 * 1024; //TODO magic numbers, could be checked too
	VkBufferCreateInfo picBufferInfo = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
										 .flags = 0,
										 .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
										 .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
										 .size = s->dstPicBufferSize,
										 .queueFamilyIndexCount = 1,
										 .pQueueFamilyIndices = &s->queueFamilyIdx };
	result = vkCreateBuffer(s->device, &picBufferInfo, NULL, &s->dstPicBuffer);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to create vulkan buffer for decoded picture!\n");
		free_buffers(s);

		return false;
	}

	assert(s->decodeBuffer != VK_NULL_HANDLE);
	assert(s->dstPicBuffer != VK_NULL_HANDLE);

	VkMemoryRequirements decodeBufferMemReq, picBufferMemReq;
	vkGetBufferMemoryRequirements(s->device, s->decodeBuffer, &decodeBufferMemReq);
	vkGetBufferMemoryRequirements(s->device, s->dstPicBuffer, &picBufferMemReq);

	uint32_t memType_idx = 0;
	if (!find_memory_type(s, decodeBufferMemReq.memoryTypeBits | picBufferMemReq.memoryTypeBits,
						  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
						  &memType_idx))
	{
		printf("[vulkan_decode] Failed to find required memory type for needed vulkan buffers!\n");
		free_buffers(s);

		return false;
	}

	s->dstPicBufferMemoryOffset = (decodeBufferMemReq.size + (picBufferMemReq.alignment - 1))
								   & ~(picBufferMemReq.alignment - 1); //alignment bit mask magic
	VkDeviceSize memSize = s->dstPicBufferMemoryOffset + picBufferMemReq.size;

	VkMemoryAllocateInfo memAllocInfo = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
										  .allocationSize = memSize,
										  .memoryTypeIndex = memType_idx };
	result = vkAllocateMemory(s->device, &memAllocInfo, NULL, &s->bufferMemory);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to allocate memory for needed vulkan buffers!\n");
		free_buffers(s);

		return false;
	}

	assert(s->bufferMemory != VK_NULL_HANDLE);

	result = vkBindBufferMemory(s->device, s->decodeBuffer, s->bufferMemory, 0);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to bind vulkan memory to vulkan decode buffer!\n");
		free_buffers(s);
		
		return false;
	}

	result = vkBindBufferMemory(s->device, s->dstPicBuffer, s->bufferMemory, s->dstPicBufferMemoryOffset);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to bind vulkan memory to vulkan decodec picture buffer!\n");
		free_buffers(s);
		
		return false;
	}

	return true;
}

static void free_buffers(struct state_vulkan_decompress *s)
{
	//buffers needs to get destroyed first
	if (vkDestroyBuffer != NULL && s->device != VK_NULL_HANDLE)
	{
		vkDestroyBuffer(s->device, s->decodeBuffer, NULL);
		vkDestroyBuffer(s->device, s->dstPicBuffer, NULL);
	}

	s->decodeBuffer = VK_NULL_HANDLE;
	s->decodeBufferSize = 0;
	s->decodeBufferOffsetAlignment = 0;

	s->dstPicBuffer = VK_NULL_HANDLE;
	s->dstPicBufferSize = 0;
	s->dstPicBufferMemoryOffset = 0;

	if (vkFreeMemory != NULL && s->device != VK_NULL_HANDLE)
	{
		vkFreeMemory(s->device, s->bufferMemory, NULL);
	}

	s->bufferMemory = VK_NULL_HANDLE;
}

static bool allocate_memory_for_video_session(struct state_vulkan_decompress *s)
{
	assert(s->device != VK_NULL_HANDLE);
	assert(s->videoSession != VK_NULL_HANDLE);
	assert(s->videoSessionMemory == NULL); // videoSessionMemory should be properly freed beforehand

	uint32_t memoryRequirements_count = 0;
	VkResult result = vkGetVideoSessionMemoryRequirementsKHR(s->device, s->videoSession, &memoryRequirements_count, NULL);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to get vulkan video session memory requirements!\n");
		return false;
	}

	VkVideoSessionMemoryRequirementsKHR *memoryRequirements = (VkVideoSessionMemoryRequirementsKHR*)
																calloc(memoryRequirements_count,
																	   sizeof(VkVideoSessionMemoryRequirementsKHR));
	VkBindVideoSessionMemoryInfoKHR *bindMemoryInfo = (VkBindVideoSessionMemoryInfoKHR*)
														calloc(memoryRequirements_count,
																sizeof(VkBindVideoSessionMemoryInfoKHR));
	s->videoSessionMemory_count = memoryRequirements_count;
	s->videoSessionMemory = (VkDeviceMemory*)calloc(memoryRequirements_count, sizeof(VkDeviceMemory));
	if (memoryRequirements == NULL || bindMemoryInfo == NULL || s->videoSessionMemory == NULL)
	{
		printf("[vulkan_decode] Failed to allocate memory for vulkan video session!\n");
		free(memoryRequirements);
		free(bindMemoryInfo);
		free_video_session_memory(s);

		return false;
	}

	for (uint32_t i = 0; i < memoryRequirements_count; ++i)
	{
		memoryRequirements[i].sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR;
		bindMemoryInfo[i].sType = VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR;
		s->videoSessionMemory[i] = VK_NULL_HANDLE;
	}
	
	result = vkGetVideoSessionMemoryRequirementsKHR(s->device, s->videoSession, &memoryRequirements_count, memoryRequirements);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to get vulkan video session memory requirements!\n");
		free(memoryRequirements);
		free(bindMemoryInfo);
		free_video_session_memory(s);

		return false;
	}

	//TODO check if we can make enough allocations using maxMemoryAllocationCount
	for (uint32_t i = 0; i < memoryRequirements_count; ++i)
	{
        uint32_t memoryTypeIndex = 0;
		if (!find_memory_type(s, memoryRequirements[i].memoryRequirements.memoryTypeBits,
							  0, &memoryTypeIndex))
		{
			printf("[vulkan_decode] No suitable memory type for vulkan video session requirments!\n");
			free(memoryRequirements);
			free(bindMemoryInfo);
			free_video_session_memory(s);

			return false;
		}

		//TODO create one big memory allocation for and bind parts of it to video session
		VkMemoryAllocateInfo allocateInfo = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
											  .allocationSize = memoryRequirements[i].memoryRequirements.size,
											  .memoryTypeIndex = memoryTypeIndex };
		result = vkAllocateMemory(s->device, &allocateInfo, NULL, &s->videoSessionMemory[i]);
		if (result != VK_SUCCESS)
		{
			printf("[vulkan_decode] Failed to allocate vulkan memory!\n");
			free(memoryRequirements);
			free(bindMemoryInfo);
			free_video_session_memory(s);

			return false;
		}

		bindMemoryInfo[i].memoryBindIndex = memoryRequirements[i].memoryBindIndex;
		bindMemoryInfo[i].memorySize = memoryRequirements[i].memoryRequirements.size;
		bindMemoryInfo[i].memoryOffset = 0 * memoryRequirements[i].memoryRequirements.alignment; //zero for now
		bindMemoryInfo[i].memory = s->videoSessionMemory[i];
	}

	result = vkBindVideoSessionMemoryKHR(s->device, s->videoSession, memoryRequirements_count, bindMemoryInfo);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Can't bind video session memory!\n");
		free(memoryRequirements);
		free(bindMemoryInfo);
		free_video_session_memory(s);

		return false;
	}

	free(memoryRequirements);
	free(bindMemoryInfo);
	return true;
}

static bool begin_cmd_buffer(struct state_vulkan_decompress *s)
{
	//maybe pointless since VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT flag
	VkResult result = vkResetCommandBuffer(s->cmdBuffer, 0);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to reset the vulkan command buffer!\n");
		return false;
	}
	
	VkCommandBufferBeginInfo cmdBufferBeginInfo = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
													.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
	result = vkBeginCommandBuffer(s->cmdBuffer, &cmdBufferBeginInfo);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to begin command buffer recording!\n");
		return false;
	}

	return true;
}

static bool end_cmd_buffer(struct state_vulkan_decompress *s)
{
	VkResult result = vkEndCommandBuffer(s->cmdBuffer);
	if (result == VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR)
	{
		printf("[vulkan_decode] Failed to end command buffer recording - Invalid video standard parameters\n");
		return false;
	}
	else if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to end command buffer recording!\n");
		return false;
	}

	return true;
}

static void deduce_stage_and_access_from_layout(VkImageLayout layout, VkPipelineStageFlags2 *stage, VkAccessFlags2 *access)
{
	switch (layout)
	{
		case VK_IMAGE_LAYOUT_UNDEFINED: // only after the initialization of image
			if (stage != NULL) *stage = VK_PIPELINE_STAGE_2_NONE;
			if (access != NULL) *access = VK_ACCESS_2_NONE;
			break;
		case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: // beginning of the cmd buffer or preparing for decoded picture copying
			if (stage != NULL) *stage = VK_PIPELINE_STAGE_2_COPY_BIT_KHR;
			if (access != NULL) *access = VK_ACCESS_2_TRANSFER_READ_BIT; //TODO
			break;
		/*case VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR: // beginning of the cmd buffer when preparing for decode, or after it
			if (stage != NULL) *stage = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR;
			if (access != NULL) *access = VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR |
										  VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR; //TODO
			break;*/
		case VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR: // beginning of the cmd buffer when preparing for decode, or after it
			if (stage != NULL) *stage = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR;
			if (access != NULL) *access = VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR |
										  VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR; //TODO
			break;
		default:
			break;
	}
}

static void transfer_image_layout(VkCommandBuffer cmdBuffer, VkImage image,
								  VkImageLayout oldLayout, VkImageLayout newLayout)
{
	// transfers image layout (and also does the image's synchronization) using pipeline barrier
	// it records the barrier even when oldLayout == newLayout, cmdBuffer must be in recording state!
	assert(cmdBuffer != VK_NULL_HANDLE);
	assert(image != VK_NULL_HANDLE);

	VkPipelineStageFlags2 srcStage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
						  dstStage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
	VkAccessFlags2 srcAccess = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
				   dstAccess = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;

	deduce_stage_and_access_from_layout(oldLayout, &srcStage, &srcAccess);
	deduce_stage_and_access_from_layout(newLayout, &dstStage, &dstAccess);

	VkImageMemoryBarrier2 imgBarrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
										 .srcStageMask = srcStage,
										 .srcAccessMask = srcAccess,
										 .dstStageMask = dstStage,
										 .dstAccessMask = dstAccess,
										 .oldLayout = oldLayout,
										 .newLayout = newLayout,
										 .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
										 .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
										 .image = image,
										 .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, // all img planes
										 					   .baseMipLevel = 0,
										 					   .levelCount = 1,
										 					   .baseArrayLayer = 0,
										 					   .layerCount = 1 } };
	VkDependencyInfo barrierInfo = { .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
									 .dependencyFlags = 0,
									 .imageMemoryBarrierCount = 1,
									 .pImageMemoryBarriers = &imgBarrier };
	vkCmdPipelineBarrier2KHR(cmdBuffer, &barrierInfo);
}

static bool create_dpb(struct state_vulkan_decompress *s, VkVideoProfileListInfoKHR *videoProfileList)
{
	// Creates the DPB (decoded picture buffer), if success then DPB must be destroyed using destroy_dpb
	// created images are left in undefined layout
	assert(s->device != VK_NULL_HANDLE);
	assert(s->dpbFormat != VK_FORMAT_UNDEFINED);

	const VkImageType imageType = VK_IMAGE_TYPE_2D;
	const VkExtent3D videoSize = { s->width, s->height, 1 }; //depth must be 1 for VK_IMAGE_TYPE_2D

	//imageCreateMaxMipLevels, imageCreateMaxArrayLayers, imageCreateMaxExtent, and imageCreateSampleCounts
	VkImageCreateInfo imgInfo = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
								  .pNext = (void*)videoProfileList,
								  .flags = 0,
								  .usage = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR |
								  		   VK_IMAGE_USAGE_VIDEO_DECODE_SRC_BIT_KHR |
										   VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR |
										   VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
								  .imageType = imageType,
								  .mipLevels = 1,
								  .samples = VK_SAMPLE_COUNT_1_BIT,
								  .format = s->dpbFormat,
								  .extent = videoSize,
								  .arrayLayers = 1,
								  //.tiling = VK_IMAGE_TILING_LINEAR,
								  .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
								  //.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED,
								  .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
								  .queueFamilyIndexCount = 1,
								  .pQueueFamilyIndices = &s->queueFamilyIdx };

	size_t dpb_len = sizeof(s->dpb) / sizeof(s->dpb[0]);

	for (size_t i = 0; i < dpb_len; ++i)
	{
		//printf("Creating image %u.\n", i);
		VkResult result = vkCreateImage(s->device, &imgInfo, NULL, s->dpb + i);
		if (result != VK_SUCCESS)
		{
			printf("[vulkan_decode] Failed to create vulkan image(%u) for DPB slot!\n", i);
			destroy_dpb(s);
			return false;
		}
	}

	VkMemoryRequirements imgMemoryRequirements;
	vkGetImageMemoryRequirements(s->device, s->dpb[0], &imgMemoryRequirements);

	uint32_t imgMemoryTypeIdx = 0;
	if (!find_memory_type(s, imgMemoryRequirements.memoryTypeBits, 0, &imgMemoryTypeIdx))
	{
		printf("[vulkan_decode] Failed to find required memory type for DPB!\n");
		destroy_dpb(s);
		return false;
	}

	VkDeviceSize imgAlignment = imgMemoryRequirements.alignment;
	VkDeviceSize imgAlignedSize = (imgMemoryRequirements.size + (imgAlignment - 1))
								   & ~(imgAlignment - 1); //alignment bit mask magic
	VkMemoryAllocateInfo dpbAllocInfo = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
										  .allocationSize = imgAlignedSize * dpb_len,
										  .memoryTypeIndex = imgMemoryTypeIdx };

	VkResult result = vkAllocateMemory(s->device, &dpbAllocInfo, NULL, &s->dpbMemory);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to allocate vulkan device memory for DPB!\n");
		destroy_dpb(s);
		return false;
	}

	for (size_t i = 0; i < dpb_len; ++i)
	{
		result = vkBindImageMemory(s->device, s->dpb[i], s->dpbMemory, i * imgAlignedSize);
		if (result != VK_SUCCESS)
		{
			printf("[vulkan_decode] Failed to bind vulkan device memory to DPB image (idx: %u)!\n", i);
			destroy_dpb(s);
			return false;
		}
	}

	VkImageSubresourceRange viewSubresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, //TODO aspect
													 .baseMipLevel = 0, .levelCount = 1,
													 .baseArrayLayer = 0, .layerCount = 1 };
	VkImageViewCreateInfo viewInfo = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
									   .flags = 0,
									   .image = VK_NULL_HANDLE, //gets correctly set in the for loop
									   .viewType = VK_IMAGE_VIEW_TYPE_2D,
									   .format = imgInfo.format,
									   .components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                            						   VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY },
									   .subresourceRange = viewSubresourceRange };
	
	for (size_t i = 0; i < dpb_len; ++i)
	{
		//printf("Creating image view %u.\n", i);
		assert(s->dpb[i] != VK_NULL_HANDLE);

		viewInfo.image = s->dpb[i];
		result = vkCreateImageView(s->device, &viewInfo, NULL, s->dpbViews + i);
		if (result != VK_SUCCESS)
		{
			printf("[vulkan_decode] Failed to create vulkan image view(%u) for DPB slot!\n", i);
			destroy_dpb(s);
			return false;
		}
	}

	return true;
}

static void destroy_dpb(struct state_vulkan_decompress *s)
{
	size_t dpb_len = sizeof(s->dpb) / sizeof(s->dpb[0]);

	// freeing dpb image views
	if (vkDestroyImageView != NULL && s->device != VK_NULL_HANDLE)
	{
		for (size_t i = 0; i < dpb_len; ++i)
		{
			vkDestroyImageView(s->device, s->dpbViews[i], NULL);
		}
	}

	// freeing dpb images
	if (vkDestroyImage != NULL && s->device != VK_NULL_HANDLE)
	{
		for (size_t i = 0; i < dpb_len; ++i)
		{
			//printf("Destroying image: %u.\n", i);
			vkDestroyImage(s->device, s->dpb[i], NULL);
		}
	}

	for (size_t i = 0; i < dpb_len; ++i)
	{
		s->dpb[i] = VK_NULL_HANDLE;
		s->dpbViews[i] = VK_NULL_HANDLE;
	}

	// freeing backing device memory
	if (vkFreeMemory != NULL && s->device != VK_NULL_HANDLE) vkFreeMemory(s->device, s->dpbMemory, NULL);
	
	s->dpbMemory = VK_NULL_HANDLE;
	s->dpbHasDefinedLayout = false;
}

static bool prepare(struct state_vulkan_decompress *s, bool *wrong_pixfmt)
{
	printf("vulkan_decode - prepare\n");
	assert(!s->prepared); //this function should be called only when decompress is not prepared

	*wrong_pixfmt = false;

	//codec_t in_codec = vulkan_flag_to_codec(s->codecOperation);
	//struct pixfmt_desc pf_desc = get_pixfmt_desc(in_codec);
	//printf("\tpf_desc - depth: %d, subsampling: %d, rgb: %d, accel_type: %d\n",
	//			pf_desc.depth, pf_desc.subsampling, pf_desc.rgb, pf_desc.accel_type);
	//if (!pf_desc.depth) //pixel format description is invalid
	//{
	//	printf("[vulkan_decode] Got invalid pixel format!\n");
	//	return false;
	//}

	printf("\tPreparing with - depth: %d %d, subsampling: %d, profile: %d\n",
			s->depth_chroma, s->depth_luma, s->subsampling, (int)s->profileIdc);

	VkVideoChromaSubsamplingFlagsKHR chromaSubsampling = subsampling_to_vulkan_flag(s->subsampling);
	if (chromaSubsampling == VK_VIDEO_CHROMA_SUBSAMPLING_INVALID_KHR)
	{
		printf("[vulkan_decode] Got unsupported chroma subsampling!\n");
		*wrong_pixfmt = true;
		return false;
	}
	//NOTE: Otherwise vulkan video fails to work (on my hardware)
	else if (chromaSubsampling != VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR)
	{
		printf("[vulkan_decode] Wrong chroma subsampling! Currently the only supported one is 4:2:0!\n");
		*wrong_pixfmt = true;
		return false;
	}

	VkVideoComponentBitDepthFlagBitsKHR vulkanChromaDepth = depth_to_vulkan_flag(s->depth_chroma);
	VkVideoComponentBitDepthFlagBitsKHR vulkanLumaDepth = depth_to_vulkan_flag(s->depth_luma);
	if (vulkanChromaDepth == VK_VIDEO_COMPONENT_BIT_DEPTH_INVALID_KHR ||
		vulkanLumaDepth == VK_VIDEO_COMPONENT_BIT_DEPTH_INVALID_KHR)
	{
		printf("[vulkan_decode] Got unsupported color channel depth!\n");
		*wrong_pixfmt = true;
		return false;
	}

	assert(s->codecOperation != VK_VIDEO_CODEC_OPERATION_NONE_KHR);
	bool isH264 = s->codecOperation == VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;

	//TODO interlacing
	VkVideoDecodeH264ProfileInfoKHR h264Profile = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_KHR,
													.stdProfileIdc = s->profileIdc,
													.pictureLayout = VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_PROGRESSIVE_KHR };
	VkVideoDecodeH265ProfileInfoKHR h265Profile = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PROFILE_INFO_KHR,
													.stdProfileIdc = STD_VIDEO_H265_PROFILE_IDC_MAIN //TODO H.265
													 };
	VkVideoProfileInfoKHR videoProfile = { .sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
										   .pNext = isH264 ? (void*)&h264Profile : (void*)&h265Profile,
										   .videoCodecOperation = s->codecOperation,
										   .chromaSubsampling = chromaSubsampling,
										   .lumaBitDepth = vulkanLumaDepth,
										   .chromaBitDepth = vulkanChromaDepth };
	
	VkVideoDecodeH264CapabilitiesKHR h264Capabilites = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_CAPABILITIES_KHR };
	VkVideoDecodeH265CapabilitiesKHR h265Capabilites = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_CAPABILITIES_KHR };
	VkVideoDecodeCapabilitiesKHR decodeCapabilities = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR,
														.pNext = isH264 ? (void*)&h264Capabilites : (void*)&h265Capabilites };
	VkVideoCapabilitiesKHR videoCapabilities = { .sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR,
												 .pNext = (void*)&decodeCapabilities };
	VkResult result = vkGetPhysicalDeviceVideoCapabilitiesKHR(s->physicalDevice, &videoProfile, &videoCapabilities);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to get physical device video capabilities!");
		if (result == VK_ERROR_OUT_OF_HOST_MEMORY) puts(" - Host out of memory.");
		else if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY) puts(" - Device out of memory.");
		else if (result == VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR) puts(" - Video picture layout not supported.");
		else if (result == VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR) puts(" - Video operation not supported.");
		else if (result == VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR) puts(" - Video format not supported.");
		else if (result == VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR) puts(" - Video codec not supported.");
		else printf(" - Vulkan error: %d\n", result);
		// it's not obvious when to set '*wrong_pixfmt = true;'
		return false;
	}

	//TODO allow dpb be implemented using only one VkImage
	if (!(videoCapabilities.flags & VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR))
	{
		printf("[vulkan_decode] Chosen physical device does not support separate reference images for DPB!\n");
		return false;
	}

	if (videoCapabilities.maxDpbSlots < MAX_REF_FRAMES + 1)
	{
		printf("[vulkan_decode] Chosen physical device does not support needed amount of DPB slots(%u)!\n",
				MAX_REF_FRAMES + 1);
		return false;
	}

	if (videoCapabilities.maxActiveReferencePictures < MAX_REF_FRAMES)
	{
		printf("[vulkan_decode] Chosen physical device does not support needed amount of active reference pictures(%u)!\n",
				MAX_REF_FRAMES);
		return false;
	}

	const VkExtent2D videoSize = { s->width, s->height };
	if (!does_video_size_fit(videoSize, videoCapabilities.minCodedExtent, videoCapabilities.maxCodedExtent))
	{
		printf("[vulkan_decode] Requested video size: %ux%u does not fit in vulkan video extents."
			   " Min extent: %ux%u, max extent: %ux%u\n",
				videoSize.width, videoSize.height,
				videoCapabilities.minCodedExtent.width, videoCapabilities.minCodedExtent.height,
				videoCapabilities.maxCodedExtent.width, videoCapabilities.maxCodedExtent.height);
		return false;
	}

	VkVideoDecodeCapabilityFlagsKHR decodeCapabilitiesFlags = decodeCapabilities.flags;
	VkVideoFormatPropertiesKHR pictureFormatProperites = { 0 }, referencePictureFormatProperties = { 0 };
	VkFormat pictureFormat = VK_FORMAT_UNDEFINED, referencePictureFormat = VK_FORMAT_UNDEFINED;

	VkVideoProfileListInfoKHR videoProfileList = { .sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR,
												   .profileCount = 1,
												   .pProfiles = &videoProfile };
	VkPhysicalDeviceVideoFormatInfoKHR videoFormatInfo = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_FORMAT_INFO_KHR,
														   .pNext = (void*)&videoProfileList };

	assert(s->physicalDevice != VK_NULL_HANDLE);

	if (decodeCapabilitiesFlags & VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR)
	{
		// usage from create_dpb
		videoFormatInfo.imageUsage = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR |
								  	 VK_IMAGE_USAGE_VIDEO_DECODE_SRC_BIT_KHR |
									 VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR |
									 VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		// err msg should be printed inside of check_for_vulkan_format function 
		if (!check_for_vulkan_format(s->physicalDevice, videoFormatInfo, &referencePictureFormatProperties))
				return false;

		pictureFormatProperites = referencePictureFormatProperties;
		referencePictureFormat = referencePictureFormatProperties.format;
		pictureFormat = pictureFormatProperites.format;
	}
	else if (decodeCapabilitiesFlags & VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_DISTINCT_BIT_KHR)
	{
		//TODO handlle this case as well
		printf("[vulkan_decode] Currently it is required for physical decoder to support DPB slot being the output as well!\n");
		return false;

		/*videoFormatInfo.imageUsage = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR;
		// err msg should be printed inside of check_for_vulkan_format function 
		if (!check_for_vulkan_format(s->physicalDevice, videoFormatInfo, s->out_codec, &referencePictureFormatProperties))
				return false;
		videoFormatInfo.imageUsage = VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR;
		if (!check_for_vulkan_format(s->physicalDevice, videoFormatInfo, s->out_codec, &pictureFormatProperites))
				return false;

		referencePictureFormat = referencePictureFormatProperties.format;
		pictureFormat = pictureFormatProperites.format;*/
	}
	else
	{
		printf("[vulkan_decode] Unsupported decodeCapabilitiesFlags value (%d)!\n", decodeCapabilitiesFlags);
		return false;
	}

	assert(pictureFormat != VK_FORMAT_UNDEFINED);
	assert(referencePictureFormat != VK_FORMAT_UNDEFINED);

	//TODO in the future make these (possibly) separate
	assert(pictureFormat == referencePictureFormat);
	s->dpbFormat = referencePictureFormat;

	// ---Creating synchronization fence---
	assert(s->device != VK_NULL_HANDLE);
	assert(s->fence == VK_NULL_HANDLE);

	VkFenceCreateInfo fenceInfo = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	result = vkCreateFence(s->device, &fenceInfo, NULL, &s->fence);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to create vulkan fence for synchronization!\n");
		return false;
	}

	// ---Creating decodeBuffer for NAL units and dstPicBuffer for decoded image---
	if (!allocate_buffers(s, videoProfileList, videoCapabilities))
	{
		// err msg should get printed inside of allocate_buffers
		if (vkDestroyFence != NULL)
		{
			vkDestroyFence(s->device, s->fence, NULL);
			s->fence = VK_NULL_HANDLE;
		}

		return false;
	}

	// ---Creating command pool---
	assert(s->commandPool == VK_NULL_HANDLE);

	VkCommandPoolCreateInfo poolInfo = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
										 .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
										 .queueFamilyIndex = s->queueFamilyIdx };
	result = vkCreateCommandPool(s->device, &poolInfo, NULL, &s->commandPool);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to create vulkan command pool!\n");
		free_buffers(s);
		if (vkDestroyFence != NULL)
		{
			vkDestroyFence(s->device, s->fence, NULL);
			s->fence = VK_NULL_HANDLE;
		}

		return false;
	}

	// ---Allocate command buffer---
	VkCommandBufferAllocateInfo cmdBufferInfo = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
												  .commandPool = s->commandPool,
												  .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
												  .commandBufferCount = 1 };
	result = vkAllocateCommandBuffers(s->device, &cmdBufferInfo, &s->cmdBuffer);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to allocate vulkan command buffer!\n");
		if (vkDestroyCommandPool != NULL)
		{
			vkDestroyCommandPool(s->device, s->commandPool, NULL);
			s->commandPool = VK_NULL_HANDLE;
		}
		free_buffers(s);
		if (vkDestroyFence != NULL)
		{
			vkDestroyFence(s->device, s->fence, NULL);
			s->fence = VK_NULL_HANDLE;
		}

		return false;
	}

	// ---Creating DPB (decoded picture buffer)---
	if (!create_dpb(s, &videoProfileList))
	{
		// err msg should get printed inside of create_dpb
		if (vkDestroyCommandPool != NULL)
		{
			vkDestroyCommandPool(s->device, s->commandPool, NULL);
			s->commandPool = VK_NULL_HANDLE;
		}
		free_buffers(s);
		if (vkDestroyFence != NULL)
		{
			vkDestroyFence(s->device, s->fence, NULL);
			s->fence = VK_NULL_HANDLE;
		}

		return false;
	}

	// ---Creating video session---
	assert(s->videoSession == VK_NULL_HANDLE);

	VkVideoSessionCreateInfoKHR sessionInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR,
												.pNext = NULL,
												.queueFamilyIndex = s->queueFamilyIdx,
												.flags = 0,
												.pVideoProfile = &videoProfile,
												.pictureFormat = pictureFormat,
												.maxCodedExtent = (VkExtent2D){ s->width, s->height },
												.referencePictureFormat = referencePictureFormat,
												.maxDpbSlots = MAX_REF_FRAMES + 1,
												.maxActiveReferencePictures = MAX_REF_FRAMES,
												.pStdHeaderVersion = &videoCapabilities.stdHeaderVersion };
	result = vkCreateVideoSessionKHR(s->device, &sessionInfo, NULL, &s->videoSession);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to create vulkan video session!\n");
		destroy_dpb(s);
		if (vkDestroyCommandPool != NULL)
		{
			vkDestroyCommandPool(s->device, s->commandPool, NULL);
			s->commandPool = VK_NULL_HANDLE;
		}
		free_buffers(s);
		if (vkDestroyFence != NULL)
		{
			vkDestroyFence(s->device, s->fence, NULL);
			s->fence = VK_NULL_HANDLE;
		}

		return false;
	}

	s->resetVideoCoding = true;

	// ---Creating video session parameters---
	//TODO probably useless to create them in prepare
	assert(s->videoSession != VK_NULL_HANDLE);

	VkVideoDecodeH264SessionParametersCreateInfoKHR h264SessionParamsInfo =
					{ .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,
					  .maxStdSPSCount = MAX_SPS_IDS,
					  .maxStdPPSCount = MAX_PPS_IDS,
					  .pParametersAddInfo = NULL };
	VkVideoDecodeH265SessionParametersCreateInfoKHR h265SessionParamsInfo =
					{ .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_CREATE_INFO_KHR,
					  .maxStdVPSCount = MAX_VPS_IDS,
					  .maxStdSPSCount = MAX_SPS_IDS,
					  .maxStdPPSCount = MAX_PPS_IDS,
					  .pParametersAddInfo = NULL };
	VkVideoSessionParametersCreateInfoKHR sessionParamsInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
																.pNext = isH264 ?
																			(void*)&h264SessionParamsInfo :
																			(void*)&h265SessionParamsInfo,
																.videoSessionParametersTemplate = VK_NULL_HANDLE,
																.videoSession = s->videoSession };
	result = vkCreateVideoSessionParametersKHR(s->device, &sessionParamsInfo, NULL, &s->videoSessionParams);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to create vulkan video session parameters!\n");
		if (vkDestroyVideoSessionKHR != NULL)
		{
			vkDestroyVideoSessionKHR(s->device, s->videoSession, NULL);
			s->videoSession = VK_NULL_HANDLE;
		}
		destroy_dpb(s);
		if (vkDestroyCommandPool != NULL)
		{
			vkDestroyCommandPool(s->device, s->commandPool, NULL);
			s->commandPool = VK_NULL_HANDLE;
		}
		free_buffers(s);
		if (vkDestroyFence != NULL)
		{
			vkDestroyFence(s->device, s->fence, NULL);
			s->fence = VK_NULL_HANDLE;
		}
		
		return false;
	}
	//s->videoSessionParams_update_count = 0; // update sequence count for video params must be resetted to zero

	// ---Allocation needed device memory and binding it to current video session---
	if (!allocate_memory_for_video_session(s))
	{
		//err msg should get printed inside of allocate_memory_for_video_session
		if (vkDestroyVideoSessionParametersKHR != NULL)
		{
			vkDestroyVideoSessionParametersKHR(s->device, s->videoSessionParams, NULL);
			s->videoSessionParams = VK_NULL_HANDLE;
		}
		if (vkDestroyVideoSessionKHR != NULL)
		{
			vkDestroyVideoSessionKHR(s->device, s->videoSession, NULL);
			s->videoSession = VK_NULL_HANDLE;
		}
		destroy_dpb(s);
		if (vkDestroyCommandPool != NULL)
		{
			vkDestroyCommandPool(s->device, s->commandPool, NULL);
			s->commandPool = VK_NULL_HANDLE;
		}
		free_buffers(s);
		if (vkDestroyFence != NULL)
		{
			vkDestroyFence(s->device, s->fence, NULL);
			s->fence = VK_NULL_HANDLE;
		}

		return false;
	}

	printf("Preparation successful.\n");
	return true;
}

static slice_info_t get_ref_slot_from_queue(struct state_vulkan_decompress *s, uint32_t index)
{
	// returns the member of reference frames queue on the given index
	// correctly handles wrapping, NO bounds checks!
	uint32_t wrappedIdx = (s->referenceSlotsQueue_start + index) % MAX_REF_FRAMES;
	return s->referenceSlotsQueue[wrappedIdx];
}

static uint32_t smallest_dpb_index_not_in_queue(struct state_vulkan_decompress *s)
{
	// returns the smallest index into DPB that's not in the reference queue
	// such index must exist as the reference queue is smaller by at least 1
	bool checks[MAX_REF_FRAMES + 1] = { 0 };

	for (uint32_t i = 0; i < s->referenceSlotsQueue_count; ++i)
	{
		uint32_t dpbIndex = get_ref_slot_from_queue(s, i).dpbIndex;
		
		assert(dpbIndex < MAX_REF_FRAMES + 1);
		assert(!checks[dpbIndex]);

		checks[dpbIndex] = true;
	}

	for (uint32_t i = 0; i < MAX_REF_FRAMES + 1; ++i)
	{
		if (!checks[i]) return i;
	}

	assert(!"Failed to find smallest dpb index that's not in reference picture queue!"); // should not happen
	return 0;
}

static void insert_ref_slot_into_queue(struct state_vulkan_decompress *s, slice_info_t slice_info)
{
	// inserts given reference picture (it's slice info) into reference slot queue,
	// if the queue is full then before the insertion discard the first queue member,
	// note that the first member is the "oldest" one (aka should be the one with lowest frame_num)
	assert(s->referenceSlotsQueue_count <= MAX_REF_FRAMES); // we also assume that MAX_REF_FRAMES is larger than zero

	/*printf("\tInserting slice_info - frame_seq: %d, frame_num: %d, poc_lsb: %d, poc: %d, pps_id: %d, is_reference: %d, is_intra: %d, dpbIndex: %u\n",
			slice_info.frame_seq, slice_info.frame_num, slice_info.poc_lsb, slice_info.poc, slice_info.pps_id,
			(int)(slice_info.is_reference), slice_info.is_intra, slice_info.dpbIndex);*/

	if (s->referenceSlotsQueue_count == MAX_REF_FRAMES) // queue full => discard the last member
	{
		s->referenceSlotsQueue_start = (s->referenceSlotsQueue_start + 1) % MAX_REF_FRAMES;
		--(s->referenceSlotsQueue_count); // correct the size of the queue
	}

	uint32_t wrappedIdx = (s->referenceSlotsQueue_start + s->referenceSlotsQueue_count) % MAX_REF_FRAMES;
	s->referenceSlotsQueue[wrappedIdx] = slice_info;
	++(s->referenceSlotsQueue_count);
}

static void clear_the_ref_slot_queue(struct state_vulkan_decompress *s)
{
	s->referenceSlotsQueue_count = 0;
}

static void fill_ref_picture_infos(struct state_vulkan_decompress *s,
								   VkVideoReferenceSlotInfoKHR refInfos[], VkVideoPictureResourceInfoKHR picInfos[],
								   VkVideoDecodeH264DpbSlotInfoKHR h264SlotInfos[], StdVideoDecodeH264ReferenceInfo h264Infos[],								 
								   uint32_t max_count, bool isH264, uint32_t *out_count)
{
	// count is a size of both given arrays (should be at most same as MAX_REF_FRAMES)
	assert(max_count <= MAX_REF_FRAMES);
	assert(isH264); //TODO H.265

	VkExtent2D videoSize = { s->width, s->height };

	uint32_t ref_count = s->referenceSlotsQueue_count;

	for (uint32_t i = 0; i < max_count && i < ref_count; ++i)
	{
		slice_info_t slice_info = get_ref_slot_from_queue(s, i);
		/*printf("\tGot slice_info - frame_seq: %d, frame_num: %d, poc_lsb: %d, poc: %d, pps_id: %d, is_reference: %d, is_intra: %d, dpbIndex: %u\n",
			slice_info.frame_seq, slice_info.frame_num, slice_info.poc_lsb, slice_info.poc, slice_info.pps_id,
			(int)(slice_info.is_reference), slice_info.is_intra, slice_info.dpbIndex);*/
		
		assert(slice_info.frame_num >= 0);
		assert(slice_info.dpbIndex < MAX_REF_FRAMES + 1);

		VkImageView view = s->dpbViews[slice_info.dpbIndex];
		assert(view != VK_NULL_HANDLE);

		h264Infos[i] = (StdVideoDecodeH264ReferenceInfo){ .flags = { 0 }, .FrameNum = slice_info.frame_num,
														  .PicOrderCnt = { slice_info.poc, slice_info.poc, } };
		h264SlotInfos[i] = (VkVideoDecodeH264DpbSlotInfoKHR){ .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,
															  .pStdReferenceInfo = h264Infos + i };
		picInfos[i] = (VkVideoPictureResourceInfoKHR){ .sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
													   .codedOffset = { 0, 0 }, // no offset
													   .codedExtent = videoSize,
													   .baseArrayLayer = 0,
													   .imageViewBinding = view };
		refInfos[i] = (VkVideoReferenceSlotInfoKHR){ .sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
													 //TODO H.265
													 //.pNext = isH264 ? (void*)&h264RefInfo : (void*)&h265RefInfo,
													 .pNext = (void*)(h264SlotInfos + i),
													 .slotIndex = slice_info.dpbIndex,
													 .pPictureResource = picInfos + i };
	}

	if (out_count != NULL) *out_count = ref_count;
}

static void begin_video_coding_scope(struct state_vulkan_decompress *s, slice_info_t *slice_info)
{
	assert(slice_info->frame_num >= 0 && slice_info->poc_lsb >= 0 &&
		   slice_info->pps_id >= 0 && slice_info->sps_id >=0);

	bool isH264 = s->codecOperation == VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
	const VkExtent2D videoSize = { s->width, s->height };

	
	// ---Filling the extisting references infos---
	uint32_t slotInfos_count = 0;
	StdVideoDecodeH264ReferenceInfo h264StdInfos[MAX_REF_FRAMES + 1] = { 0 };
	VkVideoDecodeH264DpbSlotInfoKHR h264SlotInfos[MAX_REF_FRAMES + 1] = { 0 };
	VkVideoPictureResourceInfoKHR picInfos[MAX_REF_FRAMES + 1] = { 0 };
	VkVideoReferenceSlotInfoKHR slotInfos[MAX_REF_FRAMES + 1] = { 0 };

	fill_ref_picture_infos(s, slotInfos, picInfos, h264SlotInfos, h264StdInfos, MAX_REF_FRAMES, isH264, &slotInfos_count);
	assert(slotInfos_count <= MAX_REF_FRAMES);

	// ---Filling the info of currently decoded picture---
	assert(slice_info->sps_id < MAX_SPS_IDS); //TODO if

	sps_t *sps = s->sps_array + slice_info->sps_id;
	//printf("\tprev_msb: %d, prev_lsb: %d\n", s->prev_poc_msb, s->prev_poc_lsb);
	slice_info->poc = get_picture_order_count(&s->prev_poc_msb, &s->prev_poc_lsb,
											  sps->log2_max_pic_order_cnt_lsb_minus4, slice_info->poc_lsb);
	//printf("\tcalculated poc: %d display_frame_seq: %d\n", slice_info->poc, slice_info->frame_seq - s->idr_frame_seq);

	h264StdInfos[slotInfos_count] = (StdVideoDecodeH264ReferenceInfo){ .flags = { 0 },
									  								   .FrameNum = slice_info->frame_num,
									  								   .PicOrderCnt = { slice_info->poc, slice_info->poc } };
	h264SlotInfos[slotInfos_count] = (VkVideoDecodeH264DpbSlotInfoKHR){ .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,
									   									.pStdReferenceInfo = h264StdInfos + slotInfos_count };
	picInfos[slotInfos_count] = (VkVideoPictureResourceInfoKHR){ .sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
																 .codedOffset = { 0, 0 }, // empty offset
																 .codedExtent = videoSize,
																 .baseArrayLayer = 0,
																 .imageViewBinding = s->dpbViews[slice_info->dpbIndex] };
	VkVideoDecodeH265DpbSlotInfoKHR h265SlotInfo = {}; //TODO H.265
	slotInfos[slotInfos_count] = (VkVideoReferenceSlotInfoKHR){ .sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
								   								.pNext = isH264 ? (void*)(h264SlotInfos + slotInfos_count)
																				: (void*)&h265SlotInfo,
								   								.slotIndex = -1, // currently decoded picture must have index -1
								   								.pPictureResource = picInfos + slotInfos_count };
	++slotInfos_count; // updating the count for currently added decoded picture

	VkVideoBeginCodingInfoKHR beginCodingInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR,
												  .flags = 0,
												  .videoSession = s->videoSession,
												  .videoSessionParameters = s->videoSessionParams,
												  .referenceSlotCount = slotInfos_count,
												  .pReferenceSlots = slotInfos };
	vkCmdBeginVideoCodingKHR(s->cmdBuffer, &beginCodingInfo);

	if (s->resetVideoCoding) // before the first use of vulkan video session we must reset the video coding for that session
	{
		VkVideoCodingControlInfoKHR vidCodingControlInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR,
															 .flags = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR }; // reset bit
		vkCmdControlVideoCodingKHR(s->cmdBuffer, &vidCodingControlInfo);

		s->resetVideoCoding = false;
	}
}

static void end_video_coding_scope(struct state_vulkan_decompress *s)
{
	VkVideoEndCodingInfoKHR endCodingInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR };
	vkCmdEndVideoCodingKHR(s->cmdBuffer, &endCodingInfo);
}

static void print_bits(unsigned char num) //DEBUG
{
	unsigned int bit = 1<<(sizeof(unsigned char) *8 - 1);

	while (bit)
	{
		printf("%i ", num & bit ? 1 : 0);
		bit >>= 1;
	}
}

//copied from rtp/rtpenc_h264.c
static uint32_t get4Bytes(const unsigned char *ptr)
{
	return (ptr[0] << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3];
}

static const unsigned char * skip_nal_start_code(const unsigned char *nal)
{
	//caller should assert that nal stream is at least 4 bytes long
	uint32_t next4Bytes = get4Bytes(nal);

	if (next4Bytes == 0x00000001) return nal + 4;
	else if ((next4Bytes & 0xFFFFFF00) == 0x00000100) return nal + 3;
	else return NULL;
}

//copied from rtp/rtpenc_h264.c
static const unsigned char * get_next_nal(const unsigned char *start, long len, bool with_start_code)
{
	const unsigned char * const stop = start + len;
	while (stop - start >= 4) {
		uint32_t next4Bytes = get4Bytes(start);
		if (next4Bytes == 0x00000001) {
			return start + (with_start_code ? 0 : 4);
		}
		if ((next4Bytes & 0xFFFFFF00) == 0x00000100) {
			return start + (with_start_code ? 0 : 3);
		}
		// We save at least some of "next4Bytes".
		if ((unsigned) (next4Bytes & 0xFF) > 1) {
			// Common case: 0x00000001 or 0x000001 definitely doesn't begin anywhere in "next4Bytes", so we save all of it:
			start += 4;
		} else {
			// Save the first byte, and continue testing the rest:
			start += 1;
		}
	}
	return NULL;
}

//copied from rtp/rtpenc_h264.c
static void print_nalu_name(int type)
{
	switch ((enum nal_type)type)
	{
		case NAL_H264_NONIDR:
			printf("H264 NON-IDR");
			break;
		case NAL_H264_IDR:
			printf("H264 IDR");
			break;
		case NAL_H264_SEI:
			printf("H264 SEI");
			break;
		case NAL_H264_SPS:
			printf("H264 SPS");
			break;
		case NAL_H264_PPS:
			printf("H264 PPS");
			break;
		case NAL_H264_AUD:
			printf("H264 AUD");
			break;
		case NAL_HEVC_VPS:
			printf("HEVC VPS");
			break;
		case NAL_HEVC_SPS:
			printf("HEVC SPS");
			break;
		case NAL_HEVC_PPS:
			printf("HEVC PPS");
			break;
		case NAL_HEVC_AUD:
			printf("HEVC AUD");
			break;
    }
}

static void print_nalu_header(unsigned char header)
{
	printf("forbidden bit: %u, idc: %u, type: %u", header & 128 ? 1 : 0,
				H264_NALU_HDR_GET_NRI(header), H264_NALU_HDR_GET_TYPE(header));
}

static bool create_rbsp(const unsigned char *nal, size_t nal_len, uint8_t **rbsp, int *rbsp_len)
{
	// allocates memory for rbsp and fills it with correct rbsp corresponding to given nal buffer range
	// returns false if error and prints error msg
	assert(nal != NULL && rbsp != NULL && rbsp_len != NULL);

	if (sizeof(unsigned char) != sizeof(uint8_t))
	{
		printf("[vulkan_decode] H.264/H.265 stream handling requires sizeof(unsigned char) == sizeof(uint8_t)!\n");
		*rbsp_len = 0;
		return false;
	}

	*rbsp_len = (int)nal_len;
	assert(*rbsp_len >= 0);
	uint8_t *out = (uint8_t*)malloc(nal_len * sizeof(uint8_t));
	if (out == NULL)
	{
		printf("[vulkan_decode] Failed to allocate memory for RBSP stream data!\n");
		*rbsp_len = 0;
		return false;
	}

	// throw away variable, could be set to 'nal_len' instead, but we are using the fact that 'rbsp_len' is already converted
	int nal_payload_len = *rbsp_len;
	// the cast here to (const uint8_t*) is valid because of the first 'if'
	int ret = nal_to_rbsp((const uint8_t*)nal, &nal_payload_len, out, rbsp_len);
	if (ret == -1)
	{
		printf("[vulkan_decode] Failed to convert NALU stream into RBSP data!\n");
		free(out);
		*rbsp_len = 0;
		return false;
	}

	assert(ret == *rbsp_len);
	*rbsp = out;
	
	return true;
}

static void destroy_rbsp(uint8_t *rbsp)
{
	// destroys rbsp previously created with create_rbsp function
	// (just frees the memory, maybe pointless)
	free(rbsp);
}

static bool get_video_info_from_sps(struct state_vulkan_decompress *s, const unsigned char *sps_src, size_t sps_src_len)
{
	uint8_t *rbsp = NULL;
	int rbsp_len = 0;

	if (!create_rbsp(sps_src, sps_src_len, &rbsp, &rbsp_len))
	{
		//err should get printed inside of create_rbsp
		return false;
	}
	assert(rbsp != NULL);

	sps_t sps = { 0 };
	bs_t b = { 0 };
	bs_init(&b, rbsp, rbsp_len);

	read_sps(&sps, &b);
	destroy_rbsp(rbsp);

	bool isH264 = s->codecOperation == VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;

	s->depth_chroma = sps.bit_depth_chroma_minus8 + 8;
	s->depth_luma = sps.bit_depth_luma_minus8 + 8;
	s->subsampling = isH264 ? h264_flag_to_subsampling((StdVideoH264ChromaFormatIdc)sps.chroma_format_idc)
							: h265_flag_to_subsampling((StdVideoH265ChromaFormatIdc)sps.chroma_format_idc);
	s->profileIdc = profile_idc_to_h264_flag(sps.profile_idc); //TODO H.265

	//printf("Profile IDC - sps: %d, state: %d\n", sps.profile_idc, (int)s->profileIdc);

	if (s->profileIdc == STD_VIDEO_H264_PROFILE_IDC_INVALID ||
		s->depth_chroma < 8 ||
		s->depth_luma < 8 ||
		s->subsampling <= 0)
	{
		s->depth_chroma = 0;
		s->depth_luma = 0;
		s->subsampling = 0;
		s->profileIdc = STD_VIDEO_H264_PROFILE_IDC_INVALID;

		return false;
	}

	return true;
}

static bool find_first_sps_vps(struct state_vulkan_decompress *s, const unsigned char *src, size_t src_len)
{
	// looks if the input stream contains SPS (or VPS for H.265) NAL units
	// if it does then it fill video information into the decompress state and returns true,
	// otherwise returns false and prints error msg if error happened
	bool isH264 = s->codecOperation == VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;

	const unsigned char *nal = get_next_nal(src, src_len, true), *next_nal = NULL;
	while (nal != NULL)
	{
		const unsigned char *nal_payload = skip_nal_start_code(nal);
		if (nal_payload == NULL) return false;

		next_nal = get_next_nal(nal_payload, src_len - (nal_payload - src), true);
		size_t nal_len = next_nal == NULL ? src_len - (nal - src) : next_nal - nal;
		if (nal_len <= 4 || (size_t)(nal_payload - nal) >= nal_len) //TODO weird?
		{
			printf("[vulkan_decode] NAL unit is too short.\n");
			return false;
		}

		size_t nal_payload_len = nal_len - (nal_payload - nal); //should be non-zero now

		int nalu_type = NALU_HDR_GET_TYPE(nal_payload[0], !isH264);
		if (isH264 &&
			nalu_type == NAL_H264_SPS)
		{
			//TODO
			if (!get_video_info_from_sps(s, nal_payload, nal_payload_len))
			{
				printf("[vulkan_decode] Found first SPS, but it was invalid! Discarting it.\n");
				return false;
			}

			return true;
		}
		else if (!isH264 &&
				 (nalu_type == NAL_HEVC_SPS || nalu_type == NAL_HEVC_VPS))
		{
			return true;
		}

		nal = next_nal;
	}

	return false;
}

static void * begin_nalu_writing(struct state_vulkan_decompress *s)
{
	void *memoryPtr = NULL;

	VkResult result = vkMapMemory(s->device, s->bufferMemory, 0, s->decodeBufferSize, 0, &memoryPtr);
	if (result != VK_SUCCESS) return NULL;

	memset(memoryPtr, 0, s->decodeBufferSize);

	return memoryPtr;
}

static VkDeviceSize write_nalu(unsigned char *nalu_buffer, size_t nalu_buffer_len,
					    	   const unsigned char *nal, size_t len)
{
	// writes one given NAL unit into NAL buffer, if error (not enough space in buffer) returns 0
	assert(nalu_buffer != NULL && nal != NULL && len > 4);
	
	// check if enough space for nal unit in the buffer, return zero if not
	// (zero => error as we always want to write at least 4 bytes anyway)
	if (nalu_buffer_len < len) return 0;

	VkDeviceSize result_len = (VkDeviceSize)len;
	// check if the cast is correct
	if (sizeof(VkDeviceSize) < sizeof(size_t) && (size_t)result_len != len) return 0;

	memcpy(nalu_buffer, nal, len);

	return result_len;
}

static void end_nalu_writing(struct state_vulkan_decompress *s)
{
	if (vkUnmapMemory == NULL || s->device == VK_NULL_HANDLE) return;

	vkUnmapMemory(s->device, s->bufferMemory);
}

static bool write_decoded_frame(struct state_vulkan_decompress *s, VkDeviceSize buffer_len, unsigned char *dst)
{
	assert(sizeof(unsigned char) == sizeof(uint8_t)); //DEBUG ?
	assert(s->device != VK_NULL_HANDLE);
	assert(s->bufferMemory != VK_NULL_HANDLE);
	//assert(buffer_len > 0);

	UNUSED(buffer_len); //DEBUG
	//460800 = 640*480 + 153600 = 640*480 + 76800 + 76800 = 640*480 + 640*480/4 + 640*480/4
	VkDeviceSize lumaSize = s->width * s->height;
	VkDeviceSize chromaSize = lumaSize / 4;
	VkDeviceSize size = lumaSize + 2 * chromaSize;

	//printf("\t%d = %d + 2 * %d\n", size, lumaSize, chromaSize);

	uint8_t *buffer_data = NULL;
	VkResult result = vkMapMemory(s->device, s->bufferMemory, s->dstPicBufferMemoryOffset, size, 0, (void**)&buffer_data);
	if (result != VK_SUCCESS) return false;

	assert(buffer_data != NULL);
	assert(dst != NULL);

	//memset(dst, 0x99, (firstPlaneSize * sizeof(unsigned char)));
	//memcpy(dst, buffer_data, firstPlaneSize);
	/*for (size_t i = 0; i < firstPlaneSize; ++i)
	{
		dst[i] = (unsigned char)buffer_data[i];
	}*/
	//memset(dst + firstPlaneSize, 0xff, (secondPlaneSize * sizeof(unsigned char))/3);
	//memcpy(dst + firstPlaneSize, buffer_data + firstPlaneSize, secondPlaneSize);

	//DEBUG - attempt at translating NV12 into I420
	// luma plane
	for (size_t i = 0; i < lumaSize; ++i) dst[i] = (unsigned char)buffer_data[i];

	// chroma plane
	for (size_t i = 0; i < chromaSize; ++i)
	{
		unsigned char Cb = (unsigned char)buffer_data[lumaSize + 2*i],
					  Cr = (unsigned char)buffer_data[lumaSize + 2*i + 1];

		dst[lumaSize + i] = Cb;
		dst[lumaSize + chromaSize + i] = Cr;
	}

	vkUnmapMemory(s->device, s->bufferMemory);

	return true;
}

static bool update_video_session_params(struct state_vulkan_decompress *s, bool isH264,
										sps_t *added_sps,// uint32_t added_sps_count,
										pps_t *added_pps)// uint32_t added_pps_count)
{
	assert(s->device != VK_NULL_HANDLE && s->videoSession != VK_NULL_HANDLE);

	StdVideoH264SequenceParameterSet vk_sps = { 0 };
	StdVideoH264SequenceParameterSetVui vk_vui = { 0 };
	StdVideoH264HrdParameters vk_hrd = { 0 };
	StdVideoH264PictureParameterSet vk_pps = { 0 };
	StdVideoH264ScalingLists vk_scalinglist = { 0 };

	uint32_t added_sps_count = 0;
	if (added_sps != NULL)
	{
		sps_to_vk_sps(added_sps, &vk_sps, &vk_vui, &vk_hrd);
		added_sps_count = 1;
	}

	uint32_t added_pps_count = 0;
	if (added_pps != NULL)
	{
		pps_to_vk_pps(added_pps, &vk_pps, &vk_scalinglist);
		added_pps_count = 1;
	}

	VkVideoDecodeH264SessionParametersAddInfoKHR h264AddInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,
																 .stdSPSCount = added_sps_count,
																 .pStdSPSs = &vk_sps,
																 .stdPPSCount = added_pps_count,
																 .pStdPPSs = &vk_pps };
	//TODO H.265
	VkVideoDecodeH265SessionParametersAddInfoKHR h265AddInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR,
																 //.stdSPSCount = added_sps_count,
																 //.pStdSPSs = &vk_sps,
																 //.stdPPSCount = added_pps_count,
																 //.pStdPPSs = &vk_pps
																};
	VkVideoDecodeH264SessionParametersCreateInfoKHR h264CreateInfo =
					{ .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,
					  .maxStdSPSCount = MAX_SPS_IDS,
					  .maxStdPPSCount = MAX_PPS_IDS,
					  .pParametersAddInfo = &h264AddInfo };
	//TODO H.265
	VkVideoDecodeH265SessionParametersCreateInfoKHR h265CreateInfo =
					{ .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_CREATE_INFO_KHR,
					  .maxStdVPSCount = MAX_VPS_IDS,
					  .maxStdSPSCount = MAX_SPS_IDS,
					  .maxStdPPSCount = MAX_PPS_IDS,
					  .pParametersAddInfo = &h265AddInfo };
	VkVideoSessionParametersCreateInfoKHR createInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
														 .pNext = isH264 ? (void*)&h264CreateInfo : (void*)&h265CreateInfo,
														 .videoSessionParametersTemplate = s->videoSessionParams,
														 .videoSession = s->videoSession };
	VkVideoSessionParametersKHR newVideoSessionParams = VK_NULL_HANDLE;
	VkResult result = vkCreateVideoSessionParametersKHR(s->device, &createInfo, NULL, &newVideoSessionParams);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to update vulkan video session parameters!\n");
		return false;
	}

	if (s->videoSessionParams != VK_NULL_HANDLE)
	{
		if (vkDestroyVideoSessionParametersKHR != NULL)
		{
			vkDestroyVideoSessionParametersKHR(s->device, s->videoSessionParams, NULL);
		}
		s->videoSessionParams = VK_NULL_HANDLE;
	}

	assert(newVideoSessionParams != VK_NULL_HANDLE);
	s->videoSessionParams = newVideoSessionParams;
	return true;
}

static bool handle_sps_nalu(struct state_vulkan_decompress *s, const unsigned char *nal_payload, size_t len, bool isH264)
{
	// reads sequence parameters set from given buffer range
	
	uint8_t *rbsp = NULL;
	int rbsp_len = 0;

	if (!create_rbsp(nal_payload, len, &rbsp, &rbsp_len))
	{
		//err should get printed inside of create_rbsp
		return false;
	}
	assert(rbsp != NULL);

	/*printf("reading pps - payload len: %u (converted: %d), rbsp_len: %d\n", len, nal_payload_len, rbsp_len);
	puts("nal_payload:");
	for (size_t i = 0; i < len && i < 30; ++i) printf("%u ", nal_payload[i]);
	putchar('\n');
	puts("rbsp:");
	for (size_t i = 0; i < rbsp_len && i < 30; ++i) printf("%u ", rbsp[i]);
	putchar('\n');

	puts("bits:");
	for (size_t i = 0; i < rbsp_len && i < 30; ++i)
	{
		print_bits((unsigned char)rbsp[i]);
		putchar('|');
	}
	putchar('\n');*/

	bs_t b = { 0 };
	sps_t sps = { 0 };

	bs_init(&b, rbsp, rbsp_len);
	
	//puts("bitstream:");
	//printf("ue1: %d, ue2: %d\n", bs_read_ue(&b), bs_read_ue(&b));
	//for (size_t i = 0; i < rbsp_len && i < 30; ++i) printf("%u ", bs_read_u8(&b));
	//putchar('\n');

	read_sps(&sps, &b);
	destroy_rbsp(rbsp);
	b = (bs_t){ 0 }; // just to be sure
	//print_sps(&sps);

	int id = sps.seq_parameter_set_id;
	if (id < 0 || id >= MAX_SPS_IDS)
	{
		printf("[vulkan_decode] Id of read SPS is out of bounds (%d)! Discarting it.\n", id);
		return false;
	}

	//TODO array is probably useless?
	assert(s->sps_array != NULL);
	s->sps_array[id] = sps;

	// potential err msg should get printed inside of update_video_session_params
	return update_video_session_params(s, isH264, &sps, NULL);
}

static bool handle_pps_nalu(struct state_vulkan_decompress *s, const unsigned char *nal_payload, size_t len, bool isH264)
{
	// reads picture parameters set from given buffer range
	
	uint8_t *rbsp = NULL;
	int rbsp_len = 0;

	if (!create_rbsp(nal_payload, len, &rbsp, &rbsp_len))
	{
		//err should get printed inside of create_rbsp
		return false;
	}
	assert(rbsp != NULL);

	/*printf("reading pps - payload len: %u (converted: %d), rbsp_len: %d\n", len, nal_payload_len, rbsp_len);
	puts("nal_payload:");
	for (size_t i = 0; i < len && i < 30; ++i) printf("%u ", nal_payload[i]);
	putchar('\n');
	puts("rbsp:");
	for (size_t i = 0; i < rbsp_len && i < 30; ++i) printf("%u ", rbsp[i]);
	putchar('\n');

	puts("bits:");
	for (size_t i = 0; i < rbsp_len && i < 30; ++i)
	{
		print_bits((unsigned char)rbsp[i]);
		putchar('|');
	}
	putchar('\n');*/

	bs_t b = { 0 };
	pps_t pps = { 0 };

	bs_init(&b, rbsp, rbsp_len);
	
	//puts("bitstream:");
	//printf("ue1: %d, ue2: %d\n", bs_read_ue(&b), bs_read_ue(&b));
	//for (size_t i = 0; i < rbsp_len && i < 30; ++i) printf("%u ", bs_read_u8(&b));
	//putchar('\n');

	read_pps(&pps, &b);
	destroy_rbsp(rbsp);
	b = (bs_t){ 0 }; // just to be sure
	//print_pps(&pps);

	int id = pps.pic_parameter_set_id;
	if (id < 0 || id >= MAX_PPS_IDS)
	{
		printf("[vulkan_decode] Id of read PPS is out of bounds (%d)! Discarting it.\n", id);
		destroy_rbsp(rbsp);
		return false;
	}

	//TODO array is probably useless?
	assert(s->pps_array != NULL);
	s->pps_array[id] = pps;

	// potential err msg should get printed inside of update_video_session_params
	return update_video_session_params(s, isH264, NULL, &pps);
}

static void fill_slice_info(struct state_vulkan_decompress *s, slice_info_t *si, const slice_header_t *sh)
{
	si->is_intra = sh->slice_type == SH_SLICE_TYPE_I || sh->slice_type == SH_SLICE_TYPE_I_ONLY;

	assert(si->idr_pic_id == -1 || si->idr_pic_id == sh->idr_pic_id);
	si->idr_pic_id = sh->idr_pic_id;

	assert(si->pps_id == -1 || si->pps_id == sh->pic_parameter_set_id);
	si->pps_id = sh->pic_parameter_set_id;

	//TODO check
	assert(sh->pic_parameter_set_id < MAX_PPS_IDS); //TODO if
	pps_t *pps = s->pps_array + sh->pic_parameter_set_id;
	int new_sps_id = pps->seq_parameter_set_id;

	assert(si->sps_id == -1 || si->sps_id == new_sps_id);
	si->sps_id = new_sps_id;

	assert(si->frame_num == -1 || si->frame_num == sh->frame_num);
	si->frame_num = sh->frame_num;

	assert(si->poc_lsb == -1 || si->poc_lsb == sh->pic_order_cnt_lsb);
	si->poc_lsb = sh->pic_order_cnt_lsb;
}

static bool handle_sh_nalu(struct state_vulkan_decompress *s, const unsigned char *nal_payload, size_t len,
						   int nal_type, int nal_idc, slice_info_t *slice_info)
{
	// reads slice header from given buffer range
	assert(slice_info != NULL);
	
	uint8_t *rbsp = NULL;
	int rbsp_len = 0;

	if (!create_rbsp(nal_payload, len, &rbsp, &rbsp_len))
	{
		//err should get printed inside of create_rbsp
		return false;
	}
	assert(rbsp != NULL);

	bs_t b = { 0 };
	slice_header_t sh = { 0 };

	bs_init(&b, rbsp, rbsp_len);

	assert(s->sps_array != NULL);
	assert(s->pps_array != NULL);

	if (!read_slice_header(&sh, nal_type, nal_idc, s->pps_array, s->sps_array, &b))
	{
		printf("[vulkan_decode] Encountered wrong SPS/PPS ids while reading a slice header! Discarting it.\n");
		destroy_rbsp(rbsp);
		return false;
	}
	//print_sh(&sh);
	fill_slice_info(s, slice_info, &sh);

	//TODO this - from ITU-T H.264 Specification, 7.4.3
	/*When present, the value of the slice header syntax elements pic_parameter_set_id, frame_num, field_pic_flag,
	bottom_field_flag, idr_pic_id, pic_order_cnt_lsb, delta_pic_order_cnt_bottom, delta_pic_order_cnt[ 0 ],
	delta_pic_order_cnt[ 1 ], sp_for_switch_flag, and slice_group_change_cycle shall be the same in all slice headers of a
	coded picture. */

	destroy_rbsp(rbsp);
	return true;
}

static void decode_frame(struct state_vulkan_decompress *s, slice_info_t slice_info, VkDeviceSize decodeBufferSize,
						 uint32_t slice_offsets[], uint32_t slice_offsets_count)
{
	//TODO
	bool isH264 = s->codecOperation == VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
	const VkExtent2D videoSize = { s->width, s->height };
	VkDeviceSize decodeBufferSizeAligned = (decodeBufferSize + (s->decodeBufferOffsetAlignment - 1))
										   & ~(s->decodeBufferOffsetAlignment - 1); //alignment bit mask magic
	//assert(slice_info.pps_id < MAX_PPS_IDS); //TODO if
	//pps_t *pps = s->pps_array + slice_info.pps_id;
	//assert(slice_info.sps_id < MAX_SPS_IDS); //TODO if
	//sps_t *sps = s->sps_array + slice_info.sps_id;
	
	assert(decodeBufferSizeAligned <= s->decodeBufferSize);
	assert(slice_offsets_count > 0);
	// for IDR pictures the id must be valid
	assert(!slice_info.is_intra || !slice_info.is_reference || slice_info.idr_pic_id >= 0);

	assert(isH264); //TODO H.265

	// ---Filling infos related to active reference pictures---
	// similar to filling of infos in begin_video_coding_scope
	uint32_t refSlotInfos_count = 0;
	StdVideoDecodeH264ReferenceInfo h264RefStdInfos[MAX_REF_FRAMES] = { 0 };
	VkVideoDecodeH264DpbSlotInfoKHR h264RefSlotInfos[MAX_REF_FRAMES] = { 0 };
	VkVideoPictureResourceInfoKHR refPicInfos[MAX_REF_FRAMES] = { 0 };
	VkVideoReferenceSlotInfoKHR refSlotInfos[MAX_REF_FRAMES] = { 0 };

	fill_ref_picture_infos(s, refSlotInfos, refPicInfos, h264RefSlotInfos, h264RefStdInfos,
						   MAX_REF_FRAMES, isH264, &refSlotInfos_count);

	// ---Filling infos related to currently decoded picture---
	VkVideoPictureResourceInfoKHR dstVideoPicInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
													  .codedOffset = { 0, 0 }, // empty offset
													  .codedExtent = videoSize,
													  .baseArrayLayer = 0,
													  .imageViewBinding = s->dpbViews[slice_info.dpbIndex] };
	StdVideoDecodeH264ReferenceInfo h264DstStdInfo = { .flags = { 0 },
													   .FrameNum = slice_info.frame_num,
													   .PicOrderCnt = { slice_info.poc, slice_info.poc } };
	VkVideoDecodeH264DpbSlotInfoKHR h264DstSlotInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,
														.pStdReferenceInfo = &h264DstStdInfo };
	VkVideoDecodeH265DpbSlotInfoKHR h265DstSlotInfo = {}; //TODO H.265
	VkVideoReferenceSlotInfoKHR dstSlotInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
												.pNext = isH264 ? (void*)&h264DstSlotInfo : (void*)&h265DstSlotInfo,
												.slotIndex = slice_info.dpbIndex,
												.pPictureResource = &dstVideoPicInfo };

	// ---Filling infos related to decodeBuffer---
	StdVideoDecodeH264PictureInfo h264DecodeStdInfo = { .flags = { .field_pic_flag = 0,
																   .is_intra = slice_info.is_intra,
																   .is_reference = slice_info.is_reference,
																    //TODO this could be potentially wrong,
																	//maybe itroduce new member to slice_info_t for it
																   .IdrPicFlag = slice_info.is_intra && slice_info.is_reference,
																   },
														.seq_parameter_set_id = slice_info.sps_id,
														.pic_parameter_set_id = slice_info.pps_id,
														.frame_num = slice_info.frame_num,
														.idr_pic_id = slice_info.idr_pic_id,
														.PicOrderCnt = { slice_info.poc, slice_info.poc } };
	VkVideoDecodeH264PictureInfoKHR h264DecodeInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PICTURE_INFO_KHR,
													   .pStdPictureInfo = &h264DecodeStdInfo,
													   .sliceCount = slice_offsets_count,
													   .pSliceOffsets = slice_offsets };
	//TODO H.265
	VkVideoDecodeH265PictureInfoKHR h265DecodeInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PICTURE_INFO_KHR };
	VkVideoDecodeInfoKHR decodeInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR,
										.pNext = isH264 ? (void*)&h264DecodeInfo : (void*)&h265DecodeInfo,
										.flags = 0,
										.srcBuffer = s->decodeBuffer,
										.srcBufferOffset = 0,
										.srcBufferRange = decodeBufferSizeAligned,
										.dstPictureResource = dstVideoPicInfo,
										// this must be the same as dstPictureResource when VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR
										// otherwise must not be the same as dstPictureResource
										.pSetupReferenceSlot = &dstSlotInfo,
										//specifies the needed used references (but not the decoded frame)
										.referenceSlotCount = refSlotInfos_count,
										.pReferenceSlots = refSlotInfos };
	vkCmdDecodeVideoKHR(s->cmdBuffer, &decodeInfo);
}

static decompress_status vulkan_decompress(void *state, unsigned char *dst, unsigned char *src,
                unsigned int src_len, int frame_seq, struct video_frame_callbacks *callbacks,
                struct pixfmt_desc *internal_prop)
{
	//printf("vulkan_decode - decompress\n");
	UNUSED(callbacks);
	struct state_vulkan_decompress *s = (struct state_vulkan_decompress *)state;
    //printf("\tdst: %p, src: %p, src_len: %u, frame_seq: %d\n",
	//		dst, src, src_len, frame_seq);
	
    decompress_status res = DECODER_NO_FRAME;

	if (s->out_codec == VIDEO_CODEC_NONE)
	{
		printf("\tProbing...\n");

		*internal_prop = get_pixfmt_desc(I420);
		return DECODER_GOT_CODEC;
	}

	if (src_len <= 5)
	{
		printf("[vulkan_decode] Source buffer too short!\n");
		return res;
	}

	/*unsigned format = src[src_len - 2];
	int av_depth = 8 + (format >> 4) * 2;
	int subs_a = ((format >> 2) & 0x3) + 1;
	int subs_b = ((format >> 1) & 0x1) * subs_a;
	int av_subsampling = 4000 + subs_a * 100 + subs_b * 10;
	int av_rgb = format & 0x1;
	printf("forced pixfmt - depth: %d, subs: %d, rg: %d\n", av_depth, av_subsampling, av_rgb);*/

	assert(s->codecOperation != VK_VIDEO_CODEC_OPERATION_NONE_KHR);

	if (!s->sps_vps_found && !find_first_sps_vps(s, src, src_len))
	{
		puts("\tStill no SPS or VPS found.");
		return res;
	}
	s->sps_vps_found = true;

	bool wrong_pixfmt = false;
	if (!s->prepared && !prepare(s, &wrong_pixfmt))
	{
		if (wrong_pixfmt)
		{
			printf("\tFailed to prepare for decompress - wrong pixel format.\n");
			return DECODER_UNSUPP_PIXFMT;
		}

		printf("\tFailed to prepare for decompress.\n");
		return res;
	}
	s->prepared = true;

	int prev_frame_seq = s->current_frame_seq;
	s->current_frame_seq = frame_seq;
	if (frame_seq > prev_frame_seq + 1)
	{
		puts("Missed frame!");
		clear_the_ref_slot_queue(s);
	}

	bool isH264 = s->codecOperation == VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
	// we hope to fill those values from NAL slice units
	slice_info_t slice_info = { .is_reference = false, .is_intra = false, .idr_pic_id = -1, .sps_id = -1, .pps_id = -1,
								.frame_num = -1, .frame_seq = frame_seq, .poc_lsb = -1,
								.dpbIndex = smallest_dpb_index_not_in_queue(s) };
	uint32_t slice_offsets[MAX_SLICES]; //TODO check those
	uint32_t slice_offsets_count = 0; 

	// ---Copying NAL units into s->decodeBuffer---
	VkDeviceSize nalu_buffer_written = 0;
	VkDeviceSize nalu_buffer_len = s->decodeBufferSize;
	unsigned char *nalu_buffer = (unsigned char*)begin_nalu_writing(s);
	if (nalu_buffer == NULL)
	{
		printf("[vulkan_decode] Failed to map needed vulkan memory for NAL units!\n");
		return res;
	}
	{
		const unsigned char *nal = get_next_nal(src, src_len, true), *next_nal = NULL;
		size_t nal_len = 0;
		if (nal == NULL) puts("First NAL is NULL");

		while (nal != NULL)
		{
			const unsigned char *nal_payload = skip_nal_start_code(nal);
			if (nal_payload == NULL)
			{
				printf("[vulkan_decode] NAL unit does not begin with a start code.\n");
				//TODO err flag
				break;
			}

			next_nal = get_next_nal(nal_payload, src_len - (nal_payload - src), true);
			nal_len = next_nal == NULL ? src_len - (nal - src) : next_nal - nal;
			if (nal_len <= 4 || (size_t)(nal_payload - nal) >= nal_len) //TODO weird?
			{
				printf("[vulkan_decode] NAL unit is too short.\n");
				//TODO err flag
				break;
			}
			
			size_t nal_payload_len = nal_len - (nal_payload - nal); //should be non-zero now

			int nalu_type = NALU_HDR_GET_TYPE(nal_payload[0], !isH264);
			int nalu_idc = H264_NALU_HDR_GET_NRI(nal_payload[0]);

			/*printf("\tNALU - input idx: %u, header %u: (", nal - src, nal_payload[0]);
			print_nalu_header(nal_payload[0]);
			printf(") type name - ");
			print_nalu_name(nalu_type);
			putchar('\n');*/

			switch(nalu_type)
			{
				case NAL_H264_SEI:
					//puts("\t\tSEI => Skipping.");
					break; //switch break
				case NAL_H264_IDR:
					{
						//puts("\t\tIDR => Decode.");
						//TODO maybe do those only after successful writing?
						s->prev_poc_lsb = 0;
						s->prev_poc_msb = 0;
						s->idr_frame_seq = frame_seq;
						clear_the_ref_slot_queue(s); // we dont need those references anymore

						bool sh_ret = handle_sh_nalu(s, nal_payload, nal_payload_len, nalu_type, nalu_idc, &slice_info);
						UNUSED(sh_ret); //TODO maybe error?

						VkDeviceSize written = write_nalu(nalu_buffer + nalu_buffer_written,
														  nalu_buffer_len - nalu_buffer_written,
														  nal, nal_len);
						if (written > 0 && slice_offsets_count < MAX_SLICES)
						{
							//printf("\t\tWriting success.\n");
							slice_offsets[slice_offsets_count++] = nalu_buffer_written;
							nalu_buffer_written += written;

							slice_info.is_reference = nalu_idc > 0;
						}
						//else printf("\t\tWriting fail.\n");
					}
					break; //switch break
				case NAL_H264_NONIDR:
					{
						//puts("\t\tNon-IDR => Decode.");
						bool sh_ret = handle_sh_nalu(s, nal_payload, nal_payload_len, nalu_type, nalu_idc, &slice_info);
						UNUSED(sh_ret); //TODO maybe error?

						VkDeviceSize written = write_nalu(nalu_buffer + nalu_buffer_written,
														nalu_buffer_len - nalu_buffer_written,
														nal, nal_len);
						if (written > 0 && slice_offsets_count < MAX_SLICES)
						{
							//printf("\t\tWriting success.\n");
							slice_offsets[slice_offsets_count++] = nalu_buffer_written;
							nalu_buffer_written += written;

							slice_info.is_reference = nalu_idc > 0;
						}
						//else printf("\t\tWriting fail.\n");
					}
					break; //switch break
				case NAL_H264_SPS:
					{
						//puts("\t\tSPS => Load SPS.");
						bool sps_ret = handle_sps_nalu(s, nal_payload, nal_payload_len, isH264);
						UNUSED(sps_ret); //TODO maybe error?
					}
					break; //switch break
				case NAL_H264_PPS:
					{
						//puts("\t\tPPS => Load PPS.");
						bool pps_ret = handle_pps_nalu(s, nal_payload, nal_payload_len, isH264);
						UNUSED(pps_ret); //TODO maybe error?
					}
					
					break; //switch break

				//TODO H.265 NAL unit types
				default:
					if (isH264) puts("\t\tOther => Skipping.");
					else puts("\t\tH265 is not implemented yet.");
					break; //switch break
			}

			nal = next_nal;
		}
	}
	end_nalu_writing(s); //TODO check for err flag here
	//printf("\tEnd of NALU writing. NAL buffer size: %u, bytes written: %u\n",
	//		nalu_buffer_len, nalu_buffer_written);
 	assert(nalu_buffer_written <= nalu_buffer_len);

	assert(s->cmdBuffer != VK_NULL_HANDLE);
	assert(s->videoSession != VK_NULL_HANDLE);
	assert(s->videoSessionParams != VK_NULL_HANDLE);

	//TODO maybe check for valid values of frame_num, poc_lsb, pps_id
	//printf("\tframe_seq: %d, frame_num: %d, poc_lsb: %d, pps_id: %d, is_reference: %d, is_intra: %d\n",
	//		slice_info.frame_seq, slice_info.frame_num, slice_info.poc_lsb, slice_info.pps_id,
	//		(int)(slice_info.is_reference), slice_info.is_intra);

	if (!begin_cmd_buffer(s))
	{
		// err msg should get printed inside of begin_cmd_buffer
		return res;
	}

	// ---VkImage layout transfering before video coding scope---
	if (!s->dpbHasDefinedLayout)	// if VkImages in DPB are in undefined layout we need to transfer them into decode layout
	{
		for (size_t i = 0; i < MAX_REF_FRAMES + 1; ++i)
		{
			assert(s->dpb[i] != VK_NULL_HANDLE);
			transfer_image_layout(s->cmdBuffer, s->dpb[i],
								  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
		}

		s->dpbHasDefinedLayout = true;
	}
	else							// otherwise we also transfer them, but from transfer src optimal layout
	{
		for (size_t i = 0; i < MAX_REF_FRAMES + 1; ++i)
		{
			assert(s->dpb[i] != VK_NULL_HANDLE);
			transfer_image_layout(s->cmdBuffer, s->dpb[i],
								  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
		}
	}

	// ---Video coding scope---
	begin_video_coding_scope(s, &slice_info);
	decode_frame(s, slice_info, nalu_buffer_written, slice_offsets, slice_offsets_count);
	end_video_coding_scope(s);
	
	// ---VkImage synchronization and layout transfering after vido coding scope---
	for (size_t i = 0; i < MAX_REF_FRAMES + 1; ++i)
	{
		transfer_image_layout(s->cmdBuffer, s->dpb[i], //VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR
							  VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	}
	
	// ---Copying decoded DPB image into decoded picture buffer---
	assert(s->dstPicBuffer != VK_NULL_HANDLE);
	assert(s->dpbFormat == VK_FORMAT_G8_B8R8_2PLANE_420_UNORM); //TODO requirement

	VkDeviceSize firstPlaneSize = s->width * s->height;
	VkDeviceSize secondPlaneSize = firstPlaneSize / 4;

	VkImageAspectFlagBits aspectFlags[2] = { VK_IMAGE_ASPECT_PLANE_0_BIT, VK_IMAGE_ASPECT_PLANE_1_BIT };
	VkBufferImageCopy dstPicRegions[2] = { { .bufferOffset = s->dstPicBufferMemoryOffset,
											 .imageOffset = { 0, 0, 0 }, // empty offset
											 // videoSize with depth == 1:
											 .imageExtent = { s->width, s->height, 1 },
											 .imageSubresource = { .aspectMask = aspectFlags[0],
											 					   .mipLevel = 0,
											 					   .baseArrayLayer = 0,
											 					   .layerCount = 1 } },
										   { .bufferOffset = s->dstPicBufferMemoryOffset + firstPlaneSize, //TODO check this
											 .imageOffset = { 0, 0, 0 }, // empty offset
											 // one half because VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:
											 .imageExtent = { s->width / 2, s->height / 2, 1 }, 
											 .imageSubresource = { .aspectMask = aspectFlags[1],
											 					   .mipLevel = 0,
											 					   .baseArrayLayer = 0,
											 					   .layerCount = 1 } } };
	//TODO this expects display order to be the same as decode order
	vkCmdCopyImageToBuffer(s->cmdBuffer, s->dpb[slice_info.dpbIndex], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
						   s->dstPicBuffer, 2, dstPicRegions);

	if (!end_cmd_buffer(s))
	{
		// relevant err msg printed inside of end_cmd_buffer
		return res;
	}

	// ---Submiting the decode commands into queue---
	VkSubmitInfo submitInfo = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
								.waitSemaphoreCount = 0,
								.commandBufferCount = 1,
								.pCommandBuffers = &s->cmdBuffer,
								.signalSemaphoreCount = 0 };
	VkResult result = vkQueueSubmit(s->decodeQueue, 1, &submitInfo, s->fence);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to submit the decode cmd buffer into queue!\n");
		return res;
	}

	// ---Reference queue management---
	// can be done before synchronization as we only work with slice_infos
	if (slice_info.is_reference) // however insert only if the decoded frame actually is a reference
	{
		// new value for s->dpbDstPictureIdx gets set in insert_ref_slot_into_queue!
		insert_ref_slot_into_queue(s, slice_info);
	}

	// ---Synchronization---
	const uint64_t synchronizationTimeout = 500 * 1000 * 1000; // = 500ms (timeout is in nanoseconds)

	result = vkWaitForFences(s->device, 1, &s->fence, VK_TRUE, synchronizationTimeout);
	if (result != VK_SUCCESS)
	{
		if (result == VK_TIMEOUT) printf("[vulkan_decode] Vulkan can't synchronize! -> Timeout reached.\n");
		else if (result == VK_ERROR_DEVICE_LOST) printf("[vulkan_decode] Vulkan can't synchronize! -> Device lost.\n");
		else printf("[vulkan_decode] Vulkan can't synchronize! -> Not enough memory.\n");
		
		return false;
	}

	result = vkResetFences(s->device, 1, &s->fence);
	if (result != VK_SUCCESS)
	{
		//should happen only when out of memory
		printf("[vulkan_decode] Failed to reset vulkan fence!\n");
		return false;
	}

	// ---Writing the decoded frame data---
	if (!write_decoded_frame(s, firstPlaneSize + 2 * secondPlaneSize, dst))
	{
		printf("[vulkan_decode] Failed to write the decoded frame into the destination buffer!\n");
		return res;
	}
	else res = DECODER_GOT_FRAME;

	//puts("Got frame!");
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
