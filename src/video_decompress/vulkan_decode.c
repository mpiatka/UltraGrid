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
#include "rtp/rtpdec_h264.h"	//TODO
#include "rtp/rtpenc_h264.h"	//TODO

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

static PFN_vkCreateInstance vkCreateInstance = NULL;
static PFN_vkDestroyInstance vkDestroyInstance = NULL;
#ifdef VULKAN_VALIDATE
static PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT = NULL;
static PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT = NULL;
#endif
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

	return vkDestroyInstance &&
	#ifdef VULKAN_VALIDATE
		   vkCreateDebugUtilsMessengerEXT && vkDestroyDebugUtilsMessengerEXT &&
	#endif
		   vkEnumeratePhysicalDevices && vkGetPhysicalDeviceProperties &&
		   vkGetPhysicalDeviceProperties2KHR &&
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
		   vkCreateImageView && vkDestroyImageView;
}

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
	bool prepared, sps_vps_found;
	VkFence fence;
	VkBuffer decodeBuffer;						// needs to be destroyed if valid
	VkDeviceMemory decodeBufferMemory;			// needs to be freed if valid
	VkDeviceSize decodeBufferSize;
	VkDeviceSize decodeBufferOffsetAlignment;
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
	VkImage dpb[MAX_REF_FRAMES + 1];			// decoded picture buffer
	VkImageView dpbViews[MAX_REF_FRAMES + 1];	// dpb image views
	VkDeviceMemory dpbMemory;					// backing memory for dpb - needs to be freed if valid (destroyed in destroy_dpb)
	VkFormat dpbFormat;							// format of VkImages in dpb
	uint32_t dpbDstPictureIdx;					// index (into dpb) of the slot for next to be decoded frame
	uint32_t referenceSlotsQueue[MAX_REF_FRAMES]; // queue containing indices (into dpb) of current reference frames 
	uint32_t referenceSlotsQueue_start;			  // index into referenceSlotsQueue where the queue starts
	uint32_t referenceSlotsQueue_count;			  // the current length of the reference slots queue

	// UltraGrid data
	int width, height;
	int pitch; //USELESS ?
	codec_t out_codec;
};

static bool end_video_decoding(struct state_vulkan_decompress *s);
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

static VkPhysicalDevice choose_physical_device(struct state_vulkan_decompress *s,
										VkPhysicalDevice devices[], uint32_t devices_count,
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

	//TODO move those to global
	/*PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties =
					(PFN_vkGetPhysicalDeviceProperties)s->loader(s->instance, "vkGetPhysicalDeviceProperties");
	assert(vkGetPhysicalDeviceProperties != NULL);*/
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

		if (!check_for_device_extensions(devices[i], requiredDeviceExtensions))
		{
			printf("\tDevice does not have required extensions.\n");
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
								  .apiVersion = VK_API_VERSION_1_0 };
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
	s->physicalDevice = choose_physical_device(s, devices, phys_devices_count, requestedFamilyQueueFlags,
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
	//VkPhysicalDeviceFeatures deviceFeatures = { 0 };
	VkDeviceCreateInfo createDeviceInfo = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
											.pNext = NULL,
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
	s->decodeBuffer = VK_NULL_HANDLE; //buffer gets created in prepare function
	s->decodeBufferMemory = VK_NULL_HANDLE;
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

	printf("[vulkan_decode] Initialization finished successfully.\n");
	return s;
}

static void free_decode_buffer(struct state_vulkan_decompress *s)
{
	//buffer needs to get destroyed first
	if (vkDestroyBuffer != NULL && s->device != VK_NULL_HANDLE)
				vkDestroyBuffer(s->device, s->decodeBuffer, NULL);
	s->decodeBuffer = VK_NULL_HANDLE;
	s->decodeBufferSize = 0;
	s->decodeBufferOffsetAlignment = 0;

	if (vkFreeMemory != NULL && s->device != VK_NULL_HANDLE)
				vkFreeMemory(s->device, s->decodeBufferMemory, NULL);
	s->decodeBufferMemory = VK_NULL_HANDLE;
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

	free_decode_buffer(s);

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

	s->codecOperation = videoCodecOperation;
	s->width = desc.width;
	s->height = desc.height;

	s->dpbFormat = VK_FORMAT_UNDEFINED;
	s->dpbDstPictureIdx = 0;
	s->referenceSlotsQueue_start = 0;
	s->referenceSlotsQueue_count = 0;

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
	printf("\tcodec color_spec: '%s', out_codec: '%s'\n", spec_name, out_name);
	if (out_codec == VIDEO_CODEC_NONE) printf("\tRequested probing.\n");

	if (desc.tile_count != 1)
	{
		//TODO they could be supported
		printf("[vulkan_decode] Tiled video formats are not supported!\n");
		return false;
	}

	s->prepared = false; //TODO - freeing resources probably needed when s->prepared == true
	s->sps_vps_found = false;
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

static bool check_for_vulkan_format(VkPhysicalDevice physDevice, VkPhysicalDeviceVideoFormatInfoKHR videoFormatInfo,
									codec_t codec, VkVideoFormatPropertiesKHR *formatProperties)
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
	VkFormat wanted = codec_to_vulkan_format(codec);

	for (uint32_t i = 0; i < properties_count; ++i)
	{
		VkFormat format = properties[i].format;
		printf("\tformat: %d image_type: %d\n", format, properties[i].imageType);
		if (format == wanted)
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

static bool create_dpb(struct state_vulkan_decompress *s, VkVideoProfileListInfoKHR *videoProfileList)
{
	assert(s->device != VK_NULL_HANDLE && s->dpbFormat != VK_FORMAT_UNDEFINED);

	const VkImageType imageType = VK_IMAGE_TYPE_2D;
	const VkExtent3D videoSize = { s->width, s->height, 1 }; //depth must be 1 for VK_IMAGE_TYPE_2D

	//imageCreateMaxMipLevels, imageCreateMaxArrayLayers, imageCreateMaxExtent, and imageCreateSampleCounts
	VkImageCreateInfo imgInfo = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
								  .pNext = (void*)videoProfileList,
								  .flags = 0,
								  .usage = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR |
								  		   VK_IMAGE_USAGE_VIDEO_DECODE_SRC_BIT_KHR |
										   VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR,
								  .imageType = imageType,
								  .mipLevels = 1,
								  .samples = VK_SAMPLE_COUNT_1_BIT,
								  .format = s->dpbFormat,
								  .extent = videoSize,
								  .arrayLayers = 1,
								  .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
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
		videoFormatInfo.imageUsage = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR |
								  	 VK_IMAGE_USAGE_VIDEO_DECODE_SRC_BIT_KHR |
									 VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR;
		// err msg should be printed inside of check_for_vulkan_format function 
		if (!check_for_vulkan_format(s->physicalDevice, videoFormatInfo, s->out_codec, &referencePictureFormatProperties))
				return false;

		pictureFormatProperites = referencePictureFormatProperties;
		referencePictureFormat = referencePictureFormatProperties.format;
		pictureFormat = pictureFormatProperites.format;
	}
	/*else if (decodeCapabilitiesFlags & VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_DISTINCT_BIT_KHR)
	{
		
		videoFormatInfo.imageUsage = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR;
		// err msg should be printed inside of check_for_vulkan_format function 
		if (!check_for_vulkan_format(s->physicalDevice, videoFormatInfo, s->out_codec, &referencePictureFormatProperties))
				return false;
		videoFormatInfo.imageUsage = VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR;
		if (!check_for_vulkan_format(s->physicalDevice, videoFormatInfo, s->out_codec, &pictureFormatProperites))
				return false;

		referencePictureFormat = referencePictureFormatProperties.format;
		pictureFormat = pictureFormatProperites.format;
	}*/
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

	// ---Creating decodeBuffer for NAL units---
	assert(s->decodeBuffer == VK_NULL_HANDLE);

	const VkDeviceSize wantedBufferSize = 1024 * 1024; //TODO magic number, check if smaller than allowed amount
	s->decodeBufferOffsetAlignment = videoCapabilities.minBitstreamBufferSizeAlignment;
	s->decodeBufferSize = (wantedBufferSize + (s->decodeBufferOffsetAlignment - 1))
						  & ~(s->decodeBufferOffsetAlignment - 1); //alignment bit mask magic
	VkBufferCreateInfo bufferInfo = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
									  .pNext = (void*)&videoProfileList,
									  .flags = 0,
									  .usage = VK_BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR, //add VK_BUFFER_USAGE_VERTEX_BUFFER_BIT?
									  .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
									  .size = s->decodeBufferSize,
									  .queueFamilyIndexCount = 1,
									  .pQueueFamilyIndices = &s->queueFamilyIdx };
	result = vkCreateBuffer(s->device, &bufferInfo, NULL, &s->decodeBuffer);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to create vulkan buffer for decoding!\n");
		if (vkDestroyFence != NULL)
		{
			vkDestroyFence(s->device, s->fence, NULL);
			s->fence = VK_NULL_HANDLE;
		}

		return false;
	}
	s->decodeBufferOffsetAlignment = videoCapabilities.minBitstreamBufferOffsetAlignment;
	
	assert(s->decodeBuffer != VK_NULL_HANDLE);

	VkMemoryRequirements bufferMemoryRequirements;
	vkGetBufferMemoryRequirements(s->device, s->decodeBuffer, &bufferMemoryRequirements);

	uint32_t bufferMemoryType_idx = 0;
	if (!find_memory_type(s, bufferMemoryRequirements.memoryTypeBits,
						  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
						  &bufferMemoryType_idx))
	{
		printf("[vulkan_decode] Failed to find required memory type for vulkan buffer!\n");
		free_decode_buffer(s);
		if (vkDestroyFence != NULL)
		{
			vkDestroyFence(s->device, s->fence, NULL);
			s->fence = VK_NULL_HANDLE;
		}

		return false;
	}

	assert(s->decodeBufferMemory == VK_NULL_HANDLE);

	VkMemoryAllocateInfo bufferAllocInfo = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
											 .allocationSize = bufferMemoryRequirements.size,
											 .memoryTypeIndex = bufferMemoryType_idx };
	result = vkAllocateMemory(s->device, &bufferAllocInfo, NULL, &s->decodeBufferMemory);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to allocate memory for vulkan buffer!\n");
		free_decode_buffer(s);
		if (vkDestroyFence != NULL)
		{
			vkDestroyFence(s->device, s->fence, NULL);
			s->fence = VK_NULL_HANDLE;
		}

		return false;
	}

	assert(s->decodeBufferMemory != VK_NULL_HANDLE);

	result = vkBindBufferMemory(s->device, s->decodeBuffer, s->decodeBufferMemory, 0);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to bind vulkan memory to vulkan buffer!\n");
		free_decode_buffer(s);
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
		free_decode_buffer(s);
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
		free_decode_buffer(s);
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
		free_decode_buffer(s);
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
												.maxDpbSlots = videoCapabilities.maxDpbSlots < MAX_REF_FRAMES ?
															   videoCapabilities.maxDpbSlots : MAX_REF_FRAMES,
												.maxActiveReferencePictures = videoCapabilities.maxActiveReferencePictures,
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
		free_decode_buffer(s);
		if (vkDestroyFence != NULL)
		{
			vkDestroyFence(s->device, s->fence, NULL);
			s->fence = VK_NULL_HANDLE;
		}

		return false;
	}

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
		free_decode_buffer(s);
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
		free_decode_buffer(s);
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

static bool start_video_decoding(struct state_vulkan_decompress *s)
{
	// ---Command buffer into recording state---
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

	// ---Begining of video coding---
	bool isH264 = s->codecOperation == VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;

	/*StdVideoDecodeH264ReferenceInfo h264ReferenceInfo = { .flags = ,
														  .FrameNum = ,
														  .PicOrderCnt = {} };
	//TODO
	VkVideoDecodeH264DpbSlotInfoKHR h264SlotInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,
													 .pStdReferenceInfo = &h264ReferenceInfo };
	VkVideoDecodeH265DpbSlotInfoKHR h265SlotInfo = {};
	VkVideoReferenceSlotInfoKHR slotInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
											 .pNext = isH264 ? (void*)&h264SlotInfo : (void*)&h265SlotInfo
											 .slotIndex = ,
											 .pPictureResource = };*/
	
	VkVideoBeginCodingInfoKHR beginCodingInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR,
												  .flags = 0,
												  .videoSession = s->videoSession,
												  .videoSessionParameters = s->videoSessionParams,
												  .referenceSlotCount = 0,
												  .pReferenceSlots = NULL };
	vkCmdBeginVideoCodingKHR(s->cmdBuffer, &beginCodingInfo);

	VkVideoCodingControlInfoKHR vidCodingControlInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR,
														 .flags = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR };
	vkCmdControlVideoCodingKHR(s->cmdBuffer, &vidCodingControlInfo);

	return true;
}

static bool end_video_decoding(struct state_vulkan_decompress *s)
{
	assert(s->cmdBuffer != VK_NULL_HANDLE);

	VkVideoEndCodingInfoKHR endCodingInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR };
	vkCmdEndVideoCodingKHR(s->cmdBuffer, &endCodingInfo);

	VkResult result = vkEndCommandBuffer(s->cmdBuffer);
	//s->videoCodingStarted = false; //we end video decoding no matter the result
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

	VkResult result = vkMapMemory(s->device, s->decodeBufferMemory, 0, VK_WHOLE_SIZE, 0, &memoryPtr);
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

	vkUnmapMemory(s->device, s->decodeBufferMemory);
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
	print_sps(&sps);

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

static bool handle_sh_nalu(struct state_vulkan_decompress *s, const unsigned char *nal_payload, size_t len,
						   int nal_type, int nal_idc)
{
	// reads slice header from given buffer range
	
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

	destroy_rbsp(rbsp);
	return true;
}

static uint32_t get_ref_slot_from_queue(struct state_vulkan_decompress *s, uint32_t index)
{
	uint32_t wrapped = (s->referenceSlotsQueue_start) + index % MAX_REF_FRAMES;
	return s->referenceSlotsQueue[wrapped];
}

/*static VkVideoPictureResourceInfoKHR get_dst_picture_info(struct state_vulkan_decompress *s)
{
	VkExtent2D videoSize = { s->width, s->height };

	VkVideoPictureResourceInfoKHR info = { .sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
										   .codedOffset = { 0, 0 }, // no offset
										   .codedExtent = videoSize,
										   .baseArrayLayer = 0,
										   .imageViewBinding = s->dpbView[s->dpbDstPictureIdx] };
	return info;
}

static bool fill_ref_picture_infos(VkVideoReferenceSlotInfoKHR refInfos[], VkVideoPictureResourceInfoKHR picInfos[],
								   uint32_t count, bool isH264)
{
	// count is a size of both given arrays (should be at most same as MAX_REF_FRAMES)
	assert(count <= MAX_REF_FRAMES);

	VkExtent2D videoSize = { s->width, s->height };

	for (uint32_t i = 0; i < count; ++i)
	{
		uint32_t slotIndex = get_ref_slot_from_queue(i);
		assert(slotIndex < MAX_REF_FRAMES + 1);
		VkImageView view = s->dpbView[slotIndex];
		assert(view != VK_NULL_HANDLE);

		//TODO infos mustnt be local variables!
		StdVideoDecodeH264ReferenceInfo h264StdRefInfo = { .flags = ,
														   .FrameNum = };
		VkVideoDecodeH264DpbSlotInfoKHR h264RefInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,
														.pStdReferenceInfo = &h264StdRefInfo }; //TODO
		VkVideoDecodeH265DpbSlotInfoKHR h265RefInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR }; //TODO
		picInfos[i] = (VkVideoPictureResourceInfoKHR){ .sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
													   .codedOffset = { 0, 0 }, // no offset
													   .codedExtent = videoSize,
													   .baseArrayLayer = 0,
													   .imageViewBinding = view };
		refInfos[i] = (VkVideoReferenceSlotInfoKHR){ .sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
													 .pNext = isH264 ? (void*)&h264RefInfo : (void*)&h265RefInfo,
													 .slotIndex = slotIndex, //TODO is it the same slot index?
													 .pPictureResource = picInfos + i };
	}

	return true;
}*/

static decompress_status vulkan_decompress(void *state, unsigned char *dst, unsigned char *src,
                unsigned int src_len, int frame_seq, struct video_frame_callbacks *callbacks,
                struct pixfmt_desc *internal_prop)
{
	printf("vulkan_decode - decompress\n");
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
	
	UNUSED(dst);
	UNUSED(frame_seq);

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

	bool isH264 = s->codecOperation == VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
	
	// ---Copying NAL units into s->decodeBuffer---
	VkDeviceSize nalu_buffer_written = 0;
	VkDeviceSize nalu_buffer_len = s->decodeBufferSize;
	unsigned char *nalu_buffer = (unsigned char*)begin_nalu_writing(s);
	if (nalu_buffer == NULL)
	{
		printf("[vulkan_decode] Failed to map needed vulkan memory for NAL units!\n");
		end_video_decoding(s); //typically err in end_video_decoding happens only when bad video parameters 
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
					puts("\t\tSEI => Skipping.");
					break; //switch break
				case NAL_H264_IDR:
					{
						puts("\t\tIDR => Decode.");
						bool sh_ret = handle_sh_nalu(s, nal_payload, nal_payload_len, nalu_type, nalu_idc);
						UNUSED(sh_ret); //TODO maybe error?

						VkDeviceSize written = write_nalu(nalu_buffer + nalu_buffer_written,
														  nalu_buffer_len - nalu_buffer_written,
														  nal, nal_len);
						nalu_buffer_written += written;
						if (written > 0) printf("\t\tWriting success.\n");
						else printf("\t\tWriting fail.\n");
					}
					break; //switch break
				case NAL_H264_NONIDR:
					{
						puts("\t\tNon-IDR => Decode.");
						bool sh_ret = handle_sh_nalu(s, nal_payload, nal_payload_len, nalu_type, nalu_idc);
						UNUSED(sh_ret); //TODO maybe error?

						VkDeviceSize written = write_nalu(nalu_buffer + nalu_buffer_written,
														nalu_buffer_len - nalu_buffer_written,
														nal, nal_len);
						nalu_buffer_written += written;
						if (written > 0) printf("\t\tWriting success.\n");
						else printf("\t\tWriting fail.\n");
					}
					break; //switch break
				case NAL_H264_SPS:
					{
						puts("\t\tSPS => Load SPS.");
						bool sps_ret = handle_sps_nalu(s, nal_payload, nal_payload_len, isH264);
						UNUSED(sps_ret); //TODO maybe error?
					}
					break; //switch break
				case NAL_H264_PPS:
					{
						puts("\t\tPPS => Load PPS.");
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

	if (!start_video_decoding(s))
	{
		printf("[vulkan_decode] Failed to start video decoding!\n");
		return res;
	}

	//TODO decompress
	// check - rtp/rtpenc_h264.h, utils/h264_stream.h
	
	/*VkVideoPictureResourceInfoKHR dstPictureInfo = get_dst_picture_info(s);
	VkVideoPictureResourceInfoKHR referencePictureInfos[MAX_REF_FRAMES] = { 0 };
	VkVideoReferenceSlotInfoKHR referenceSlotInfos[MAX_REF_FRAMES] = { 0 };
	uint32_t referencePicture_count = s->referenceSlotsQueue_count;
	
	assert(referencePicture_count <= MAX_REF_FRAMES);
	if (!fill_ref_picture_infos(referenceSlotInfos, referencePictureInfos, referencePicture_count, isH264))
	{
		//TODO err
		return res;
	}

	VkVideoDecodeH264PictureInfoKHR h264DecodeInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PICTURE_INFO_KHR };
	VkVideoDecodeH265PictureInfoKHR h265DecodeInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PICTURE_INFO_KHR };
	VkVideoDecodeInfoKHR decodeInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR,
										.pNext = isH264 ? (void*)&h264DecodeInfo : (void*)&h265DecodeInfo,
										.flags = 0,
										.srcBuffer = s->decodeBuffer,
										.srcBufferOffset = 0,
										.srcBufferRange = nalu_buffer_written,
										.dstPictureResource = dstPictureInfo,
										//specifies the needed used references (but not the decoded frame)
										.referenceSlotCount = referencePicture_count,
										.pReferenceSlots = referenceSlotInfos,
										.pSetupReferenceSlot = NULL };
	vkCmdDecodeVideoKHR(s->cmdBuffer, &decodeInfo);*/

	if (!end_video_decoding(s))
	{
		//relevant error printed inside of end_video_decoding
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

	// ---Synchronization---
	const uint64_t synchronizationTimeout = 5 * 1000 * 1000; // = 5ms (timeout is in nanoseconds)

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
