/**
 * @file   video_decompress/vulkan_decode.c
 * @author Ondřej Richtr     <524885@mail.muni.cz>
 */
 
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

// copied from vulkan_sdl2.cpp
#ifdef __MINGW32__
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_decompress.h"

#include "utils/bs.h"
#include "rtp/rtpdec_h264.h"

#ifdef VK_USE_PLATFORM_WIN32_KHR
//#include <windows.h>	//LoadLibrary
#else
#include <dlfcn.h>		//dlopen, dlsym, dlclose
#endif

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif

#include <vulkan/vulkan.h>
#include <vk_video/vulkan_video_codec_h264std.h>
#include <vk_video/vulkan_video_codec_h264std_decode.h>

// helper functions for parsing H.264:
#include "vulkan_decode_h264.h"

// activates vulkan validation layers if defined
// if defined your vulkan loader needs to know where to find the validation layer manifest
// (for example through VK_LAYER_PATH or VK_ADD_LAYER_PATH env. variables)
//#define VULKAN_VALIDATE

// one of value from enum VkDebugUtilsMessageSeverityFlagBitsEXT included from vulkan.h
// (definition in vulkan_core.h)
//#define VULKAN_VALIDATE_SHOW_SEVERITY VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
#define VULKAN_VALIDATE_SHOW_SEVERITY VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
//#define VULKAN_VALIDATE_SHOW_SEVERITY VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT
//#define VULKAN_VALIDATE_SHOW_SEVERITY VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT

// activates vukan queries if defined
// query prints result value for each decode command (one per decoded frame)
// typical query result values are: -1 => error, 1 => success, 1000331003 => ???
//#define VULKAN_QUERIES

#define MAX_REF_FRAMES 16
#define MAX_SLICES 128
#define MAX_OUTPUT_FRAMES_QUEUE_SIZE 32
#define BITSTREAM_STARTCODE 0, 0, 1

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
static PFN_vkCmdCopyImage vkCmdCopyImage = NULL;
static PFN_vkGetPhysicalDeviceFormatProperties2 vkGetPhysicalDeviceFormatProperties2 = NULL;
static PFN_vkCreateQueryPool vkCreateQueryPool = NULL;
static PFN_vkDestroyQueryPool vkDestroyQueryPool = NULL;
static PFN_vkCmdResetQueryPool vkCmdResetQueryPool = NULL;
static PFN_vkGetQueryPoolResults vkGetQueryPoolResults = NULL;

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
	vkGetDeviceQueue = (PFN_vkGetDeviceQueue)loader(instance, "vkGetDeviceQueue");
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
	vkCmdCopyImage = (PFN_vkCmdCopyImage)loader(instance, "vkCmdCopyImage");
	vkCmdPipelineBarrier2KHR = (PFN_vkCmdPipelineBarrier2)loader(instance, "vkCmdPipelineBarrier2KHR");
	vkGetPhysicalDeviceFormatProperties2 = (PFN_vkGetPhysicalDeviceFormatProperties2)
												loader(instance, "vkGetPhysicalDeviceFormatProperties2");
	vkCreateQueryPool = (PFN_vkCreateQueryPool)loader(instance, "vkCreateQueryPool");
	vkDestroyQueryPool = (PFN_vkDestroyQueryPool)loader(instance, "vkDestroyQueryPool");
	vkCmdResetQueryPool = (PFN_vkCmdResetQueryPool)loader(instance, "vkCmdResetQueryPool");
	vkGetQueryPoolResults = (PFN_vkGetQueryPoolResults)loader(instance, "vkGetQueryPoolResults");

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
		   vkCmdCopyImageToBuffer && vkCmdCopyImage &&
		   vkCmdPipelineBarrier2KHR && vkGetPhysicalDeviceFormatProperties2 &&
		   vkCreateQueryPool && vkDestroyQueryPool &&
		   vkCmdResetQueryPool && vkGetQueryPoolResults;
}

typedef struct	// structure used to pass around variables related to currently decoded frame (slice)
{
	bool is_intra, is_reference;
	int nal_idc;
	int idr_pic_id;
	int sps_id;
	int pps_id;
	int frame_num;
	int frame_seq; // Ultragrid's frame_seq parameter in decompress function
	int poc, poc_lsb;
	uint32_t dpbIndex;

} slice_info_t;

typedef struct
{
	size_t data_idx; // index (data_idx * s->outputFrameQueue_data_size) into s->outputFrameQueue_data
	size_t data_len; // now is the same as s->outputFrameQueue_data_size

	int idr_frame_seq; // frame_seq value of last encountered IDR frame in decode order (identifies a GOP this frame belongs to)
	int frame_seq;
	int poc;
	int poc_wrap;
} frame_data_t;

struct state_vulkan_decompress
{
#ifdef VK_USE_PLATFORM_WIN32_KHR
	HMODULE vulkanLib;							// needs to be destroyed if valid
#else
	void *vulkanLib;
#endif
	VkInstance instance; 						// needs to be destroyed if valid
	PFN_vkGetInstanceProcAddr loader;
	//maybe this could be present only when VULKAN_VALIDATE is defined?
	VkDebugUtilsMessengerEXT debugMessenger;	// needs to be destroyed if valid
	VkPhysicalDevice physicalDevice;
	VkDevice device;							// needs to be destroyed if valid
	uint32_t queueFamilyIdx;
	VkQueue decodeQueue;
	VkVideoCodecOperationFlagsKHR queueVideoFlags;
	VkVideoCodecOperationFlagsKHR codecOperation;
	bool prepared, sps_vps_found, resetVideoCoding;
	VkFence fence;
	VkBuffer bitstreamBuffer;					// needs to be destroyed if valid
	VkDeviceSize bitstreamBufferSize;
	VkDeviceSize bitstreambufferSizeAlignment;
	VkDeviceMemory bitstreamBufferMemory;		// allocated memory for bitstreamBuffer, needs to be freed if valid
	VkCommandPool commandPool;					// needs to be destroyed if valid
	VkCommandBuffer cmdBuffer;
	VkVideoSessionKHR videoSession;				// needs to be destroyed if valid
	uint32_t videoSessionMemory_count;
	// array of size videoSessionMemory_count, needs to be freed and VkvideoSessionMemory deallocated
	VkDeviceMemory *videoSessionMemory;

	// Parameters of the incoming video:
	int depth_chroma, depth_luma;
	int subsampling; // in the Ultragrid format
	StdVideoH264ProfileIdc profileIdc; //TODO H.265
	VkVideoSessionParametersKHR videoSessionParams; // needs to be destroyed if valid

	// Pointers to arrays of sps (length MAX_SPS_IDS), pps (length MAX_PPS_IDS)
	// could be static arrays but that would take too much memory of this struct
	sps_t *sps_array;
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

	int prev_poc_lsb, prev_poc_msb, idr_frame_seq, current_frame_seq, poc_wrap, last_poc;
	int last_displayed_frame_seq, last_displayed_poc;

	// Output frame data queue
	VkDeviceSize outputFrameQueue_data_size;
	frame_data_t outputFrameQueue[MAX_OUTPUT_FRAMES_QUEUE_SIZE];
	size_t outputFrameQueue_capacity, outputFrameQueue_start, outputFrameQueue_count;
	uint8_t *outputFrameQueue_data;

	// Output video description data
	int width, height;
	VkDeviceSize lumaSize, chromaSize;
	int pitch; // currently not used (maybe not needed for multiplanar output formats)
	codec_t out_codec;

	// Output frame data
	VkImage outputLumaPlane, outputChromaPlane;	// planes for output decoded image, must be destroyed if valid
	VkDeviceMemory outputImageMemory;
	VkDeviceSize outputChromaPlaneOffset;

	// vulkan queries
	VkQueryPool queryPool; // needs to be destroyed if valid, should be always VK_NULL_HANDLE when VULKAN_QUERIES is not defined
};

static void free_buffers(struct state_vulkan_decompress *s);
static void destroy_output_image(struct state_vulkan_decompress *s);
static void destroy_dpb(struct state_vulkan_decompress *s);
static void destroy_output_queue(struct state_vulkan_decompress *s);
static void destroy_queries(struct state_vulkan_decompress *s);

static bool load_vulkan(struct state_vulkan_decompress *s)
{
	#ifdef VK_USE_PLATFORM_WIN32_KHR // windows
	const char vulkan_lib_filename[] = "vulkan-1.dll";
	const char vulkan_proc_name[] = "vkGetInstanceProcAddr";

    s->vulkanLib = LoadLibrary(vulkan_lib_filename);
	if (s->vulkanLib == NULL)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Vulkan loader file '%s' not found!\n", vulkan_lib_filename);
		return false;
	}

	s->loader = (PFN_vkGetInstanceProcAddr)GetProcAddress(s->vulkanLib, vulkan_proc_name);
    if (s->loader == NULL) {

		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Vulkan function '%s' not found!\n", vulkan_proc_name);
        return false;
    }
	#else // non windows
	const char vulkan_lib_filename[] = "libvulkan.so.1";
	const char vulkan_proc_name[] = "vkGetInstanceProcAddr";

	s->vulkanLib = dlopen(vulkan_lib_filename, RTLD_LAZY);
	if (s->vulkanLib == NULL)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Vulkan loader file '%s' not found!\n", vulkan_lib_filename);
		return false;
	}
	
	s->loader = (PFN_vkGetInstanceProcAddr)dlsym(s->vulkanLib, vulkan_proc_name);
    if (s->loader == NULL) {

		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Vulkan function '%s' not found!\n", vulkan_proc_name);
        return false;
    }
	#endif

	return true;
}

static void unload_vulkan(struct state_vulkan_decompress *s)
{
	#ifdef VK_USE_PLATFORM_WIN32_KHR
	FreeLibrary(s->vulkanLib);
	#else
	dlclose(s->vulkanLib);
	#endif
}

static bool check_for_instance_extensions(const char * const requiredInstanceextensions[])
{
	uint32_t extensions_count = 0;
	vkEnumerateInstanceExtensionProperties(NULL, &extensions_count, NULL);
	if (extensions_count == 0)
	{
		if (requiredInstanceextensions[0] != NULL)
		{
			log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] No instance extensions supported.\n");
			return false;
		}
		
		log_msg(LOG_LEVEL_INFO, "[vulkan_decode] No instance extensions found and none required.\n");
		return true;
	}

	VkExtensionProperties *extensions = (VkExtensionProperties*)calloc(extensions_count, sizeof(VkExtensionProperties));
	if (extensions == NULL)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to allocate array for instance extensions!\n");
		return false;
	}

	vkEnumerateInstanceExtensionProperties(NULL, &extensions_count, extensions);

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
			log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Required instance extension: '%s' was not found!\n",
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
	if (extensions_count == 0)
	{
		if (requiredDeviceExtensions[0] != NULL)
		{
			log_msg(LOG_LEVEL_WARNING, "[vulkan_decode] No device extensions supported.\n");
			return false;
		}

		log_msg(LOG_LEVEL_INFO, "[vulkan_decode] No device extensions found and none required.\n");
		return true;
	}

	VkExtensionProperties *extensions = (VkExtensionProperties*)calloc(extensions_count, sizeof(VkExtensionProperties));
	if (extensions == NULL)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to allocate array for device extensions!\n");
		return false;
	}

	vkEnumerateDeviceExtensionProperties(physDevice, NULL, &extensions_count, extensions);

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
			log_msg(LOG_LEVEL_WARNING, "[vulkan_decode] Required device extension: '%s' was not found!\n",
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
		log_msg(LOG_LEVEL_WARNING, "VULKAN VALIDATION: '%s'\n", pCallbackData->pMessage);
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
											   bool requireQueries,
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

		if (!check_for_device_extensions(devices[i], requiredDeviceExtensions))
		{
			log_msg(LOG_LEVEL_NOTICE, "[vulkan_decode] Device '%s' does not have required extensions.\n",
									  deviceProperties.deviceName);
			continue;
		}

		VkPhysicalDeviceVideoMaintenance1FeaturesKHR videoFeatures1 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_MAINTENANCE_1_FEATURES_KHR };
		VkPhysicalDeviceVulkan13Features deviceFeatures13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
															  .pNext = (void*)&videoFeatures1 };
		VkPhysicalDeviceFeatures2 deviceFeatures = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
													 .pNext = (void*)&deviceFeatures13 };
		vkGetPhysicalDeviceFeatures2(devices[i], &deviceFeatures);

		if (!deviceFeatures13.synchronization2 || !deviceFeatures13.maintenance4 ||
			(requireQueries && !videoFeatures1.videoMaintenance1))
		{
			log_msg(LOG_LEVEL_NOTICE, "[vulkan_decode] Device '%s' does not have required features.\n",
									  deviceProperties.deviceName);
			continue;
		}

		uint32_t queues_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties2(devices[i], &queues_count, NULL);

		if (queues_count == 0)
		{
			log_msg(LOG_LEVEL_NOTICE, "[vulkan_decode] Device '%s' has no queue families.\n",
									  deviceProperties.deviceName);
			continue;
		}
		
		VkQueueFamilyProperties2 *properties = (VkQueueFamilyProperties2*)calloc(queues_count, sizeof(VkQueueFamilyProperties2));
		VkQueueFamilyVideoPropertiesKHR *video_properties = (VkQueueFamilyVideoPropertiesKHR*)calloc(queues_count, sizeof(VkQueueFamilyVideoPropertiesKHR));
		VkQueueFamilyQueryResultStatusPropertiesKHR *query_properties = (VkQueueFamilyQueryResultStatusPropertiesKHR*)
																		calloc(queues_count, sizeof(VkQueueFamilyQueryResultStatusPropertiesKHR));
		if (properties == NULL || video_properties == NULL || query_properties == NULL)
		{
			log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to allocate properties and/or video_properties arrays!\n");
			free(properties);
			free(video_properties);
			break;
		}

		for (uint32_t j = 0; j < queues_count; ++j)
		{
			query_properties[j] = (VkQueueFamilyQueryResultStatusPropertiesKHR){ .sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_QUERY_RESULT_STATUS_PROPERTIES_KHR };
			video_properties[j] = (VkQueueFamilyVideoPropertiesKHR){ .sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR,
																	 .pNext = (void*)(query_properties + j) };
			properties[j] = (VkQueueFamilyProperties2){ .sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2,
														.pNext = (void*)(video_properties + j) };
		}

		vkGetPhysicalDeviceQueueFamilyProperties2(devices[i], &queues_count, properties);

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

			if (requestedQueueFamilyFlags == (flags & requestedQueueFamilyFlags) && videoFlags &&
				(!requireQueries || query_properties[j].queryResultStatusSupport))
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
				queue_family->pNext = NULL; // prevent dangling pointer
			}
			if (queue_video_props)
			{
				*queue_video_props = video_properties[preferred_queue_family];
				queue_video_props->pNext = NULL; // prevent dangling pointer
			}
		}

		free(properties);
		free(video_properties);
		free(query_properties);
	}

	return chosen;
}

static void * vulkan_decompress_init(void)
{
	// ---Allocation of the vulkan_decompress state and sps/pps arrays---
	struct state_vulkan_decompress *s = calloc(1, sizeof(struct state_vulkan_decompress));
	if (s == NULL)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Couldn't allocate memory for state struct!\n");
		return NULL;
	}

	sps_t *sps_array = calloc(MAX_SPS_IDS, sizeof(sps_t));
	if (sps_array == NULL)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Couldn't allocate memory for SPS array (num of members: %u)!\n", MAX_SPS_IDS);
		free(s);
		return NULL;
	}

	pps_t *pps_array = calloc(MAX_PPS_IDS, sizeof(pps_t));
	if (pps_array == NULL)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Couldn't allocate memory for PPS array (num of members: %u)!\n", MAX_PPS_IDS);
		free(sps_array);
		free(s);
		return NULL;
	}

	s->sps_array = sps_array;
	s->pps_array = pps_array;

	// ---Dynamic loading of the vulkan loader library and loader function---
	if (!load_vulkan(s))
	{
		// err msg printed inside of load_vulkan
		free(pps_array);
		free(sps_array);
		free(s);
		return false;
	}

	// ---Loading function pointers where the instance is not needed---
	if (!load_vulkan_functions_globals(s->loader))
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to load all vulkan functions!\n");
		unload_vulkan(s);
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
	
	if (!check_for_validation_layers(validationLayers))
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Required vulkan validation layers not found!\n");
		unload_vulkan(s);
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
		unload_vulkan(s);
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
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to create vulkan instance! Error: %d\n", result);
        unload_vulkan(s);
		free(pps_array);
		free(sps_array);
		free(s);
		return NULL;
	}

	if (!load_vulkan_functions_with_instance(s->loader, s->instance))
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to load all instance related vulkan functions!\n");
		if (vkDestroyInstance != NULL) vkDestroyInstance(s->instance, NULL);
		unload_vulkan(s);
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
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to setup debug messenger!\n");
		vkDestroyInstance(s->instance, NULL);
        unload_vulkan(s);
		free(pps_array);
		free(sps_array);
		free(s);
		return NULL;
	}
	#endif

	// ---Choosing of physical device---
	const VkQueueFlags requestedFamilyQueueFlags = VK_QUEUE_VIDEO_DECODE_BIT_KHR | VK_QUEUE_TRANSFER_BIT;
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
	const bool requireQueries = true;

	uint32_t physDevices_count = 0;
	vkEnumeratePhysicalDevices(s->instance, &physDevices_count, NULL);
	log_msg(LOG_LEVEL_DEBUG, "[vulkan_decode] Found %d physical devices.\n", physDevices_count);
	if (physDevices_count == 0)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] No physical devices found!\n");
		destroy_debug_messenger(s->instance, s->debugMessenger);
		vkDestroyInstance(s->instance, NULL);
        unload_vulkan(s);
		free(pps_array);
		free(sps_array);
		free(s);
		return NULL;
	}

	VkPhysicalDevice *physDevices = (VkPhysicalDevice*)calloc(physDevices_count, sizeof(VkPhysicalDevice));
	if (physDevices == NULL)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to allocate array for devices!\n");
		destroy_debug_messenger(s->instance, s->debugMessenger);
		vkDestroyInstance(s->instance, NULL);
        unload_vulkan(s);
		free(pps_array);
		free(sps_array);
		free(s);
		return NULL;
	}

	vkEnumeratePhysicalDevices(s->instance, &physDevices_count, physDevices);
	VkQueueFamilyProperties2 chosen_queue_family;
	VkQueueFamilyVideoPropertiesKHR chosen_queue_video_props;
	s->physicalDevice = choose_physical_device(physDevices, physDevices_count, requestedFamilyQueueFlags,
											   requiredDeviceExtensions, requireQueries,
											   &s->queueFamilyIdx, &chosen_queue_family, &chosen_queue_video_props);
	free(physDevices);
	if (s->physicalDevice == VK_NULL_HANDLE)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to choose a appropriate physical device!\n");
		destroy_debug_messenger(s->instance, s->debugMessenger);
		vkDestroyInstance(s->instance, NULL);
        unload_vulkan(s);
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
	vkGetPhysicalDeviceProperties2KHR(s->physicalDevice, &physDeviceProperties);

	log_msg(LOG_LEVEL_INFO, "[vulkan_decode] Chosen physical device is: '%s', chosen queue family index is: %d\n", 
							physDeviceProperties.properties.deviceName, s->queueFamilyIdx);

	// ---Creating a logical device---
	VkPhysicalDeviceVideoMaintenance1FeaturesKHR enabledDeviceVideoFeatures1 =
							{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_MAINTENANCE_1_FEATURES_KHR,
							  .videoMaintenance1 = VK_TRUE };
	VkPhysicalDeviceVulkan13Features enabledDeviceFeatures13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
																 .pNext = requireQueries ? (void*)&enabledDeviceVideoFeatures1 : NULL,
																 .synchronization2 = 1, .maintenance4 = 1 };
	float queue_priorities = 1.0f;
	VkDeviceQueueCreateInfo queueCreateInfo = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
												.queueFamilyIndex = s->queueFamilyIdx,
												.queueCount = 1,
												.pQueuePriorities = &queue_priorities };
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
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to create a appropriate vulkan logical device!\n");
		destroy_debug_messenger(s->instance, s->debugMessenger);
		vkDestroyInstance(s->instance, NULL);
        unload_vulkan(s);
		free(pps_array);
		free(sps_array);
		free(s);
		return NULL;
	}

	vkGetDeviceQueue(s->device, s->queueFamilyIdx, 0, &s->decodeQueue);

	s->fence = VK_NULL_HANDLE;
	s->bitstreamBuffer = VK_NULL_HANDLE; //buffer gets created in allocate_buffers function
	s->bitstreamBufferMemory = VK_NULL_HANDLE;
	s->bitstreambufferSizeAlignment = 0;
	s->commandPool = VK_NULL_HANDLE;  	 //command pool gets created in prepare function
	s->cmdBuffer = VK_NULL_HANDLE;	  	 //same
	s->videoSession = VK_NULL_HANDLE; 	 //video session gets created in prepare function
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

	s->outputLumaPlane = VK_NULL_HANDLE;
	s->outputChromaPlane = VK_NULL_HANDLE;
	s->outputImageMemory = VK_NULL_HANDLE;
	s->outputChromaPlaneOffset = 0;

	s->queryPool = VK_NULL_HANDLE;

	log_msg(LOG_LEVEL_INFO, "[vulkan_decode] Initialization finished successfully.\n");
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
	struct state_vulkan_decompress *s = (struct state_vulkan_decompress *)state;
	if (!s) return;

	destroy_queries(s);

	if (vkDestroyVideoSessionParametersKHR != NULL && s->device != VK_NULL_HANDLE)
			vkDestroyVideoSessionParametersKHR(s->device, s->videoSessionParams, NULL);

	if (vkDestroyVideoSessionKHR != NULL && s->device != VK_NULL_HANDLE)
			vkDestroyVideoSessionKHR(s->device, s->videoSession, NULL);
	
	free_video_session_memory(s);

	destroy_output_queue(s);

	destroy_output_image(s);

	destroy_dpb(s);
	
	if (vkDestroyCommandPool != NULL && s->device != VK_NULL_HANDLE)
			vkDestroyCommandPool(s->device, s->commandPool, NULL);

	free_buffers(s);

	if (vkDestroyFence != NULL && s->device != VK_NULL_HANDLE)
			vkDestroyFence(s->device, s->fence, NULL);
	
	if (vkDestroyDevice != NULL) vkDestroyDevice(s->device, NULL);

	destroy_debug_messenger(s->instance, s->debugMessenger);

	if (vkDestroyInstance != NULL) vkDestroyInstance(s->instance, NULL);

	unload_vulkan(s);

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

static bool configure_with(struct state_vulkan_decompress *s, struct video_desc desc)
{
	const char *spec_name = get_codec_name_long(desc.color_spec);

	s->codecOperation = VK_VIDEO_CODEC_OPERATION_NONE_KHR;
	VkVideoCodecOperationFlagsKHR videoCodecOperation = codec_to_vulkan_flag(desc.color_spec);

	if (!(s->queueVideoFlags & videoCodecOperation))
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Wanted color spec: '%s' is not supported by chosen vulkan queue family!\n", spec_name);
		return false;
	}

	assert(videoCodecOperation != VK_VIDEO_CODEC_OPERATION_NONE_KHR);

	s->codecOperation = videoCodecOperation;

	s->width = desc.width;
	s->height = desc.height;

	// assuming NV12 format
	s->lumaSize = s->width * s->height;
	s->chromaSize = (s->width / 2) * (s->height / 2);

	s->dpbHasDefinedLayout = false;
	s->dpbFormat = VK_FORMAT_UNDEFINED;
	s->referenceSlotsQueue_start = 0;
	s->referenceSlotsQueue_count = 0;

	s->prev_poc_lsb = 0;
	s->prev_poc_msb = 0;
	s->idr_frame_seq = 0;
	s->current_frame_seq = 0;
	s->poc_wrap = 0;
	s->last_poc = -999999; 				// some invalid POC that should be smaller than any encountered valid POC
	s->last_displayed_frame_seq = -1;
	s->last_displayed_poc = -999999; 	// some invalid POC that should be smaller than any encountered valid POC

	s->outputFrameQueue_data_size = 0;
	s->outputFrameQueue_capacity = 0;
	s->outputFrameQueue_count = 0;
	s->outputFrameQueue_start = 0;
	s->outputFrameQueue_data = NULL;

	return true;
}

static int vulkan_decompress_reconfigure(void *state, struct video_desc desc,
										 int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
	struct state_vulkan_decompress *s = (struct state_vulkan_decompress *)state;
	if (!s) return false;

	if (out_codec == VIDEO_CODEC_NONE) log_msg(LOG_LEVEL_NOTICE, "[vulkan_decode] Requested probing.\n");

	if (desc.tile_count != 1)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Tiled video formats are not supported!\n");
		return false;
	}

	s->prepared = false;
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
	struct state_vulkan_decompress *s = (struct state_vulkan_decompress *)state;
    UNUSED(s);
	
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
	UNUSED(internal);

	if (ugc != VIDEO_CODEC_NONE && ugc != I420) return 2500;

	VkVideoCodecOperationFlagsKHR vulkanFlag = codec_to_vulkan_flag(compression);

	if (vulkanFlag == VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR) return 100;
	if (vulkanFlag == VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR) return 500;

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

static bool check_format(const VkVideoFormatPropertiesKHR *props, const VkFormatProperties3 *extraProps)
{
	VkFormatFeatureFlags2 optFlags = extraProps->optimalTilingFeatures;
	
	return props->format == VK_FORMAT_G8_B8R8_2PLANE_420_UNORM && //TODO allow different formats
		   optFlags & VK_FORMAT_FEATURE_2_VIDEO_DECODE_DPB_BIT_KHR &&
		   optFlags & VK_FORMAT_FEATURE_2_VIDEO_DECODE_OUTPUT_BIT_KHR && //only for VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR
		   optFlags & VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT_KHR &&
		   optFlags & VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT;
}

static bool check_for_vulkan_format(VkPhysicalDevice physDevice, VkPhysicalDeviceVideoFormatInfoKHR videoFormatInfo,
									VkVideoFormatPropertiesKHR *formatProperties)
{
	uint32_t properties_count = 0;

	VkResult result = vkGetPhysicalDeviceVideoFormatPropertiesKHR(physDevice, &videoFormatInfo,
																  &properties_count, NULL);
	if (result != VK_SUCCESS)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to get video format properties of physical device!\n");
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
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to allocate memory for video format properties!\n");
		return false;
	}
	for (uint32_t i = 0; i < properties_count; ++i)
			properties[i].sType = VK_STRUCTURE_TYPE_VIDEO_FORMAT_PROPERTIES_KHR;

	result = vkGetPhysicalDeviceVideoFormatPropertiesKHR(physDevice, &videoFormatInfo,
														 &properties_count, properties);
	if (result != VK_SUCCESS)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to get video format properties of physical device!\n");
		free(properties);
		return false;
	}

	log_msg(LOG_LEVEL_INFO, "[vulkan_decode] Found %d suitable formats.\n", properties_count);

	for (uint32_t i = 0; i < properties_count; ++i)
	{
		VkFormat format = properties[i].format;

		VkFormatProperties3 extraProps = { .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3 };
		VkFormatProperties2 formatProps = { .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
											.pNext = (void*)&extraProps };
		vkGetPhysicalDeviceFormatProperties2(physDevice, format, &formatProps);

		//DEBUG
		//printf("\tformat: %d, image_type: %d, imageCreateFlags: %d, imageTiling: %d, imageUsageFlags: %d\n",
		//		format, properties[i].imageType, properties[i].imageCreateFlags, properties[i].imageTiling, properties[i].imageUsageFlags);
		//printf("\tusage flags: ");
		//print_bits((unsigned char)(properties[i].imageUsageFlags >> 8)); 
		//print_bits((unsigned char)properties[i].imageUsageFlags);
		//putchar('\n');
		
		bool blitSrc = VK_FORMAT_FEATURE_2_BLIT_SRC_BIT & extraProps.optimalTilingFeatures ? 1 : 0;
		bool blitDst = VK_FORMAT_FEATURE_2_BLIT_DST_BIT & extraProps.optimalTilingFeatures ? 1 : 0;
		bool decodeOutput = VK_FORMAT_FEATURE_2_VIDEO_DECODE_OUTPUT_BIT_KHR & extraProps.optimalTilingFeatures ? 1 : 0;
		bool decodeDPB = VK_FORMAT_FEATURE_2_VIDEO_DECODE_DPB_BIT_KHR & extraProps.optimalTilingFeatures ? 1 : 0;
		bool transferSrc = VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT_KHR & extraProps.optimalTilingFeatures ? 1 : 0;
		bool transferDst = VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT_KHR  & extraProps.optimalTilingFeatures ? 1 : 0;
		bool hostTransfer = VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT & extraProps.optimalTilingFeatures ? 1 : 0;
		
		log_msg(LOG_LEVEL_DEBUG, "[vulkan_decode] format %d - blit: %d %d, decode out: %d, decode dpb: %d, transfer: %d %d, host transfer: %d\n",
								 format, blitSrc, blitDst, decodeOutput, decodeDPB, transferSrc, transferDst, hostTransfer);
		VkComponentMapping swizzle = properties[i].componentMapping;
		log_msg(LOG_LEVEL_DEBUG, "[vulkan_decode] format %d - swizzle identity - r: %d, g: %d, b: %d, a: %d\n", format,
								 swizzle.r == VK_COMPONENT_SWIZZLE_IDENTITY, swizzle.g == VK_COMPONENT_SWIZZLE_IDENTITY,
								 swizzle.b == VK_COMPONENT_SWIZZLE_IDENTITY, swizzle.a == VK_COMPONENT_SWIZZLE_IDENTITY);

		if (check_format(properties + i, &extraProps))
		{
			if (formatProperties != NULL) *formatProperties = properties[i];
			
			free(properties);
			return true;
		}
	}

	log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Wanted output video format is not supported!\n");
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
	assert(s->bitstreamBuffer == VK_NULL_HANDLE);
	assert(s->bitstreamBufferMemory == VK_NULL_HANDLE);

	const VkDeviceSize wantedBitstreamBufferSize = 10 * 1024 * 1024; //TODO magic number for size
	s->bitstreambufferSizeAlignment = videoCapabilities.minBitstreamBufferSizeAlignment;
	s->bitstreamBufferSize = (wantedBitstreamBufferSize + (s->bitstreambufferSizeAlignment - 1))
							 & ~(s->bitstreambufferSizeAlignment - 1); //alignment bit mask magic
	VkBufferCreateInfo bitstreamBufferInfo = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
									  		   .pNext = (void*)&videoProfileList,
									  		   .flags = 0,
									  		   .usage = VK_BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR,
									  		   .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
									  		   .size = s->bitstreamBufferSize,
									  		   .queueFamilyIndexCount = 1,
									  		   .pQueueFamilyIndices = &s->queueFamilyIdx };
	VkResult result = vkCreateBuffer(s->device, &bitstreamBufferInfo, NULL, &s->bitstreamBuffer);
	if (result != VK_SUCCESS)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to create vulkan buffer for decoding!\n");
		free_buffers(s);

		return false;
	}

	assert(s->bitstreamBuffer != VK_NULL_HANDLE);

	VkMemoryRequirements bitstreamBufferMemReq;
	vkGetBufferMemoryRequirements(s->device, s->bitstreamBuffer, &bitstreamBufferMemReq);

	uint32_t memType_idx = 0;
	if (!find_memory_type(s, bitstreamBufferMemReq.memoryTypeBits,
						  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
						  &memType_idx))
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to find required memory type for vulkan bitstream buffer!\n");
		free_buffers(s);

		return false;
	}

	VkMemoryAllocateInfo memAllocInfo = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
										  .allocationSize = bitstreamBufferMemReq.size,
										  .memoryTypeIndex = memType_idx };
	result = vkAllocateMemory(s->device, &memAllocInfo, NULL, &s->bitstreamBufferMemory);
	if (result != VK_SUCCESS)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to allocate memory for vulkan bitstream buffer!\n");
		free_buffers(s);

		return false;
	}

	assert(s->bitstreamBufferMemory != VK_NULL_HANDLE);

	result = vkBindBufferMemory(s->device, s->bitstreamBuffer, s->bitstreamBufferMemory, 0);
	if (result != VK_SUCCESS)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to bind vulkan memory to vulkan bitstream buffer!\n");
		free_buffers(s);
		
		return false;
	}

	return true;
}

static void free_buffers(struct state_vulkan_decompress *s)
{
	//buffers needs to get destroyed first
	if (vkDestroyBuffer != NULL && s->device != VK_NULL_HANDLE)
				vkDestroyBuffer(s->device, s->bitstreamBuffer, NULL);

	s->bitstreamBuffer = VK_NULL_HANDLE;
	s->bitstreamBufferSize = 0;
	s->bitstreambufferSizeAlignment = 0;

	if (vkFreeMemory != NULL && s->device != VK_NULL_HANDLE)
				vkFreeMemory(s->device, s->bitstreamBufferMemory, NULL);

	s->bitstreamBufferMemory = VK_NULL_HANDLE;
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
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to get vulkan video session memory requirements!\n");
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
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to allocate memory for vulkan video session!\n");
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
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to get vulkan video session memory requirements!\n");
		free(memoryRequirements);
		free(bindMemoryInfo);
		free_video_session_memory(s);

		return false;
	}

	for (uint32_t i = 0; i < memoryRequirements_count; ++i)
	{
        uint32_t memoryTypeIndex = 0;
		if (!find_memory_type(s, memoryRequirements[i].memoryRequirements.memoryTypeBits,
							  0, &memoryTypeIndex))
		{
			log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] No suitable memory type for vulkan video session requirments!\n");
			free(memoryRequirements);
			free(bindMemoryInfo);
			free_video_session_memory(s);

			return false;
		}

		VkMemoryAllocateInfo allocateInfo = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
											  .allocationSize = memoryRequirements[i].memoryRequirements.size,
											  .memoryTypeIndex = memoryTypeIndex };
		result = vkAllocateMemory(s->device, &allocateInfo, NULL, &s->videoSessionMemory[i]);
		if (result != VK_SUCCESS)
		{
			log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to allocate vulkan memory!\n");
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
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Can't bind video session memory!\n");
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
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to reset the vulkan command buffer!\n");
		return false;
	}
	
	VkCommandBufferBeginInfo cmdBufferBeginInfo = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
													.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
	result = vkBeginCommandBuffer(s->cmdBuffer, &cmdBufferBeginInfo);
	if (result != VK_SUCCESS)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to begin command buffer recording!\n");
		return false;
	}

	return true;
}

static bool end_cmd_buffer(struct state_vulkan_decompress *s)
{
	VkResult result = vkEndCommandBuffer(s->cmdBuffer);
	if (result == VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to end command buffer recording - Invalid video standard parameters\n");
		return false;
	}
	else if (result != VK_SUCCESS)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to end command buffer recording!\n");
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
			if (access != NULL) *access = VK_ACCESS_2_TRANSFER_READ_BIT;
			break;
		case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: // copying into outputImage
			if (stage != NULL) *stage = VK_PIPELINE_STAGE_2_COPY_BIT_KHR;
			if (access != NULL) *access = VK_ACCESS_2_TRANSFER_WRITE_BIT;
			break;
		/*case VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR: // beginning of the cmd buffer when preparing for decode, or after it
			if (stage != NULL) *stage = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR;
			if (access != NULL) *access = VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR |
										  VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR;
			break;*/
		case VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR: // beginning of the cmd buffer when preparing for decode, or after it
			if (stage != NULL) *stage = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR;
			if (access != NULL) *access = VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR |
										  VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR;
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

static bool create_output_image(struct state_vulkan_decompress *s)
{
	assert(s->device != VK_NULL_HANDLE);
	assert(s->outputLumaPlane == VK_NULL_HANDLE);
	assert(s->outputChromaPlane == VK_NULL_HANDLE);
	assert(s->outputImageMemory == VK_NULL_HANDLE);

	const VkExtent3D videoSize = { s->width, s->height, 1 }; //depth must be 1 for VK_IMAGE_TYPE_2D

	// ---Creating the luma plane---
	VkImageCreateInfo lumaPlaneInfo = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
									 	.pNext = NULL,
									 	.flags = 0,
									 	.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
									 			 VK_IMAGE_USAGE_TRANSFER_DST_BIT,
									 	.imageType = VK_IMAGE_TYPE_2D,
									 	.mipLevels = 1,
									 	.samples = VK_SAMPLE_COUNT_1_BIT,
									 	.format = VK_FORMAT_R8_UNORM,
									 	.extent = videoSize,
									 	.arrayLayers = 1,
									 	.tiling = VK_IMAGE_TILING_LINEAR,
									 	.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
									 	.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
									 	.queueFamilyIndexCount = 1,
									 	.pQueueFamilyIndices = &s->queueFamilyIdx };
	VkResult result = vkCreateImage(s->device, &lumaPlaneInfo, NULL, &s->outputLumaPlane);
	if (result != VK_SUCCESS)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to create vulkan image for the output frame (luma plane)!\n");
		destroy_output_image(s);
		return false;
	}

	// ---Creating the chroma plane---
	VkImageCreateInfo chromaPlaneInfo = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
									 	   .pNext = NULL,
									 	   .flags = 0,
									 	   .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
									 	   			VK_IMAGE_USAGE_TRANSFER_DST_BIT,
									 	   .imageType = VK_IMAGE_TYPE_2D,
									 	   .mipLevels = 1,
									 	   .samples = VK_SAMPLE_COUNT_1_BIT,
									 	   .format = VK_FORMAT_R8G8_UNORM,
									 	   .extent = { videoSize.width / 2, videoSize.height / 2, 1 },
									 	   .arrayLayers = 1,
									 	   .tiling = VK_IMAGE_TILING_LINEAR,
									 	   .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
									 	   .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
									 	   .queueFamilyIndexCount = 1,
									 	   .pQueueFamilyIndices = &s->queueFamilyIdx };
	result = vkCreateImage(s->device, &chromaPlaneInfo, NULL, &s->outputChromaPlane);
	if (result != VK_SUCCESS)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to create vulkan image for the output frame (chroma plane)!\n");
		destroy_output_image(s);
		return false;
	}

	// ---Getting wanted memory requirements and memory type---
	VkMemoryRequirements lumaMemReq, chromaMemReq;
	vkGetImageMemoryRequirements(s->device, s->outputLumaPlane, &lumaMemReq);
	vkGetImageMemoryRequirements(s->device, s->outputChromaPlane, &chromaMemReq);

	VkDeviceSize chromaAlignment = chromaMemReq.alignment;
	s->outputChromaPlaneOffset = (lumaMemReq.size + (chromaAlignment - 1)) & ~(chromaAlignment - 1); //alignment bit mask magic

	log_msg(LOG_LEVEL_DEBUG, "[vulkan_decode] Luma size: %llu, chroma size: %llu, chroma offset: %llu.\n",
							lumaMemReq.size, chromaMemReq.size, s->outputChromaPlaneOffset);

	uint32_t imgMemoryTypeIdx = 0;
	if (!find_memory_type(s, lumaMemReq.memoryTypeBits & chromaMemReq.memoryTypeBits, // taking the susbet
						  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
						  &imgMemoryTypeIdx))
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to find required memory type for output image!\n");
		destroy_output_image(s);
		return false;
	}
	
	// ---Allocating memory for both planes---
	VkMemoryAllocateInfo allocInfo = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
									   .allocationSize = s->outputChromaPlaneOffset + chromaMemReq.size,
									   .memoryTypeIndex = imgMemoryTypeIdx };
	result = vkAllocateMemory(s->device, &allocInfo, NULL, &s->outputImageMemory);
	if (result != VK_SUCCESS)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to allocate device memory for output image!\n");
		destroy_output_image(s);
		return false;
	}

	// ---Binding memory for each plane---
	result = vkBindImageMemory(s->device, s->outputLumaPlane, s->outputImageMemory, 0);
	if (result != VK_SUCCESS)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to bind vulkan device memory to output image (luma plane)!\n");
		destroy_output_image(s);
		return false;
	}

	result = vkBindImageMemory(s->device, s->outputChromaPlane, s->outputImageMemory, s->outputChromaPlaneOffset);
	if (result != VK_SUCCESS)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to bind vulkan device memory to output image (chroma plane)!\n");
		destroy_output_image(s);
		return false;
	}

	return true;
}

static void destroy_output_image(struct state_vulkan_decompress *s)
{
	// freeing both planes
	if (vkDestroyImage != NULL && s->device != VK_NULL_HANDLE)
	{
		vkDestroyImage(s->device, s->outputLumaPlane, NULL);
		vkDestroyImage(s->device, s->outputChromaPlane, NULL);
	}

	s->outputLumaPlane = VK_NULL_HANDLE;
	s->outputChromaPlane = VK_NULL_HANDLE;

	// freeing backing device memory
	if (vkFreeMemory != NULL && s->device != VK_NULL_HANDLE) vkFreeMemory(s->device, s->outputImageMemory, NULL);
	
	s->outputImageMemory = VK_NULL_HANDLE;
}

static bool create_dpb(struct state_vulkan_decompress *s, VkVideoProfileListInfoKHR *videoProfileList)
{
	// Creates the DPB (decoded picture buffer) images and output image for decoded result,
	// if success then returns true - resulting images must be destroyed using destroy_dpb,
	// all created images are left in undefined layout
	assert(s->device != VK_NULL_HANDLE);
	assert(s->dpbFormat != VK_FORMAT_UNDEFINED);

	const VkImageType imageType = VK_IMAGE_TYPE_2D;
	const VkExtent3D videoSize = { s->width, s->height, 1 }; //depth must be 1 for VK_IMAGE_TYPE_2D

	//imageCreateMaxMipLevels, imageCreateMaxArrayLayers, imageCreateMaxExtent, and imageCreateSampleCounts

	// ---Creating DPB VkImages---
	VkImageCreateInfo dpbImgInfo = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
									 .pNext = (void*)videoProfileList,
									 .flags = 0,
									 .usage = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR |
									 		  //VK_IMAGE_USAGE_VIDEO_DECODE_SRC_BIT_KHR |
								 		   	  VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR |
								 			  VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
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
		VkResult result = vkCreateImage(s->device, &dpbImgInfo, NULL, s->dpb + i);
		if (result != VK_SUCCESS)
		{
			log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to create vulkan image(%zu) for DPB slot!\n", i);
			destroy_dpb(s);
			return false;
		}
	}

	// ---Device memory allocation---
	VkMemoryRequirements dpbImgMemoryRequirements;
	vkGetImageMemoryRequirements(s->device, s->dpb[0], &dpbImgMemoryRequirements);

	uint32_t imgMemoryTypeIdx = 0;
	if (!find_memory_type(s, dpbImgMemoryRequirements.memoryTypeBits, 0, &imgMemoryTypeIdx))
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to find required memory type for DPB!\n");
		destroy_dpb(s);
		return false;
	}

	VkDeviceSize dpbImgAlignment = dpbImgMemoryRequirements.alignment;
	VkDeviceSize dpbImgAlignedSize = (dpbImgMemoryRequirements.size + (dpbImgAlignment - 1))
									 & ~(dpbImgAlignment - 1); //alignment bit mask magic
	VkDeviceSize dpbSize = dpbImgAlignedSize * dpb_len;

	VkMemoryAllocateInfo allocInfo = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
									   .allocationSize = dpbSize,
									   .memoryTypeIndex = imgMemoryTypeIdx };
	VkResult result = vkAllocateMemory(s->device, &allocInfo, NULL, &s->dpbMemory);
	if (result != VK_SUCCESS)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to allocate vulkan device memory for DPB!\n");
		destroy_dpb(s);
		return false;
	}

	// ---Binding memory for DPB pictures---
	for (size_t i = 0; i < dpb_len; ++i)
	{
		result = vkBindImageMemory(s->device, s->dpb[i], s->dpbMemory, i * dpbImgAlignedSize);
		if (result != VK_SUCCESS)
		{
			log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to bind vulkan device memory to DPB image (idx: %zu)!\n", i);
			destroy_dpb(s);
			return false;
		}
	}

	// ---Creating DPB image views---
	VkImageViewUsageCreateInfo viewUsageInfo = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
												 .usage = VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR |
												 		  VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR };
	VkImageSubresourceRange viewSubresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
													 .baseMipLevel = 0, .levelCount = 1,
													 .baseArrayLayer = 0, .layerCount = 1 };
	VkImageViewCreateInfo dpbViewInfo = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
										  .pNext = (void*)&viewUsageInfo,
										  .flags = 0,
										  .image = VK_NULL_HANDLE, // gets correctly set in the for loop
										  .viewType = VK_IMAGE_VIEW_TYPE_2D,
										  .format = dpbImgInfo.format,
										  .components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                            			  				  VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY },
										  .subresourceRange = viewSubresourceRange };
	
	for (size_t i = 0; i < dpb_len; ++i)
	{
		assert(s->dpb[i] != VK_NULL_HANDLE);

		dpbViewInfo.image = s->dpb[i];
		result = vkCreateImageView(s->device, &dpbViewInfo, NULL, s->dpbViews + i);
		if (result != VK_SUCCESS)
		{
			log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to create vulkan image view(%zu) for DPB slot!\n", i);
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

static bool create_output_queue(struct state_vulkan_decompress *s)
{
	// Creates queue for output decoded frames
	assert(s->outputFrameQueue_data == NULL);

	s->outputFrameQueue_data_size = s->lumaSize + 2 * s->chromaSize;
	assert(s->outputFrameQueue_data_size > 0);
	
	size_t capacity = MAX_OUTPUT_FRAMES_QUEUE_SIZE;
	for (; capacity > 0; capacity /= 2)
	{
		uint8_t *memory = (uint8_t*)calloc(capacity, s->outputFrameQueue_data_size);
		if (memory != NULL)
		{
			s->outputFrameQueue_data = memory;
			break;
		}
	}

	if (s->outputFrameQueue_data == NULL)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to allocate memory for output decoded frames queue!\n");
		return false;
	}

	s->outputFrameQueue_capacity = capacity;
	s->outputFrameQueue_count = 0; // no data in the queue at the beginning
	s->outputFrameQueue_start = 0; // we start at index 0
	// zeroing out the first s->outputFrameQueue_capacity members of output frame queue (only those that will be used)
	memset(s->outputFrameQueue, 0, sizeof(frame_data_t) * MAX_OUTPUT_FRAMES_QUEUE_SIZE);

	s->last_displayed_frame_seq = -1;
	s->last_displayed_poc = -999999; // some invalid POC that should be smaller than any encountered valid POC

	log_msg(LOG_LEVEL_INFO, "[vulkan_decode] Allocated output decoded frame queue of capacity: %zu\n",
							s->outputFrameQueue_capacity);

	return true;
}

static void destroy_output_queue(struct state_vulkan_decompress *s)
{
	free(s->outputFrameQueue_data);
	s->outputFrameQueue_data = NULL;

	s->outputFrameQueue_data_size = 0;
	s->outputFrameQueue_capacity = 0;
	s->outputFrameQueue_count = 0;
	s->outputFrameQueue_start = 0;
}

static frame_data_t get_frame_from_output_queue(struct state_vulkan_decompress *s, size_t index)
{
	// returns the member of output frames queue on the given index
	// correctly handles wrapping, NO bounds checks!
	uint32_t wrappedIdx = (s->outputFrameQueue_start + index) % s->outputFrameQueue_capacity;
	return s->outputFrameQueue[wrappedIdx];
}

//DEBUG
/*static void output_queue_print(struct state_vulkan_decompress *s)
{
	for (size_t i = 0; i < s->outputFrameQueue_count; ++i)
	{
		frame_data_t frame = get_frame_from_output_queue(s, i);

		printf("\tIDR: %d, poc_wrap: %d, poc: %d, frame_seq: %d\n",
				frame.idr_frame_seq, frame.poc_wrap, frame.poc, frame.frame_seq);
	}
	putchar('\n');
}*/

static size_t get_unused_output_queue_data_index(struct state_vulkan_decompress *s)
{
	// returns the first index to s->outputFrameQueue_data that's not used by any
	// other entry in the output queue (s->outputFrameQueue)
	// such index must exist as the output queue is smaller than capacity by at least 1
	bool checks[MAX_OUTPUT_FRAMES_QUEUE_SIZE] = { 0 };

	for (size_t i = 0; i < s->outputFrameQueue_count; ++i)
	{
		size_t data_idx = get_frame_from_output_queue(s, i).data_idx;
		
		assert(data_idx < s->outputFrameQueue_capacity);
		assert(!checks[data_idx]);

		checks[data_idx] = true;
	}

	for (size_t i = 0; i < s->outputFrameQueue_capacity; ++i)
	{
		if (!checks[i]) return i;
	}

	assert(!"Failed to find unused output queue data index!"); // should not happen
	return 0;
}

static void output_queue_swap_frames(struct state_vulkan_decompress *s, size_t index1, size_t index2)
{
	assert(index1 != index2);
	assert(index1 < s->outputFrameQueue_count);
	assert(index2 < s->outputFrameQueue_count);

	size_t wrapped1 = (s->outputFrameQueue_start + index1) % s->outputFrameQueue_capacity;
	size_t wrapped2 = (s->outputFrameQueue_start + index2) % s->outputFrameQueue_capacity;

	frame_data_t tmp = s->outputFrameQueue[wrapped1];
	s->outputFrameQueue[wrapped1] = s->outputFrameQueue[wrapped2];
	s->outputFrameQueue[wrapped2] = tmp;
}

static bool display_order_smaller_than(frame_data_t frame1, frame_data_t frame2)
{
	// returns true if frame1 has lower display order than frame2
	return frame1.idr_frame_seq < frame2.idr_frame_seq ||
		  (frame1.idr_frame_seq == frame2.idr_frame_seq && frame1.poc_wrap < frame2.poc_wrap) ||
		  (frame1.idr_frame_seq == frame2.idr_frame_seq && frame1.poc_wrap == frame2.poc_wrap && frame1.poc < frame2.poc);
}

static size_t output_queue_bubble_index_forward(struct state_vulkan_decompress *s, size_t index)
{
	// bubbles queue frame on given index forwards in the queue with respect to display order
	// (lower display order is closer to the queue start than higher display order)
	// returns the new bubbled index of this frame

	frame_data_t bubbled = get_frame_from_output_queue(s, index);
	while (index > 0)
	{
		frame_data_t compared_to = get_frame_from_output_queue(s, index - 1);
		if (display_order_smaller_than(compared_to, bubbled)) // we found frame that has lower display order than bubbled frame
		{
			break;
		}
		// else swap frames
		output_queue_swap_frames(s, index - 1, index);

		--index;
	}

	return index;
}

static frame_data_t * make_new_entry_in_output_queue(struct state_vulkan_decompress *s, slice_info_t slice_info)
{
	assert(s->outputFrameQueue_count <= s->outputFrameQueue_capacity);

	if (s->outputFrameQueue_count == s->outputFrameQueue_capacity) // the output queue is full => make space
	{
		// currently we make space in the queue by removing the first element
		// such element should also be the oldest (in display order)
		s->outputFrameQueue_start = (s->outputFrameQueue_start + 1) % s->outputFrameQueue_capacity;
		--(s->outputFrameQueue_count); // correct the size of the queue
	}

	UNUSED(slice_info);
	frame_data_t frame = { .data_idx = get_unused_output_queue_data_index(s), // should not fail as there is now room in the queue
						   .data_len = s->outputFrameQueue_data_size,
						   .idr_frame_seq = s->idr_frame_seq,
						   .frame_seq = slice_info.frame_seq,
						   .poc = slice_info.poc,
						   .poc_wrap = s->poc_wrap };
	
	size_t queue_index = s->outputFrameQueue_count;
	size_t queue_index_wrapped = (s->outputFrameQueue_start + queue_index) % s->outputFrameQueue_capacity;
	s->outputFrameQueue[queue_index_wrapped] = frame;
	++(s->outputFrameQueue_count);

	//queue_index = output_queue_bubble_index_forward(s, queue_index); // bubble new element and return it's bubbled index
	// wrap again for updated queue_index
	queue_index_wrapped = (s->outputFrameQueue_start + queue_index) % s->outputFrameQueue_capacity;

	return s->outputFrameQueue + queue_index_wrapped;
}

static frame_data_t * output_queue_pop_first(struct state_vulkan_decompress *s)
{
	if (s->outputFrameQueue_count == 0) return NULL; // empty queue

	frame_data_t *popped = s->outputFrameQueue + s->outputFrameQueue_start;

	s->outputFrameQueue_start = (s->outputFrameQueue_start + 1) % s->outputFrameQueue_capacity;
	--(s->outputFrameQueue_count);

	return popped;
}

static frame_data_t * output_queue_get_next_frame(struct state_vulkan_decompress *s)
{
	//output_queue_print(s);

	if (s->outputFrameQueue_count == 0)
	{
		return NULL; // empty queue case
	}

	const frame_data_t *first = s->outputFrameQueue + s->outputFrameQueue_start;

	//NOTE: this may lead to potential wrong display order when s->outputFrameQueue_capacity is small!
	//		(smaller than current GOP size) So maybe could be commented out
	if (s->outputFrameQueue_count == s->outputFrameQueue_capacity)
	{	// queue is full => display the first queued frame, as it would get throwed out in another decompress anyways
		return output_queue_pop_first(s);
	}

	if (first->frame_seq < s->idr_frame_seq)
	{	// first queued frame is from another GOP than currently decoded one => display it 
		return output_queue_pop_first(s);
	}

	return NULL;
}

static bool create_queries(struct state_vulkan_decompress *s, VkVideoProfileInfoKHR *profile)
{
	// returns false when VULKAN_QUERIES macro not defined,
	// otherwise attempts to create query pool with 1 query for decode command,
	// returns whether the creation was successful
	#ifndef VULKAN_QUERIES
	UNUSED(s);
	UNUSED(profile);
	return false;
	#else
	assert(s->device != VK_NULL_HANDLE);
	assert(s->queryPool == VK_NULL_HANDLE);

	// Creating query pool for wanted decode query
	VkQueryPoolCreateInfo createInfo = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
										 .pNext = (void*)profile,
										 .flags = 0,
										 .queryType = VK_QUERY_TYPE_RESULT_STATUS_ONLY_KHR,
										 .queryCount = 1 };
	VkResult result = vkCreateQueryPool(s->device, &createInfo, NULL, &s->queryPool);
	if (result != VK_SUCCESS)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to create vulkan query pool!\n");
		destroy_queries(s); // probably useless
		return false;
	}

	return true;
	#endif
}

static void destroy_queries(struct state_vulkan_decompress *s)
{
	#ifndef VULKAN_QUERIES
	UNUSED(s);
	#else
	if (vkDestroyQueryPool != NULL && s->device != VK_NULL_HANDLE)
				vkDestroyQueryPool(s->device, s->queryPool, NULL);
	
	s->queryPool = VK_NULL_HANDLE;
	#endif
}

static bool prepare(struct state_vulkan_decompress *s, bool *wrong_pixfmt)
{
	assert(!s->prepared); //this function should be called only when decompress is not prepared

	*wrong_pixfmt = false;

	log_msg(LOG_LEVEL_INFO, "[vulkan_decode] Preparing with - depth: %d %d, subsampling: %d, profile: %d\n",
							s->depth_luma, s->depth_chroma, s->subsampling, (int)s->profileIdc);

	VkVideoChromaSubsamplingFlagsKHR chromaSubsampling = subsampling_to_vulkan_flag(s->subsampling);
	if (chromaSubsampling == VK_VIDEO_CHROMA_SUBSAMPLING_INVALID_KHR)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Got unsupported chroma subsampling!\n");
		*wrong_pixfmt = true;
		return false;
	}
	//NOTE: Otherwise vulkan video fails to work (on my hardware)
	else if (chromaSubsampling != VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Wrong chroma subsampling! Currently the only supported one is 4:2:0!\n");
		*wrong_pixfmt = true;
		return false;
	}

	VkVideoComponentBitDepthFlagBitsKHR vulkanChromaDepth = depth_to_vulkan_flag(s->depth_chroma);
	VkVideoComponentBitDepthFlagBitsKHR vulkanLumaDepth = depth_to_vulkan_flag(s->depth_luma);
	if (vulkanChromaDepth == VK_VIDEO_COMPONENT_BIT_DEPTH_INVALID_KHR ||
		vulkanLumaDepth == VK_VIDEO_COMPONENT_BIT_DEPTH_INVALID_KHR)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Got unsupported color channel depth!\n");
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
	VkVideoDecodeUsageInfoKHR decodeUsageHint = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_USAGE_INFO_KHR,
												  .pNext = isH264 ? (void*)&h264Profile : (void*)&h265Profile,
												  .videoUsageHints = VK_VIDEO_DECODE_USAGE_STREAMING_BIT_KHR };
	VkVideoProfileInfoKHR videoProfile = { .sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
										   .pNext = (void*)&decodeUsageHint,
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
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to get physical device video capabilities!\n");
		// it's not obvious if to set '*wrong_pixfmt = true;'
		return false;
	}

	//TODO allow dpb be implemented using only one VkImage
	if (!(videoCapabilities.flags & VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR))
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Chosen physical device does not support separate reference images for DPB!\n");
		return false;
	}

	if (videoCapabilities.maxDpbSlots < MAX_REF_FRAMES + 1)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Chosen physical device does not support needed amount of DPB slots(%u)!\n",
								 MAX_REF_FRAMES + 1);
		return false;
	}

	if (videoCapabilities.maxActiveReferencePictures < MAX_REF_FRAMES)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Chosen physical device does not support needed amount of active reference pictures(%u)!\n",
								 MAX_REF_FRAMES);
		return false;
	}

	const VkExtent2D videoSize = { s->width, s->height };
	if (!does_video_size_fit(videoSize, videoCapabilities.minCodedExtent, videoCapabilities.maxCodedExtent))
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Requested video size: %ux%u does not fit in vulkan video extents."
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
														   .pNext = (void*)&videoProfileList,
														   .imageUsage = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR |
														   				 VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR |
																		 VK_IMAGE_USAGE_TRANSFER_SRC_BIT };

	assert(s->physicalDevice != VK_NULL_HANDLE);

	log_msg(LOG_LEVEL_DEBUG, "[vulkan_decode] Decode capability flags - coincide: %d, distinct: %d\n",
							 decodeCapabilitiesFlags & VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR ? 1 : 0,
							 decodeCapabilitiesFlags & VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_DISTINCT_BIT_KHR ? 1 : 0);

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
		log_msg(LOG_LEVEL_ERROR,
				"[vulkan_decode] Currently it is required for physical decoder to support DPB slot being the output as well!\n");
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
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Unsupported decodeCapabilitiesFlags value (%d)!\n", decodeCapabilitiesFlags);
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
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to create vulkan fence for synchronization!\n");
		return false;
	}

	// ---Creating bitstreamBuffer for encoded NAL bitstream---
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
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to create vulkan command pool!\n");
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
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to allocate vulkan command buffer!\n");
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

	// ---Creating the decoded output image---
	if (!create_output_image(s))
	{
		// err msg should get printed inside of create_output_image
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

	// ---Creting the decoded output frames queue---
	if (!create_output_queue(s))
	{
		// err msg should get printed inside of create_output_queue
		destroy_output_image(s);
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

	// ---Creating query pool---
	// is true when VULKAN_QUERIES defined and queries were created successfully
	bool query_ret = create_queries(s, &videoProfile);

	// ---Creating video session---
	assert(s->videoSession == VK_NULL_HANDLE);

	VkVideoSessionCreateInfoKHR sessionInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR,
												.pNext = NULL,
												.queueFamilyIndex = s->queueFamilyIdx,
												.flags = query_ret ? VK_VIDEO_SESSION_CREATE_INLINE_QUERIES_BIT_KHR : 0,
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
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to create vulkan video session!\n");
		destroy_queries(s);
		destroy_output_queue(s);
		destroy_output_image(s);
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
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to create vulkan video session parameters!\n");
		if (vkDestroyVideoSessionKHR != NULL)
		{
			vkDestroyVideoSessionKHR(s->device, s->videoSession, NULL);
			s->videoSession = VK_NULL_HANDLE;
		}
		destroy_queries(s);
		destroy_output_queue(s);
		destroy_output_image(s);
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
		destroy_queries(s);
		destroy_output_queue(s);
		destroy_output_image(s);
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

	log_msg(LOG_LEVEL_INFO, "[vulkan_decode] Preparation successful.\n");
	return true;
}

static slice_info_t get_ref_slot_from_queue(struct state_vulkan_decompress *s, uint32_t index)
{
	// returns the member of reference frames queue on the given index
	// correctly handles wrapping, NO bounds checks!
	uint32_t wrappedIdx = (s->referenceSlotsQueue_start + index) % MAX_REF_FRAMES;
	return s->referenceSlotsQueue[wrappedIdx];
}

//DEBUG
static void print_ref_queue(struct state_vulkan_decompress *s)
{
	for (uint32_t i = 0; i < s->referenceSlotsQueue_count; ++i)
	{
		const slice_info_t si = get_ref_slot_from_queue(s, i);
		printf("%d|%d|%d ", si.dpbIndex, si.nal_idc, si.frame_seq);
	}
}

static uint32_t smallest_dpb_index_not_in_queue(struct state_vulkan_decompress *s)
{
	// returns the smallest index into DPB that's not in the reference queue
	// such index must exist as the reference queue is smaller than capacity by at least 1
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

static void ref_queue_swap_references(struct state_vulkan_decompress *s, uint32_t index1, uint32_t index2)
{
	assert(index1 != index2);
	assert(index1 < s->referenceSlotsQueue_count);
	assert(index2 < s->referenceSlotsQueue_count);

	uint32_t wrapped1 = (s->referenceSlotsQueue_start + index1) % MAX_REF_FRAMES;
	uint32_t wrapped2 = (s->referenceSlotsQueue_start + index2) % MAX_REF_FRAMES;

	slice_info_t tmp = s->referenceSlotsQueue[wrapped1];
	s->referenceSlotsQueue[wrapped1] = s->referenceSlotsQueue[wrapped2];
	s->referenceSlotsQueue[wrapped2] = tmp;
}

static bool ref_slot_priority_smaller_than(slice_info_t ref1, slice_info_t ref2)
{
	// returns true if ref1 has lower or same priorty in the reference queue than ref2
	return ref1.nal_idc < ref2.nal_idc ||
		  (ref1.nal_idc == ref2.nal_idc && ref1.frame_seq <= ref2.frame_seq);
}

static uint32_t ref_slot_queue_bubble_index_forward(struct state_vulkan_decompress *s, uint32_t index)
{
	// bubbles reference picture at given index in the referecene queue forwards with respect to priority
	// (lower priority is closer to the queue start -> dropped from the queue earlier if needed)
	// returns the new bubbled index of this picture
	
	slice_info_t bubbled = get_ref_slot_from_queue(s, index);
	while (index > 0)
	{
		slice_info_t compared_to = get_ref_slot_from_queue(s, index - 1);
		if (ref_slot_priority_smaller_than(compared_to, bubbled)) // we found lower priority reference
		{
			break;
		}
		// else swap frames
		ref_queue_swap_references(s, index - 1, index);

		--index;
	}

	return index;
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

	uint32_t idx = s->referenceSlotsQueue_count;
	uint32_t wrappedIdx = (s->referenceSlotsQueue_start + idx) % MAX_REF_FRAMES;
	s->referenceSlotsQueue[wrappedIdx] = slice_info;
	++(s->referenceSlotsQueue_count);

	//ref_slot_queue_bubble_index_forward(s, idx);
}

static void clear_the_ref_slot_queue(struct state_vulkan_decompress *s)
{
	s->referenceSlotsQueue_count = 0;
	s->resetVideoCoding = true; // as we cleared the queue we can use this to invalidate all activated dpb slots
}

static void fill_ref_picture_infos(struct state_vulkan_decompress *s,
								   VkVideoReferenceSlotInfoKHR refInfos[], VkVideoPictureResourceInfoKHR picInfos[],
								   VkVideoDecodeH264DpbSlotInfoKHR h264SlotInfos[], StdVideoDecodeH264ReferenceInfo h264Infos[],								 
								   uint32_t max_count, uint32_t sps_ref_frames_num, bool isH264, uint32_t *out_count)
{
	// count is a size of both given arrays (should be at most same as MAX_REF_FRAMES)
	assert(max_count <= MAX_REF_FRAMES);
	assert(isH264); //TODO H.265

	VkExtent2D videoSize = { s->width, s->height };

	uint32_t ref_count = min(s->referenceSlotsQueue_count, sps_ref_frames_num);
	uint32_t ref_offset = s->referenceSlotsQueue_count - ref_count;
	/*rintf("Reference queue (%u):\n", ref_count);
	print_ref_queue(s);
	putchar('\n');*/

	for (uint32_t i = 0; i < max_count && i < ref_count; ++i)
	{
		const slice_info_t slice_info = get_ref_slot_from_queue(s, ref_offset + i);
		
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

static void begin_video_coding_scope(struct state_vulkan_decompress *s, VkVideoReferenceSlotInfoKHR *slotInfos, uint32_t slotInfos_count)
{
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

//DEBUG
/*static void print_bits(unsigned char num)
{
	unsigned int bit = 1<<(sizeof(unsigned char) *8 - 1);

	while (bit)
	{
		printf("%i ", num & bit ? 1 : 0);
		bit >>= 1;
	}
}*/

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

//DEBUG - copied from rtp/rtpenc_h264.c
/*static void print_nalu_name(int type)
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

//DEBUG
static void print_nalu_header(unsigned char header)
{
	printf("forbidden bit: %u, idc: %u, type: %u", header & 128 ? 1 : 0,
				H264_NALU_HDR_GET_NRI(header), H264_NALU_HDR_GET_TYPE(header));
}*/

static bool create_rbsp(const unsigned char *nal, size_t nal_len, uint8_t **rbsp, int *rbsp_len)
{
	// allocates memory for rbsp and fills it with correct rbsp corresponding to given nal buffer range
	// returns false if error and prints error msg
	assert(nal != NULL && rbsp != NULL && rbsp_len != NULL);

	if (sizeof(unsigned char) != sizeof(uint8_t))
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] H.264/H.265 stream handling requires sizeof(unsigned char) == sizeof(uint8_t)!\n");
		*rbsp_len = 0;
		return false;
	}

	*rbsp_len = (int)nal_len;
	assert(*rbsp_len >= 0);
	uint8_t *out = (uint8_t*)malloc(nal_len * sizeof(uint8_t));
	if (out == NULL)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to allocate memory for RBSP stream data!\n");
		*rbsp_len = 0;
		return false;
	}

	// throw away variable, could be set to 'nal_len' instead, but we are using the fact that 'rbsp_len' is already converted
	int nal_payload_len = *rbsp_len;
	// the cast here to (const uint8_t*) is valid because of the first 'if'
	int ret = nal_to_rbsp((const uint8_t*)nal, &nal_payload_len, out, rbsp_len);
	if (ret == -1)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to convert NALU stream into RBSP data!\n");
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

	s->profileIdc = profile_idc_to_h264_flag(sps.profile_idc); //TODO H.265
	s->depth_chroma = sps.bit_depth_chroma_minus8 + 8;
	s->depth_luma = sps.bit_depth_luma_minus8 + 8;

	int chroma_format_idc = 1; // resembles 4:2:0 subsampling
	// same if as in read_sps parsing function
	if (sps.profile_idc == 100 || sps.profile_idc == 110 ||
		sps.profile_idc == 122 || sps.profile_idc == 144)
	{
		chroma_format_idc = sps.chroma_format_idc;
	}
	//printf("profile_idc: %d, chroma_format_idc: %d\n", sps.profile_idc, chroma_format_idc);
	s->subsampling = isH264 ? h264_flag_to_subsampling((StdVideoH264ChromaFormatIdc)chroma_format_idc)
							: h265_flag_to_subsampling((StdVideoH265ChromaFormatIdc)chroma_format_idc);
	//printf("chroma: %d, luma: %d, subs: %d\n", s->depth_chroma, s->depth_luma, s->subsampling);

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
		if (nal_len <= 4 || (size_t)(nal_payload - nal) >= nal_len)
		{
			log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] NAL unit is too short.\n");
			return false;
		}

		size_t nal_payload_len = nal_len - (nal_payload - nal); //should be non-zero now

		int nalu_type = NALU_HDR_GET_TYPE(nal_payload[0], !isH264);
		if (isH264 &&
			nalu_type == NAL_H264_SPS)
		{
			if (!get_video_info_from_sps(s, nal_payload, nal_payload_len))
			{
				log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Found first SPS, but it was invalid! Discarting it.\n");
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

static void * begin_bitstream_writing(struct state_vulkan_decompress *s)
{
	void *memoryPtr = NULL;

	VkResult result = vkMapMemory(s->device, s->bitstreamBufferMemory, 0, VK_WHOLE_SIZE, 0, &memoryPtr);
	if (result != VK_SUCCESS) return NULL;

	memset(memoryPtr, 0, s->bitstreamBufferSize);

	return memoryPtr;
}

static VkDeviceSize write_bitstream(uint8_t *bitstream, VkDeviceSize bitstream_len, VkDeviceSize bitstream_capacity,
					    			const uint8_t *rbsp, int len, unsigned char nal_header, bool long_startcode)
{
	// writes one given NAL unit into NAL buffer, if error (not enough space in buffer) returns 0
	assert(bitstream != NULL && rbsp != NULL && len > 0);

	const uint8_t startcode_short[] = { BITSTREAM_STARTCODE };
	/*const uint8_t startcode_long[] = { 0, BITSTREAM_STARTCODE };

	const uint8_t *startcode = long_startcode ? startcode_long : startcode_short;
	size_t startcode_len = long_startcode ?
						   sizeof(startcode_long) / sizeof(startcode_long[0]) :
						   sizeof(startcode_short) / sizeof(startcode_short[0]);*/
	
	//DEBUG
	UNUSED(long_startcode);
	const uint8_t *startcode = startcode_short;
	size_t startcode_len = sizeof(startcode_short) / sizeof(startcode_short[0]);
	
	VkDeviceSize result_len = (VkDeviceSize)startcode_len + 1 + (VkDeviceSize)len; // startcode + header + rbsp data
	
	// check if enough space for rbsp in the buffer, return zero if not
	// (zero => error as we always want to write at least the startcode length anyway)
	if (bitstream_len + result_len > bitstream_capacity) return 0;

	memcpy(bitstream + bitstream_len, startcode, startcode_len);		// writing the nal start code
	bitstream[bitstream_len + startcode_len] = (uint8_t)nal_header;		// writing the nal header
	memcpy(bitstream + bitstream_len + startcode_len + 1, rbsp, len);	// writing the rbsp data

	return result_len;
}

static void end_bitstream_writing(struct state_vulkan_decompress *s)
{
	if (vkUnmapMemory == NULL || s->device == VK_NULL_HANDLE) return;

	vkUnmapMemory(s->device, s->bitstreamBufferMemory);
}

static bool detect_poc_wrapping(struct state_vulkan_decompress *s)
{
	// if there is already a frame in the output queue from the same GOP with the same POC nad POC wrap
	// as the currently decoded frame, than we return that poc wrapping occurred
	// not ideal system, but better than nothing - can fail for example when such frame was missed or already displayed

	for (size_t i = 0; i < s->outputFrameQueue_count; ++i)
	{
		frame_data_t queued = get_frame_from_output_queue(s, i);
		if (queued.idr_frame_seq == s->idr_frame_seq &&
			queued.poc == s->last_poc &&
			queued.poc_wrap == s->poc_wrap)
				return true;
	}

	return false;
}

static void copy_decoded_image(struct state_vulkan_decompress *s, VkImage srcDpbImage)
{
	assert(s->cmdBuffer != VK_NULL_HANDLE);
	assert(s->outputLumaPlane != VK_NULL_HANDLE);
	assert(s->outputLumaPlane != VK_NULL_HANDLE);
	assert(srcDpbImage != VK_NULL_HANDLE);

	VkImageCopy lumaRegion = { .srcSubresource = { .aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT,
												   .mipLevel = 0,
												   .baseArrayLayer = 0,
												   .layerCount = 1 },
							   .srcOffset = { 0, 0, 0 },
							   .dstSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
						   						   .mipLevel = 0,
						   						   .baseArrayLayer = 0,
						   						   .layerCount = 1 },
							   .dstOffset = { 0, 0, 0 },
							   .extent = { s->width, s->height, 1 } };
	VkImageCopy chromaRegion = { .srcSubresource = { .aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT,
												   	 .mipLevel = 0,
												   	 .baseArrayLayer = 0,
												   	 .layerCount = 1 },
								 .srcOffset = { 0, 0, 0 },
								 .dstSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
						   							 .mipLevel = 0,
						   							 .baseArrayLayer = 0,
						   							 .layerCount = 1 },
								 .dstOffset = { 0, 0, 0 },
								 .extent = { s->width / 2, s->height / 2, 1 } };
	
	vkCmdCopyImage(s->cmdBuffer, srcDpbImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				   s->outputLumaPlane, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &lumaRegion);
	vkCmdCopyImage(s->cmdBuffer, srcDpbImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				   s->outputChromaPlane, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &chromaRegion);
}

static bool write_decoded_frame(struct state_vulkan_decompress *s, frame_data_t *dst_frame)
{
	assert(s->device != VK_NULL_HANDLE);
	assert(s->outputImageMemory != VK_NULL_HANDLE);
	assert(dst_frame != NULL);

	VkDeviceSize lumaSize = s->lumaSize;		// using local variable, otherwise would get bad performance
	VkDeviceSize chromaSize = s->chromaSize;	// using local variable, otherwise would get bad performance
	VkDeviceSize size = lumaSize + 2 * chromaSize;
	assert((size_t)size == dst_frame->data_len);
	assert((size_t)size == s->outputFrameQueue_data_size);
	//printf("\t%d = %d + 2 * %d\n", size, lumaSize, chromaSize);

	assert(lumaSize <= s->outputChromaPlaneOffset);

	uint8_t *memory = NULL;
	VkResult result = vkMapMemory(s->device, s->outputImageMemory, 0, s->outputChromaPlaneOffset + chromaSize,
								  0, (void**)&memory);
	if (result != VK_SUCCESS) return false;
	assert(memory != NULL);

	uint8_t *dst_mem = s->outputFrameQueue_data + dst_frame->data_idx * s->outputFrameQueue_data_size;

	// Translating NV12 into I420:
	// luma plane
	const uint8_t *luma = memory;
	//for (size_t i = 0; i < lumaSize; ++i) dst_mem[i] = (unsigned char)luma[i]; // could be memcpy I guess
	memcpy(dst_mem, luma, lumaSize);

	// chroma plane
	const uint8_t *chroma = memory + s->outputChromaPlaneOffset;
	for (size_t i = 0; i < chromaSize; ++i)
	{
		unsigned char Cb = (unsigned char)chroma[2*i],
					  Cr = (unsigned char)chroma[2*i + 1];

		dst_mem[lumaSize + i] = Cb;
		dst_mem[lumaSize + chromaSize + i] = Cr;
	}
	
	vkUnmapMemory(s->device, s->outputImageMemory);

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
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to update vulkan video session parameters!\n");
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

static bool handle_sps_nalu(struct state_vulkan_decompress *s, uint8_t *rbsp, int rbsp_len)
{
	// reads sequence parameters set from given buffer range
	bool isH264 = s->codecOperation == VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
	assert(isH264); //TODO H.265

	assert(rbsp != NULL);
	assert(rbsp_len > 0);

	bs_t b = { 0 };
	sps_t sps = { 0 };

	bs_init(&b, rbsp, rbsp_len);
	
	read_sps(&sps, &b);
	b = (bs_t){ 0 }; // just to be sure
	//print_sps(&sps);

	assert(sps.pic_order_cnt_type == 0 || sps.pic_order_cnt_type == 2); //TODO handle other types as well

	int id = sps.seq_parameter_set_id;
	if (id < 0 || id >= MAX_SPS_IDS)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Id of read SPS is out of bounds (%d)! Discarting it.\n", id);
		return false;
	}

	assert(s->sps_array != NULL);
	s->sps_array[id] = sps;

	// potential err msg should get printed inside of update_video_session_params
	return update_video_session_params(s, isH264, &sps, NULL);
}

static bool handle_pps_nalu(struct state_vulkan_decompress *s, uint8_t *rbsp, int rbsp_len)
{
	// reads picture parameters set from given buffer range
	bool isH264 = s->codecOperation == VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
	assert(isH264); //TODO H.265

	assert(rbsp != NULL);
	assert(rbsp_len > 0);

	bs_t b = { 0 };
	pps_t pps = { 0 };

	bs_init(&b, rbsp, rbsp_len);

	read_pps(&pps, &b);
	b = (bs_t){ 0 }; // just to be sure
	//print_pps(&pps);

	int id = pps.pic_parameter_set_id;
	if (id < 0 || id >= MAX_PPS_IDS)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Id of read PPS is out of bounds (%d)! Discarting it.\n", id);
		return false;
	}

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

	assert(sh->pic_parameter_set_id < MAX_PPS_IDS);
	pps_t *pps = s->pps_array + sh->pic_parameter_set_id;
	int new_sps_id = pps->seq_parameter_set_id;

	assert(si->sps_id == -1 || si->sps_id == new_sps_id);
	si->sps_id = new_sps_id;

	assert(si->frame_num == -1 || si->frame_num == sh->frame_num);
	si->frame_num = sh->frame_num;

	assert(si->poc_lsb == -1 || si->poc_lsb == sh->pic_order_cnt_lsb);
	si->poc_lsb = sh->pic_order_cnt_lsb;
}

static bool handle_sh_nalu(struct state_vulkan_decompress *s, uint8_t *rbsp, int rbsp_len,
						   int nal_type, int nal_idc, slice_info_t *slice_info)
{
	// reads slice header from given buffer range
	assert(slice_info != NULL);
	assert(rbsp != NULL);
	
	bs_t b = { 0 };
	slice_header_t sh = { 0 };

	bs_init(&b, rbsp, rbsp_len);

	assert(s->sps_array != NULL);
	assert(s->pps_array != NULL);

	if (!read_slice_header(&sh, nal_type, nal_idc, s->pps_array, s->sps_array, &b))
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Encountered wrong SPS/PPS ids while reading a slice header! Discarting it.\n");
		return false;
	}
	//print_sh(&sh);
	fill_slice_info(s, slice_info, &sh);

	return true;
}

static VkDeviceSize handle_vcl(struct state_vulkan_decompress *s,
							   uint8_t *bitstream, VkDeviceSize bitstream_written, VkDeviceSize bitstream_capacity,
							   uint8_t *rbsp, int rbsp_len, unsigned char nal_header, slice_info_t *slice_info,
							   uint32_t slice_offsets[], uint32_t *slice_offsets_count, bool long_startcode)
{
	bool isH264 = s->codecOperation == VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
	assert(isH264); //TODO H.265
	int nal_type = NALU_HDR_GET_TYPE(nal_header, !isH264);
	int nal_idc = H264_NALU_HDR_GET_NRI(nal_header);

	bool sh_ret = handle_sh_nalu(s, rbsp, rbsp_len, nal_type, nal_idc, slice_info);
	UNUSED(sh_ret); // we want to write it into bitstream anyway

	VkDeviceSize written = write_bitstream(bitstream, bitstream_written, bitstream_capacity,
										   rbsp, rbsp_len, nal_header, long_startcode);
	if (written > 0 && *slice_offsets_count < MAX_SLICES)
	{
		slice_offsets[*slice_offsets_count] = bitstream_written; // pointing at the written startcode
		*slice_offsets_count += 1;

		slice_info->is_reference = nal_idc > 0;
		if (nal_idc > slice_info->nal_idc) slice_info->nal_idc = nal_idc;
	}
	else log_msg(LOG_LEVEL_WARNING, "[vulkan_decode] NAL writing fail!\n");

	return written;
}

static void decode_frame(struct state_vulkan_decompress *s, slice_info_t slice_info, VkDeviceSize bitstreamBufferWritten,
						 uint32_t slice_offsets[], uint32_t slice_offsets_count,
						 VkVideoReferenceSlotInfoKHR refSlotInfos[], uint32_t refSlotInfos_count,// uint32_t take_references,
						 VkVideoReferenceSlotInfoKHR *dstSlotInfo, VkVideoPictureResourceInfoKHR *dstVideoPicInfo)
{
	bool isH264 = s->codecOperation == VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
	VkDeviceSize bitstreamBufferWrittenAligned = (bitstreamBufferWritten + (s->bitstreambufferSizeAlignment - 1))
												 & ~(s->bitstreambufferSizeAlignment - 1); //alignment bit mask magic
	
	assert(bitstreamBufferWrittenAligned <= s->bitstreamBufferSize);
	assert(slice_offsets_count > 0);
	// for IDR pictures the id must be valid
	assert(!slice_info.is_intra || !slice_info.is_reference || slice_info.idr_pic_id >= 0);
	assert(isH264); //TODO H.265

	//assert(take_references >= 0);
	//assert(take_references <= refSlotInfos_count);

	// ---Filling infos related to bitstream---
	StdVideoDecodeH264PictureInfo h264DecodeStdInfo = { .flags = { .field_pic_flag = 0,
																   .is_intra = slice_info.is_intra,
																   .is_reference = slice_info.is_reference,
																	//IDEA maybe introduce new member to slice_info_t for it
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

	bool enable_queries = s->queryPool != VK_NULL_HANDLE;
	VkVideoInlineQueryInfoKHR queryInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_INLINE_QUERY_INFO_KHR,
											.pNext = isH264 ? (void*)&h264DecodeInfo : (void*)&h265DecodeInfo,
											.queryPool = s->queryPool,
											.firstQuery = 0,
											.queryCount = 1 };

	VkVideoDecodeInfoKHR decodeInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR,
										.pNext = enable_queries ? (void*)&queryInfo :
												 (isH264 ? (void*)&h264DecodeInfo : (void*)&h265DecodeInfo),
										.flags = 0,
										.srcBuffer = s->bitstreamBuffer,
										.srcBufferOffset = 0,
										.srcBufferRange = bitstreamBufferWrittenAligned,
										.dstPictureResource = *dstVideoPicInfo,
										// this must be the same as dstPictureResource when VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR
										// otherwise must not be the same as dstPictureResource
										.pSetupReferenceSlot = dstSlotInfo,
										//specifies the needed used references (but not the decoded frame)
										//TODO take_references
										.referenceSlotCount = refSlotInfos_count,
										.pReferenceSlots = refSlotInfos,
										};
	vkCmdDecodeVideoKHR(s->cmdBuffer, &decodeInfo);
}

static bool parse_and_decode(struct state_vulkan_decompress *s, unsigned char *src, unsigned int src_len,
							 int frame_seq, slice_info_t *slice_info)
{
	bool isH264 = s->codecOperation == VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;
	const VkExtent2D videoSize = { s->width, s->height };

	//int prev_idr_frame_seq = s->idr_frame_seq;
	
	uint32_t slice_offsets[MAX_SLICES] = { 0 };
	uint32_t slice_offsets_count = 0; 

	// ---Copying NAL units into s->bitstreamBuffer---
	// those flags are for debugging purposes - true, false are probably correct for normal usage
	const bool filter_nal = true, convert_to_rbsp = false;

	VkDeviceSize bitstream_written = 0;
	const VkDeviceSize bitstream_max_size = s->bitstreamBufferSize;
	assert(bitstream_max_size > 0);

	uint8_t *bitstream = (uint8_t*)begin_bitstream_writing(s);
	if (bitstream == NULL)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to map needed vulkan memory for NAL units!\n");
		return false;
	}
	{
		const unsigned char *nal = get_next_nal(src, src_len, true), *next_nal = NULL;
		size_t nal_len = 0;
		if (nal == NULL) log_msg(LOG_LEVEL_WARNING, "[vulkan_decode] First NAL is NULL!\n");

		while (nal != NULL)
		{
			const unsigned char *nal_payload = skip_nal_start_code(nal);
			if (nal_payload == NULL)
			{
				log_msg(LOG_LEVEL_WARNING, "[vulkan_decode] Encountered NAL unit that does not begin with a start code.\n");
				break;
			}

			next_nal = get_next_nal(nal_payload, src_len - (nal_payload - src), true);
			nal_len = next_nal == NULL ? src_len - (nal - src) : next_nal - nal;
			if (nal_len <= 4 || (size_t)(nal_payload - nal) >= nal_len)
			{
				log_msg(LOG_LEVEL_WARNING, "[vulkan_decode] Encountered too short NAL unit.\n");
				break;
			}
			
			size_t nal_startcode_len = nal_payload - nal;
			size_t nal_payload_len = nal_len - nal_startcode_len; //should be non-zero now
			bool long_startcode = nal_len - nal_payload_len > 3;

			unsigned char nalu_header = nal_payload[0];
			int nalu_type = NALU_HDR_GET_TYPE(nalu_header, !isH264);

			uint8_t *rbsp = NULL;
			int rbsp_len = 0;
			if (!create_rbsp(nal_payload, nal_payload_len, &rbsp, &rbsp_len))
			{
				//err should get printed inside of create_rbsp
				break;
			}
			assert(rbsp != NULL);
			assert(rbsp_len > 0);

			if (!filter_nal && nalu_type != NAL_H264_IDR && nalu_type != NAL_H264_NONIDR)
			{
				VkDeviceSize written = 0;
				if (convert_to_rbsp)
				{
					written = write_bitstream(bitstream, bitstream_written, bitstream_max_size,
											  rbsp, rbsp_len, nalu_header, long_startcode);
				}
				else //DEBUG version without conversion to rbsp:
				{
					written = write_bitstream(bitstream, bitstream_written, bitstream_max_size,
											  (uint8_t*)(nal_payload + 1), (int)(nal_payload_len - 1), nalu_header, long_startcode);
				}
				
				bitstream_written += written;
			}

			switch(nalu_type)
			{
				case NAL_H264_SEI:
					break; //switch break
				case NAL_H264_IDR:
					{
						s->prev_poc_lsb = 0;
						s->prev_poc_msb = 0;
						s->idr_frame_seq = frame_seq;
						s->poc_wrap = 0;
						clear_the_ref_slot_queue(s); // we dont need those references anymore
					}
					// intentional fallthrough
				case NAL_H264_NONIDR:
					{
						VkDeviceSize written = 0;
						if (convert_to_rbsp)
						{
							written = handle_vcl(s, bitstream, bitstream_written, bitstream_max_size,
								   				 rbsp, rbsp_len, nalu_header, slice_info,
								   				 slice_offsets, &slice_offsets_count, long_startcode);
						}
						else //DEBUG version without conversion to rbsp:
						{
							written = handle_vcl(s, bitstream, bitstream_written, bitstream_max_size,
												 (uint8_t*)(nal_payload + 1), (int)(nal_payload_len - 1), nalu_header, slice_info,
												 slice_offsets, &slice_offsets_count, long_startcode);
						}
						bitstream_written += written;
					}
					break; //switch break
				case NAL_H264_SPS:
					{
						bool sps_ret = handle_sps_nalu(s, rbsp, rbsp_len);
						UNUSED(sps_ret);
					}
					break; //switch break
				case NAL_H264_PPS:
					{
						bool pps_ret = handle_pps_nalu(s, rbsp, rbsp_len);
						UNUSED(pps_ret);
					}
					break; //switch break

				//TODO H.265 NAL unit types
				default:
					if (isH264) log_msg(LOG_LEVEL_DEBUG, "[vulkan_decode] Irrelevant NAL unit => Skipping it.\n");
					if (!isH264) log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] H265 is not implemented!\n");
					break; //switch break
			}

			destroy_rbsp(rbsp);
			nal = next_nal;
		}
	}
	end_bitstream_writing(s);
	bitstream = NULL; // just to be sure
 	assert(bitstream_written <= bitstream_max_size);

	if (slice_offsets_count == 0)
	{
		log_msg(LOG_LEVEL_WARNING, "[vulkan_decode] Got no video data to decompress!\n");
		return false;
	}

	//DEBUG
	//if (slice_info->is_intra && slice_info->is_reference)
	//		log_msg(LOG_LEVEL_DEBUG, "[vulkan_decode] Got IDR frame - %d\n", slice_info->idr_pic_id);
	//else log_msg(LOG_LEVEL_DEBUG, "[vulkan_decode] Got non-IDR frame.\n");

	assert(s->cmdBuffer != VK_NULL_HANDLE);
	assert(s->videoSession != VK_NULL_HANDLE);
	assert(s->videoSessionParams != VK_NULL_HANDLE);

	/*log_msg(LOG_LEVEL_DEBUG, "\tframe_seq: %d, frame_num: %d, poc_lsb: %d, pps_id: %d, is_reference: %d, is_intra: %d\n",
			slice_info->frame_seq, slice_info->frame_num, slice_info->poc_lsb, slice_info->pps_id,
			(int)(slice_info->is_reference), slice_info->is_intra);*/

	if (!begin_cmd_buffer(s))
	{
		// err msg should get printed inside of begin_cmd_buffer
		return false;
	}

	// ---Resetting queries if they are enabled---
	bool enable_queries = s->queryPool != VK_NULL_HANDLE;
	if (enable_queries)
	{
		vkCmdResetQueryPool(s->cmdBuffer, s->queryPool, 0, 1);
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

		// also transfer the output image planes into defined layout
		assert(s->outputLumaPlane != VK_NULL_HANDLE && s->outputChromaPlane != VK_NULL_HANDLE);
		transfer_image_layout(s->cmdBuffer, s->outputLumaPlane,
							  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		transfer_image_layout(s->cmdBuffer, s->outputChromaPlane,
							  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

		s->dpbHasDefinedLayout = true;
	}
	else							// otherwise we also transfer them, but from transfer src optimal layout
	{
		for (size_t i = 0; i < MAX_REF_FRAMES + 1; ++i)
		{
			transfer_image_layout(s->cmdBuffer, s->dpb[i],
								  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
		}

		// also transfer the output image planes into tranfer dst layout
		transfer_image_layout(s->cmdBuffer, s->outputLumaPlane,
							  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		transfer_image_layout(s->cmdBuffer, s->outputChromaPlane,
							  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	}

	// ---Video coding scope---
	{
		assert(slice_info->frame_num >= 0 && slice_info->poc_lsb >= 0 &&
		   	   slice_info->pps_id >= 0 && slice_info->sps_id >= 0);
		assert(slice_info->sps_id < MAX_SPS_IDS);

		sps_t *sps = s->sps_array + slice_info->sps_id;
		
		// Filling the references infos
		uint32_t slotInfos_ref_count = 0;
		StdVideoDecodeH264ReferenceInfo h264StdInfos[MAX_REF_FRAMES + 1] = { 0 };
		VkVideoDecodeH264DpbSlotInfoKHR h264SlotInfos[MAX_REF_FRAMES + 1] = { 0 };
		VkVideoPictureResourceInfoKHR picInfos[MAX_REF_FRAMES + 1] = { 0 };
		VkVideoReferenceSlotInfoKHR slotInfos[MAX_REF_FRAMES + 1] = { 0 };

		fill_ref_picture_infos(s, slotInfos, picInfos, h264SlotInfos, h264StdInfos, MAX_REF_FRAMES, sps->num_ref_frames,
							   isH264, &slotInfos_ref_count);
		assert(slotInfos_ref_count <= MAX_REF_FRAMES);

		// Filling the decoded frame info
		slice_info->poc = get_picture_order_count(sps, slice_info->poc_lsb, slice_info->frame_num, slice_info->is_reference,
												  &s->prev_poc_msb, &s->prev_poc_lsb);

		h264StdInfos[slotInfos_ref_count] = (StdVideoDecodeH264ReferenceInfo){ .flags = { 0 },
																		   	   .FrameNum = slice_info->frame_num,
																		   	   .PicOrderCnt = { slice_info->poc, slice_info->poc } };
		h264SlotInfos[slotInfos_ref_count] = (VkVideoDecodeH264DpbSlotInfoKHR){ .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,
																				.pStdReferenceInfo = h264StdInfos + slotInfos_ref_count };
		picInfos[slotInfos_ref_count] = (VkVideoPictureResourceInfoKHR){ .sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
																		 .codedOffset = { 0, 0 }, // empty offset
																		 .codedExtent = videoSize,
																		 .baseArrayLayer = 0,
																		 .imageViewBinding = s->dpbViews[slice_info->dpbIndex] };
		VkVideoDecodeH265DpbSlotInfoKHR h265SlotInfo = {}; //TODO H.265
		slotInfos[slotInfos_ref_count] = (VkVideoReferenceSlotInfoKHR){ .sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
																		.pNext = isH264 ? (void*)(h264SlotInfos + slotInfos_ref_count)
																						: (void*)&h265SlotInfo,
																		.slotIndex = -1, // currently decoded picture must have index -1
																		.pPictureResource = picInfos + slotInfos_ref_count };

		begin_video_coding_scope(s, slotInfos, slotInfos_ref_count + 1);
		slotInfos[slotInfos_ref_count].slotIndex = slice_info->dpbIndex;
		decode_frame(s, *slice_info, bitstream_written, slice_offsets, slice_offsets_count,
					 slotInfos, slotInfos_ref_count,
					 slotInfos + slotInfos_ref_count, picInfos + slotInfos_ref_count);
		end_video_coding_scope(s);

	}
	
	// ---VkImage synchronization and layout transfering after video coding scope---
	for (size_t i = 0; i < MAX_REF_FRAMES + 1; ++i)
	{
		transfer_image_layout(s->cmdBuffer, s->dpb[i], //VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR
							  VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	}
	
	// ---Copying decoded DPB image into output image---
	assert(s->dpbFormat == VK_FORMAT_G8_B8R8_2PLANE_420_UNORM); //TODO handle other potential formats too
	copy_decoded_image(s, s->dpb[slice_info->dpbIndex]);
	transfer_image_layout(s->cmdBuffer, s->outputLumaPlane,
						  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	transfer_image_layout(s->cmdBuffer, s->outputChromaPlane,
						  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

	if (!end_cmd_buffer(s))
	{
		// relevant err msg printed inside of end_cmd_buffer
		return false;
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
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to submit the decode cmd buffer into queue!\n");
		return false;
	}

	// ---Reference queue management---
	// can be done before synchronization as we only work with slice_infos
	if (slice_info->is_reference) // however insert only if the decoded frame actually is a reference
	{
		insert_ref_slot_into_queue(s, *slice_info);
	}

	// ---Synchronization---
	const uint64_t synchronizationTimeout = 500 * 1000 * 1000; // = 500ms (timeout is in nanoseconds)

	result = vkWaitForFences(s->device, 1, &s->fence, VK_TRUE, synchronizationTimeout);
	if (result != VK_SUCCESS)
	{
		if (result == VK_TIMEOUT) log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Vulkan can't synchronize! -> Timeout reached.\n");
		else if (result == VK_ERROR_DEVICE_LOST) log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Vulkan can't synchronize! -> Device lost.\n");
		else log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Vulkan can't synchronize! -> Not enough memory.\n");
		
		return false;
	}

	result = vkResetFences(s->device, 1, &s->fence);
	if (result != VK_SUCCESS)
	{
		// should happen only when out of memory
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to reset vulkan fence!\n");
		return false;
	}

	// ---Getting potential query results---
	if (enable_queries)
	{
		int32_t queryResult[1] = { 0 };
		size_t queryResult_size = sizeof(queryResult);

		result = vkGetQueryPoolResults(s->device, s->queryPool, 0, 1,
									   queryResult_size, (void*)queryResult, queryResult_size,
									   VK_QUERY_RESULT_WITH_STATUS_BIT_KHR);
		if (result != VK_SUCCESS)
		{
			log_msg(LOG_LEVEL_WARNING, "[vulkan_decode] Get query results returned error: %d!\n", result);
		}
		else
		{
			VkQueryResultStatusKHR *status = (VkQueryResultStatusKHR*)queryResult;
			log_msg(LOG_LEVEL_DEBUG, "[vulkan_decode] Decode query result: %d.\n", *status);
		}
	}

	// ---Try to detect if POC wrapping occurred---
	s->last_poc = slice_info->poc;
	if (detect_poc_wrapping(s))
	{
		++(s->poc_wrap);
	}

	// ---Make new slot in the output queue for newly decoded frame---
	frame_data_t *decoded_frame = make_new_entry_in_output_queue(s, *slice_info);

	// ---Writing the newly decoded frame data into output queue slot---
	if (!write_decoded_frame(s, decoded_frame))
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to write the decoded frame into the destination buffer!\n");
		return false;
	}

	return true;
}

static decompress_status vulkan_decompress(void *state, unsigned char *dst, unsigned char *src,
                unsigned int src_len, int frame_seq, struct video_frame_callbacks *callbacks,
                struct pixfmt_desc *internal_prop)
{
	UNUSED(callbacks);
	struct state_vulkan_decompress *s = (struct state_vulkan_decompress *)state;
    //log_msg(LOG_LEVEL_DEBUG, "[vulkan_decode] Decompress - dst: %p, src: %p, src_len: %u, frame_seq: %d\n",
	//						 dst, src, src_len, frame_seq);
	
    decompress_status res = DECODER_NO_FRAME;

	if (s->out_codec == VIDEO_CODEC_NONE)
	{
		log_msg(LOG_LEVEL_NOTICE, "[vulkan_decode] Probing...\n");

		*internal_prop = get_pixfmt_desc(I420);
		return DECODER_GOT_CODEC;
	}

	if (src_len < 5)
	{
		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Source buffer is too short!\n");
		return res;
	}

	assert(s->codecOperation != VK_VIDEO_CODEC_OPERATION_NONE_KHR);

	if (!s->sps_vps_found && !find_first_sps_vps(s, src, src_len))
	{
		log_msg(LOG_LEVEL_NOTICE, "[vulkan_decode] Still no SPS or VPS found.\n");
		return res;
	}
	s->sps_vps_found = true;

	bool wrong_pixfmt = false;
	if (!s->prepared && !prepare(s, &wrong_pixfmt))
	{
		if (wrong_pixfmt)
		{
			log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to prepare for decompress - wrong pixel format.\n");
			return DECODER_UNSUPP_PIXFMT;
		}

		log_msg(LOG_LEVEL_ERROR, "[vulkan_decode] Failed to prepare for decompress.\n");
		return res;
	}
	s->prepared = true;

	int prev_frame_seq = s->current_frame_seq;
	s->current_frame_seq = frame_seq;
	if (frame_seq > prev_frame_seq + 1)
	{
		log_msg(LOG_LEVEL_DEBUG, "[vulkan_decode] Missed frame.\n");
		clear_the_ref_slot_queue(s);
	}

	// gets filled in parse_and_decode
	slice_info_t slice_info = { .is_reference = false, .is_intra = false, .nal_idc = 0,
								.idr_pic_id = -1, .sps_id = -1, .pps_id = -1,
								.frame_num = -1, .frame_seq = frame_seq, .poc_lsb = -1,
								.dpbIndex = smallest_dpb_index_not_in_queue(s) };

	// potential err msg gets printed inside of parse_and_decode
	bool ret_decode = parse_and_decode(s, src, src_len, frame_seq, &slice_info);
	UNUSED(ret_decode);

	// ---Getting next frame to be displayed (in display order) from output queue---
	const frame_data_t *display_frame = output_queue_get_next_frame(s);
	//const frame_data_t *display_frame = output_queue_pop_first(s);
	if (display_frame == NULL) return res;// no frame to display
	else res = DECODER_GOT_FRAME;

	// ---Copying data of display frame into dst buffer---
	uint8_t *display_data = s->outputFrameQueue_data + display_frame->data_idx * s->outputFrameQueue_data_size;
	memcpy(dst, display_data, display_frame->data_len);

	s->last_displayed_frame_seq = display_frame->frame_seq;
	s->last_displayed_poc = display_frame->poc;

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
