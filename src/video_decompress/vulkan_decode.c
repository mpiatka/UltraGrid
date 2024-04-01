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

// activates vulkan validation layers if defined
// if defined your vulkan loader needs to know where to find the validation layer manifest
// (for example through VK_LAYER_PATH or VK_ADD_LAYER_PATH env. variables)
#define VULKAN_VALIDATE

// one of value from enum VkDebugUtilsMessageSeverityFlagBitsEXT from vulkan.h (vulkan_core.h)
#define VULKAN_VALIDATE_SHOW_SEVERITY VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT

static PFN_vkCreateInstance vkCreateInstance = NULL;
static PFN_vkDestroyInstance vkDestroyInstance = NULL;
#ifdef VULKAN_VALIDATE
static PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT = NULL;
static PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT = NULL;
#endif
static PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = NULL;
static PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = NULL;
static PFN_vkCreateDevice vkCreateDevice = NULL;
static PFN_vkDestroyDevice vkDestroyDevice = NULL;
static PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties = NULL;
static PFN_vkEnumerateInstanceLayerProperties vkEnumerateInstanceLayerProperties = NULL;
static PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties = NULL;
static PFN_vkGetDeviceQueue vkGetDeviceQueue = NULL;
static PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR vkGetPhysicalDeviceVideoCapabilitiesKHR = NULL;
static PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR vkGetPhysicalDeviceVideoFormatPropertiesKHR = NULL;
static PFN_vkCreateVideoSessionKHR vkCreateVideoSessionKHR = NULL;
static PFN_vkDestroyVideoSessionKHR vkDestroyVideoSessionKHR = NULL;

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
	
	vkCreateDevice = (PFN_vkCreateDevice)loader(instance, "vkCreateDevice");
	vkDestroyDevice = (PFN_vkDestroyDevice)loader(instance, "vkDestroyDevice");
	vkEnumerateDeviceExtensionProperties = (PFN_vkEnumerateDeviceExtensionProperties)
												loader(instance, "vkEnumerateDeviceExtensionProperties");
	vkGetDeviceQueue = (PFN_vkGetDeviceQueue)loader(instance, "vkGetDeviceQueue");
	vkGetPhysicalDeviceVideoCapabilitiesKHR = (PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR)
												loader(instance, "vkGetPhysicalDeviceVideoCapabilitiesKHR");
	vkGetPhysicalDeviceVideoFormatPropertiesKHR = (PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR)
												loader(instance, "vkGetPhysicalDeviceVideoFormatPropertiesKHR");
	vkCreateVideoSessionKHR = (PFN_vkCreateVideoSessionKHR)loader(instance, "vkCreateVideoSessionKHR");
	vkDestroyVideoSessionKHR = (PFN_vkDestroyVideoSessionKHR)loader(instance, "vkDestroyVideoSessionKHR");

	return vkDestroyInstance &&
	#ifdef VULKAN_VALIDATE
		   vkCreateDebugUtilsMessengerEXT && vkDestroyDebugUtilsMessengerEXT &&
	#endif
		   vkEnumeratePhysicalDevices && vkGetPhysicalDeviceProperties &&
		   vkCreateDevice && vkDestroyDevice &&
		   vkEnumerateDeviceExtensionProperties &&
		   vkGetDeviceQueue && vkGetPhysicalDeviceVideoCapabilitiesKHR &&
		   vkGetPhysicalDeviceVideoFormatPropertiesKHR &&
		   vkCreateVideoSessionKHR && vkDestroyVideoSessionKHR;
}

struct state_vulkan_decompress
{
	HMODULE vulkanLib;							// needs to be destroyed if valid
	VkInstance instance; 						// needs to be destroyed if valid
	PFN_vkGetInstanceProcAddr loader;
	//maybe this could be present only when VULKAN_VALIDATE is defined?
	VkDebugUtilsMessengerEXT debugMessenger;	// needs to be destroyed if valid
	VkPhysicalDevice physicalDevice;
	VkDevice device;							// needs to be destroyed if valid
	uint32_t queueFamilyIdx;
	VkQueue decode_queue; //USELESS maybe we need only the index?
	VkVideoCodecOperationFlagsKHR queueVideoFlags;
	VkVideoCodecOperationFlagsKHR codecOperation;
	bool prepared;
	VkVideoSessionKHR videoSession;				// needs to be destroyed if valid

	// UltraGrid
	//video_desc desc;
	int width, height;
	int rshift, gshift, bshift;
	int pitch;
	codec_t out_codec;
};

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

	// ---Allocation of the vulkan_decompress state---
	struct state_vulkan_decompress *s = calloc(1, sizeof(struct state_vulkan_decompress));
	if (!s)
	{
		printf("[vulkan_decode] Couldn't allocate memory for state struct!\n");
		return NULL;
	}

	// ---Dynamic loading of the vulkan loader library---
	const char vulkan_lib_filename[] = "vulkan-1.dll"; //TODO .so
    s->vulkanLib = LoadLibrary(vulkan_lib_filename);
	if (s->vulkanLib == NULL)
	{
		printf("[vulkan_decode] Vulkan loader file '%s' not found!\n", vulkan_lib_filename);
		free(s);
		return NULL;
	}

	// ---Getting the loader function---
	const char vulkan_proc_name[] = "vkGetInstanceProcAddr";
	PFN_vkGetInstanceProcAddr getInstanceProcAddr = (PFN_vkGetInstanceProcAddr)GetProcAddress(s->vulkanLib, vulkan_proc_name);
    if (getInstanceProcAddr == NULL) {

		printf("[vulkan_decode] Vulkan function '%s' not found!\n", vulkan_proc_name);
        FreeLibrary(s->vulkanLib);
		free(s);
        return NULL;
    }
	s->loader = getInstanceProcAddr;

	// ---Loading function pointers where the instance is not needed---
	if (!load_vulkan_functions_globals(getInstanceProcAddr))
	{
		printf("[vulkan_decode] Failed to load all vulkan functions!\n");
		FreeLibrary(s->vulkanLib);
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
						NULL };
	
	if (!check_for_instance_extensions(requiredInstanceExtensions))
	{
		//error msg should be printed inside of check_for_extensions
		FreeLibrary(s->vulkanLib);
		free(s);
        return NULL;
	}

	// ---Creating the vulkan instance---
	//TODO InitDebugReport form NVidia example
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
		free(s);
		return NULL;
	}

	if (!load_vulkan_functions_with_instance(getInstanceProcAddr, s->instance))
	{
		printf("[vulkan_decode] Failed to load all instance related vulkan functions!\n");
		if (vkDestroyInstance != NULL) vkDestroyInstance(s->instance, NULL);
		FreeLibrary(s->vulkanLib);
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
		free(s);
		return NULL;
	}
	#endif

	// ---Choosing of physical device---
	VkQueueFlags requestedFamilyQueueFlags = VK_QUEUE_VIDEO_DECODE_BIT_KHR | VK_QUEUE_TRANSFER_BIT;
	const char* const requiredDeviceExtensions[] = {
	#if defined(__linux) || defined(__linux__) || defined(linux)
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME,
	#endif
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
        VK_KHR_VIDEO_QUEUE_EXTENSION_NAME,
        VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME,
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
		free(s);
		return NULL;
	}

	s->queueVideoFlags = chosen_queue_video_props.videoCodecOperations;
	assert(chosen_queue_family.pNext == NULL && chosen_queue_video_props.pNext == NULL);
	assert(s->queueVideoFlags != 0);

	VkPhysicalDeviceProperties deviceProperties;
	vkGetPhysicalDeviceProperties(s->physicalDevice, &deviceProperties);
	printf("Chosen physical device is: '%s' and chosen queue family index is: %d\n", 
				deviceProperties.deviceName, s->queueFamilyIdx);

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
		free(s);
		return NULL;
	}

	vkGetDeviceQueue(s->device, s->queueFamilyIdx, 0, &s->decode_queue);

	s->videoSession = VK_NULL_HANDLE; //video session gets created in prepare function

	printf("[vulkan_decode] Initialization finished successfully.\n");
	return s;
}

static void vulkan_decompress_done(void *state)
{
	printf("vulkan_decode - done\n");
	struct state_vulkan_decompress *s = (struct state_vulkan_decompress *)state;
	if (!s) return;

	if (vkDestroyVideoSessionKHR != NULL && s->device != VK_NULL_HANDLE)
			vkDestroyVideoSessionKHR(s->device, s->videoSession, NULL);

	if (vkDestroyDevice != NULL) vkDestroyDevice(s->device, NULL);

	destroy_debug_messenger(s->instance, s->debugMessenger);

	if (vkDestroyInstance != NULL) vkDestroyInstance(s->instance, NULL);

	FreeLibrary(s->vulkanLib);

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
									codec_t codec, VkFormat *pictureFormat)
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
			*pictureFormat = format;
			
			free(properties);
			return true;
		}
	}

	printf("[vulkan_decode] Wanted output video format is not supported!\n");
	free(properties);
	return false;
}

static bool prepare(struct state_vulkan_decompress *s)
{
	printf("vulkan_decode - prepare\n");
	assert(!s->prepared); //this function should be called only when decompress is not prepared

	struct pixfmt_desc pf_desc = get_pixfmt_desc(s->out_codec);
	printf("\tpf_desc - depth: %d, subsampling: %d, rgb: %d, accel_type: %d\n",
				pf_desc.depth, pf_desc.subsampling, pf_desc.rgb, pf_desc.accel_type);
	
	if (!pf_desc.depth) //pixel format description is invalid
	{
		printf("[vulkan_decode] Got invalid pixel format!\n");
		return false;
	}

	VkVideoChromaSubsamplingFlagsKHR chromaSubsampling = subsampling_to_vulkan_flag(pf_desc.subsampling);
	if (chromaSubsampling == VK_VIDEO_CHROMA_SUBSAMPLING_INVALID_KHR)
	{
		printf("[vulkan_decode] Got unsupported subsampling!\n");
		return false; //TODO maybe return DECODER_UNSUPP_PIXFMT?
	}

	VkVideoComponentBitDepthFlagBitsKHR vulkanDepth = depth_to_vulkan_flag(pf_desc.depth);
	if (vulkanDepth == VK_VIDEO_COMPONENT_BIT_DEPTH_INVALID_KHR)
	{
		printf("[vulkan_decode] Got unsupported color channel depth!\n");
		return false; //TODO maybe return DECODER_UNSUPP_PIXFMT?
	}

	assert(s->codecOperation != VK_VIDEO_CODEC_OPERATION_NONE_KHR);
	bool isH264 = s->codecOperation == VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR;

	//TODO interlacing and stdProfileIdc in h264/h265
	VkVideoDecodeH264ProfileInfoKHR h264Profile = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_KHR };
	VkVideoDecodeH265ProfileInfoKHR h265Profile = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PROFILE_INFO_KHR };
	VkVideoProfileInfoKHR videoProfile = { .sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
										   .pNext = isH264 ? (void*)&h264Profile : (void*)&h265Profile,
										   .videoCodecOperation = s->codecOperation,
										   .chromaSubsampling = chromaSubsampling,
										   .lumaBitDepth = vulkanDepth,
										   .chromaBitDepth = vulkanDepth };
	
	VkVideoDecodeH264CapabilitiesKHR h264Capabilites = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_CAPABILITIES_KHR };
	VkVideoDecodeH265CapabilitiesKHR h265Capabilites = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_CAPABILITIES_KHR };
	VkVideoDecodeCapabilitiesKHR decodeCapabilities = { .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR,
														.pNext = isH264 ? (void*)&h264Capabilites : (void*)&h265Capabilites };
	VkVideoCapabilitiesKHR videoCapabilities = { .sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR,
												 .pNext = (void*)&decodeCapabilities };
	VkResult result = vkGetPhysicalDeviceVideoCapabilitiesKHR(s->physicalDevice, &videoProfile, &videoCapabilities);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to get physical device video capabilities!\n");
		if (result == VK_ERROR_OUT_OF_HOST_MEMORY) puts("\t- Host out of memory.");
		else if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY) puts("\t- Device out of memory.");
		else if (result == VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR) puts("\t- Video picture layout not supported.");
		else if (result == VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR) puts("\t- Video operation not supported.");
		else if (result == VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR) puts("\t- Video format not supported.");
		else if (result == VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR) puts("\t- Video codec not supported.");
		else printf("\t- Vulkan error: %d\n", result);
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
	VkFormat pictureFormat = VK_FORMAT_UNDEFINED, referencePictureFormat = VK_FORMAT_UNDEFINED;

	VkVideoProfileListInfoKHR videoProfileList = { .sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR,
												   .profileCount = 1,
												   .pProfiles = &videoProfile };
	VkPhysicalDeviceVideoFormatInfoKHR videoFormatInfo = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_FORMAT_INFO_KHR,
														  .pNext = (void*)&videoProfileList };

	assert(s->physicalDevice != VK_NULL_HANDLE);

	if (decodeCapabilitiesFlags & VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR)
	{
		videoFormatInfo.imageUsage = VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR | VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR;
		// err msg should be printed inside of check_for_vulkan_format function 
		if (!check_for_vulkan_format(s->physicalDevice, videoFormatInfo, s->out_codec, &referencePictureFormat)) return false;

		pictureFormat = referencePictureFormat;
	}
	else if (decodeCapabilitiesFlags & VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_DISTINCT_BIT_KHR)
	{
		
		videoFormatInfo.imageUsage = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR;
		// err msg should be printed inside of check_for_vulkan_format function 
		if (!check_for_vulkan_format(s->physicalDevice, videoFormatInfo, s->out_codec, &referencePictureFormat)) return false;
		videoFormatInfo.imageUsage = VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR;
		if (!check_for_vulkan_format(s->physicalDevice, videoFormatInfo, s->out_codec, &pictureFormat)) return false;
	}
	else
	{
		printf("[vulkan_decode] Unsupported decodeCapabilitiesFlags value (%d)!\n", decodeCapabilitiesFlags);
		return false;
	}

	assert(pictureFormat != VK_FORMAT_UNDEFINED);
	assert(referencePictureFormat != VK_FORMAT_UNDEFINED);
	assert(s->device != VK_NULL_HANDLE);
	assert(s->videoSession == VK_NULL_HANDLE);

	VkVideoSessionCreateInfoKHR sessionInfo = { .sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR,
												.pNext = NULL,
												.queueFamilyIndex = s->queueFamilyIdx,
												.flags = 0,
												.pVideoProfile = &videoProfile,
												.pictureFormat = pictureFormat,
												.maxCodedExtent = (VkExtent2D){ s->width, s->height },
												.referencePictureFormat = referencePictureFormat,
												.maxDpbSlots = videoCapabilities.maxDpbSlots,
												.maxActiveReferencePictures = videoCapabilities.maxActiveReferencePictures,
												//TODO version might be wrong
												.pStdHeaderVersion = &videoCapabilities.stdHeaderVersion };
	result = vkCreateVideoSessionKHR(s->device, &sessionInfo, NULL, &s->videoSession);
	if (result != VK_SUCCESS)
	{
		printf("[vulkan_decode] Failed to create vulkan video session!\n");
		return false;
	}

	printf("Preparation successful.\n");
	return true;
}

static decompress_status vulkan_decompress(void *state, unsigned char *dst, unsigned char *src,
                unsigned int src_len, int frame_seq, struct video_frame_callbacks *callbacks,
                struct pixfmt_desc *internal_prop)
{
	printf("vulkan_decode - decompress\n");
	UNUSED(callbacks);
	UNUSED(internal_prop);
	struct state_vulkan_decompress *s = (struct state_vulkan_decompress *)state;
    //printf("\tdst: %p, src: %p, src_len: %u, frame_seq: %d\n",
	//		dst, src, src_len, frame_seq);
	
    decompress_status res = DECODER_NO_FRAME;

	if (s->out_codec == VIDEO_CODEC_NONE)
	{
		printf("\tProbing...\n");
		codec_t pf = vulkan_flag_to_codec(s->codecOperation);
		if (pf == VIDEO_CODEC_NONE)
		{
			printf("[vulkan_decode] Unsupported codec operation!\n");
			return res;
		}
		
		struct pixfmt_desc pf_desc = get_pixfmt_desc(pf);

		//TODO
		if (pf_desc.rgb)
		{
			printf("[vulkan_decode] RGB is not supported!\n");
			return DECODER_UNSUPP_PIXFMT;
		}

		*internal_prop = pf_desc;
		return DECODER_GOT_CODEC;
	}

	if (!s->prepared)
	{
		if (!(s->prepared = prepare(s)))
		{
			printf("\tFailed to prepare for decompress.\n");
			return res;
		}
	}

	//TODO decompress
	UNUSED(src);
	UNUSED(dst);
	UNUSED(src_len);
	UNUSED(frame_seq);

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
