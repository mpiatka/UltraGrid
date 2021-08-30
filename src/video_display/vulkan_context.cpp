#include "vulkan_context.h"
#include <cassert>
#include <iostream>

using namespace vulkan_display_detail;

#ifdef NO_EXCEPTIONS
std::string vulkan_display_error_message{};
#endif // NO_EXCEPTIONS

namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        [[maybe_unused]] VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        [[maybe_unused]] void* pUserData)
{
        std::cout << "validation layer: " << pCallbackData->pMessage << '\n' << std::endl;
        return VK_FALSE;
}

RETURN_TYPE check_validation_layers(const std::vector<c_str>& required_layers) {
        std::vector<vk::LayerProperties>  layers;
        CHECKED_ASSIGN(layers, vk::enumerateInstanceLayerProperties());
        //for (auto& l : layers) puts(l.layerName);

        for (const auto& req_layer : required_layers) {
                auto layer_equals = [req_layer](auto layer) { return strcmp(req_layer, layer.layerName) == 0; };
                bool found = std::any_of(layers.begin(), layers.end(), layer_equals);
                CHECK(found, "Layer "s + req_layer + " is not supported.");
        }
        return RETURN_TYPE();
}

RETURN_TYPE check_instance_extensions(const std::vector<c_str>& required_extensions) {
        std::vector<vk::ExtensionProperties> extensions;
        CHECKED_ASSIGN(extensions, vk::enumerateInstanceExtensionProperties(nullptr));

        for (const auto& req_exten : required_extensions) {
                auto extension_equals = [req_exten](auto exten) { return strcmp(req_exten, exten.extensionName) == 0; };
                bool found = std::any_of(extensions.begin(), extensions.end(), extension_equals);
                CHECK(found, "Instance extension "s + req_exten + " is not supported.");
        }
        return RETURN_TYPE();
}


RETURN_TYPE check_device_extensions(bool& result, bool propagate_error,
        const std::vector<c_str>& required_extensions, const vk::PhysicalDevice& device)
{
        std::vector<vk::ExtensionProperties> extensions;
        CHECKED_ASSIGN(extensions, device.enumerateDeviceExtensionProperties(nullptr));

        for (const auto& req_exten : required_extensions) {
                auto extension_equals = [req_exten](auto exten) { return strcmp(req_exten, exten.extensionName) == 0; };
                bool found = std::any_of(extensions.begin(), extensions.end(), extension_equals);
                if (!found) {
                        result = false;
                        if (propagate_error) {
                                CHECK(false, "Device extension "s + req_exten + " is not supported.");
                        }
                        return RETURN_TYPE();
                }
        }
        result = true;
        return RETURN_TYPE();
}

RETURN_TYPE get_queue_family_index(uint32_t& index, vk::PhysicalDevice gpu, vk::SurfaceKHR surface) {
        assert(gpu);

        std::vector<vk::QueueFamilyProperties> families = gpu.getQueueFamilyProperties();

        index = NO_QUEUE_FAMILY_INDEX_FOUND;
        for (uint32_t i = 0; i < families.size(); i++) {
                VkBool32 surface_supported = true;
                if (surface) {
                        CHECKED_ASSIGN(surface_supported, gpu.getSurfaceSupportKHR(i, surface));
                }

                if (surface_supported && (families[i].queueFlags & vk::QueueFlagBits::eGraphics)) {
                        index = i;
                        break;
                }
        }
        return RETURN_TYPE();
}

const std::vector required_gpu_extensions = { "VK_KHR_swapchain" };

RETURN_TYPE is_gpu_suitable(bool& result, bool propagate_error, vk::PhysicalDevice gpu, vk::SurfaceKHR surface = nullptr) {
        PASS_RESULT(check_device_extensions(result, propagate_error, required_gpu_extensions, gpu));
        if (!result) {
                return RETURN_TYPE();
        }
        uint32_t index = NO_QUEUE_FAMILY_INDEX_FOUND;
        PASS_RESULT(get_queue_family_index(index, gpu, surface));
        if (index == NO_QUEUE_FAMILY_INDEX_FOUND) {
                result = false;
        }
        return RETURN_TYPE();
}

RETURN_TYPE choose_suitable_GPU(vk::PhysicalDevice& suitable_gpu, const std::vector<vk::PhysicalDevice>& gpus, vk::SurfaceKHR surface) {
        assert(surface);
        bool is_suitable = false;
        for (const auto& gpu : gpus) {
                auto properties = gpu.getProperties();
                if (properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
                        PASS_RESULT(is_gpu_suitable(is_suitable, false, gpu, surface));
                        if (is_suitable) {
                                suitable_gpu = gpu;
                                return RETURN_TYPE();
                        }
                }
        }

        for (const auto& gpu : gpus) {
                auto properties = gpu.getProperties();
                if (properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) {
                        PASS_RESULT(is_gpu_suitable(is_suitable, false, gpu, surface));
                        if (is_suitable) {
                                suitable_gpu = gpu;
                                return RETURN_TYPE();
                        }
                }
        }

        for (const auto& gpu : gpus) {
                PASS_RESULT(is_gpu_suitable(is_suitable, false, gpu, surface));
                if (is_suitable) {
                        suitable_gpu = gpu;
                        return RETURN_TYPE();
                }
        }

        CHECK(false, "No suitable gpu found.");
        return RETURN_TYPE();
}

RETURN_TYPE choose_gpu_by_index(vk::PhysicalDevice& gpu, std::vector<vk::PhysicalDevice>& gpus, uint32_t gpu_index) {
        CHECK(gpu_index < gpus.size(), "GPU index is not valid.");
        std::vector<std::pair<std::string, vk::PhysicalDevice>> gpu_names;
        gpu_names.reserve(gpus.size());

        auto get_gpu_name = [](auto gpu) -> std::pair<std::string, vk::PhysicalDevice> {
                return { gpu.getProperties().deviceName, gpu };
        };

        std::transform(gpus.begin(), gpus.end(), std::back_inserter(gpu_names), get_gpu_name);

        std::sort(gpu_names.begin(), gpu_names.end());
        gpu = gpu_names[gpu_index].second;
        return RETURN_TYPE();
}

constexpr vk::CompositeAlphaFlagBitsKHR get_composite_alpha(vk::CompositeAlphaFlagsKHR capabilities) {
        uint32_t result = 1;
        while (!(result & static_cast<uint32_t>(capabilities))) {
                result <<= 1u;
        }
        return static_cast<vk::CompositeAlphaFlagBitsKHR>(result);
}

} //namespace ------------------------------------------------------------------------


namespace vulkan_display {

RETURN_TYPE vulkan_instance::init(std::vector<c_str>& required_extensions, bool enable_validation) {
        std::vector<c_str> validation_layers{};
        if (enable_validation) {
                validation_layers.push_back("VK_LAYER_KHRONOS_validation");
                PASS_RESULT(check_validation_layers(validation_layers));

                const char* debug_extension = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
                required_extensions.push_back(debug_extension);
        }

        PASS_RESULT(check_instance_extensions(required_extensions));

        vk::ApplicationInfo app_info{};
        app_info.setApiVersion(VK_API_VERSION_1_1);
        vulkan_version = VK_API_VERSION_1_1;

        vk::InstanceCreateInfo instance_info{};
        instance_info
                .setPApplicationInfo(&app_info)
                .setEnabledLayerCount(static_cast<uint32_t>(validation_layers.size()))
                .setPpEnabledLayerNames(validation_layers.data())
                .setEnabledExtensionCount(static_cast<uint32_t>(required_extensions.size()))
                .setPpEnabledExtensionNames(required_extensions.data());
        auto result = vk::createInstance(&instance_info, nullptr, &instance);
        if (result == vk::Result::eErrorIncompatibleDriver) {
                app_info.apiVersion = VK_API_VERSION_1_0;
                vulkan_version = VK_API_VERSION_1_0;
                result = vk::createInstance(&instance_info, nullptr, &instance);
        }
        CHECK(result, "Vulkan instance cannot be created: "s + vk::to_string(result));

        if (enable_validation) {
                dynamic_dispatcher = std::make_unique<vk::DispatchLoaderDynamic>(instance, vkGetInstanceProcAddr);
                PASS_RESULT(init_validation_layers_error_messenger());
        }

        return RETURN_TYPE();
}

RETURN_TYPE vulkan_instance::init_validation_layers_error_messenger() {
        vk::DebugUtilsMessengerCreateInfoEXT messenger_info{};
        using severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
        using type = vk::DebugUtilsMessageTypeFlagBitsEXT;
        messenger_info
                .setMessageSeverity(severity::eError | severity::eInfo | severity::eWarning) // severity::eInfo |
                .setMessageType(type::eGeneral | type::ePerformance | type::eValidation)
                .setPfnUserCallback(debugCallback)
                .setPUserData(nullptr);
        CHECKED_ASSIGN(messenger, instance.createDebugUtilsMessengerEXT(messenger_info, nullptr, *dynamic_dispatcher));
        return RETURN_TYPE();
}

RETURN_TYPE vulkan_instance::get_available_gpus(std::vector<std::pair<std::string, bool>>& gpus) {
        assert(instance);

        std::vector<vk::PhysicalDevice> physical_devices;
        CHECKED_ASSIGN(physical_devices, instance.enumeratePhysicalDevices());
        gpus.clear();
        gpus.reserve(physical_devices.size());
        for (const auto& gpu : physical_devices) {
                auto properties = gpu.getProperties();
                gpus.emplace_back(properties.deviceName, true);
        }
        std::sort(gpus.begin(), gpus.end());
        return RETURN_TYPE();
}

RETURN_TYPE vulkan_instance::destroy() {
        if (instance) {
                instance.destroy();
                if (messenger) {
                        instance.destroy(messenger, nullptr, *dynamic_dispatcher);
                }
                dynamic_dispatcher = nullptr;
                instance = nullptr;
        }
        return RETURN_TYPE();
}


} // namespace vulkan_display ----------------------------------------------------------------------------


namespace vulkan_display_detail { //------------------------------------------------------------------------

RETURN_TYPE vulkan_context::create_physical_device(uint32_t gpu_index) {
        assert(instance);
        assert(surface);
        std::vector<vk::PhysicalDevice> gpus;
        CHECKED_ASSIGN(gpus, instance.enumeratePhysicalDevices());

        if (gpu_index == vulkan_display::NO_GPU_SELECTED) {
                PASS_RESULT(choose_suitable_GPU(gpu, gpus, surface));
        } else {
                PASS_RESULT(choose_gpu_by_index(gpu, gpus, gpu_index));
                bool suitable = false;
                PASS_RESULT(is_gpu_suitable(suitable, true, gpu, surface));
        }
        auto properties = gpu.getProperties();
        if (properties.apiVersion == VK_API_VERSION_1_0) {
                vulkan_version = VK_API_VERSION_1_0;
        }
        std::cout << "Vulkan uses GPU called: " << properties.deviceName << std::endl;
        std::cout << "Used Vulkan API: " 
                << VK_VERSION_MAJOR(vulkan_version) << '.'
                << VK_VERSION_MINOR(vulkan_version) << std::endl;

        return RETURN_TYPE();
}

RETURN_TYPE vulkan_context::create_logical_device() {
        assert(gpu);
        assert(queue_family_index != NO_QUEUE_FAMILY_INDEX_FOUND);

        constexpr std::array priorities = { 1.0f };
        vk::DeviceQueueCreateInfo queue_info{};
        queue_info
                .setQueueFamilyIndex(queue_family_index)
                .setPQueuePriorities(priorities.data())
                .setQueueCount(1);

        vk::DeviceCreateInfo device_info{};
        device_info
                .setPNext(nullptr)
                .setQueueCreateInfoCount(1)
                .setPQueueCreateInfos(&queue_info)
                .setEnabledExtensionCount(static_cast<uint32_t>(required_gpu_extensions.size()))
                .setPpEnabledExtensionNames(required_gpu_extensions.data());

        vk::PhysicalDeviceFeatures2 features2{};
        vk::PhysicalDeviceSamplerYcbcrConversionFeatures yCbCr_feature{};
        if (vulkan_version == VK_API_VERSION_1_1) {
                features2.setPNext(&yCbCr_feature);
                gpu.getFeatures2(&features2);
                if (yCbCr_feature.samplerYcbcrConversion) {
                        yCbCr_supported = true;
                        device_info.setPNext(&features2);
                        std::cout << "yCbCr feature supported." << std::endl;
                }
        }

        CHECKED_ASSIGN(device, gpu.createDevice(device_info));
        return RETURN_TYPE();

}

RETURN_TYPE vulkan_context::get_present_mode() {
        std::vector<vk::PresentModeKHR> modes;
        CHECKED_ASSIGN(modes, gpu.getSurfacePresentModesKHR(surface));

        vk::PresentModeKHR first_choice{}; 
        vk::PresentModeKHR second_choice{};
        if (vsync) {
                first_choice = vk::PresentModeKHR::eFifo;
                second_choice = vk::PresentModeKHR::eFifoRelaxed;
        } else {
                first_choice = vk::PresentModeKHR::eMailbox;
                second_choice = vk::PresentModeKHR::eImmediate;
        }

        if (std::any_of(modes.begin(), modes.end(), [first_choice](auto mode) { return mode == first_choice; })) {
                swapchain_atributes.mode = first_choice;
                swapchain_atributes.mode = first_choice;
                return RETURN_TYPE();
        }
        if (std::any_of(modes.begin(), modes.end(), [second_choice](auto mode) { return mode == second_choice; })) {
                swapchain_atributes.mode = second_choice;
                return RETURN_TYPE();
        }
        swapchain_atributes.mode = modes[0];
        return RETURN_TYPE();
}

RETURN_TYPE vulkan_context::get_surface_format() {
        std::vector<vk::SurfaceFormatKHR> formats;
        CHECKED_ASSIGN(formats, gpu.getSurfaceFormatsKHR(surface));

        vk::SurfaceFormatKHR default_format{};
        default_format.format = vk::Format::eB8G8R8A8Srgb;
        default_format.colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;

        if (std::any_of(formats.begin(), formats.end(), [default_format](auto& format) {return format == default_format; })) {
                swapchain_atributes.format = default_format;
                return RETURN_TYPE();
        }
        swapchain_atributes.format = formats[0];
        return RETURN_TYPE();
}

RETURN_TYPE vulkan_context::create_swap_chain(vk::SwapchainKHR old_swapchain) {
        auto& capabilities = swapchain_atributes.capabilities;
        CHECKED_ASSIGN(capabilities, gpu.getSurfaceCapabilitiesKHR(surface));

        PASS_RESULT(get_present_mode());
        PASS_RESULT(get_surface_format());

        window_size.width = std::clamp(window_size.width,
                capabilities.minImageExtent.width,
                capabilities.maxImageExtent.width);
        window_size.height = std::clamp(window_size.height,
                capabilities.minImageExtent.height,
                capabilities.maxImageExtent.height);

        uint32_t image_count = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount != 0) {
                image_count = std::min(image_count, capabilities.maxImageCount);
        }

        //assert(capabilities.supportedUsageFlags & vk::ImageUsageFlagBits::eTransferDst);
        vk::SwapchainCreateInfoKHR swapchain_info{};
        swapchain_info
                .setSurface(surface)
                .setImageFormat(swapchain_atributes.format.format)
                .setImageColorSpace(swapchain_atributes.format.colorSpace)
                .setPresentMode(swapchain_atributes.mode)
                .setMinImageCount(image_count)
                .setImageExtent(window_size)
                .setImageArrayLayers(1)
                .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
                .setImageSharingMode(vk::SharingMode::eExclusive)
                .setPreTransform(swapchain_atributes.capabilities.currentTransform)
                .setCompositeAlpha(get_composite_alpha(swapchain_atributes.capabilities.supportedCompositeAlpha))
                .setClipped(true)
                .setOldSwapchain(old_swapchain);
        CHECKED_ASSIGN(swapchain, device.createSwapchainKHR(swapchain_info));
        return RETURN_TYPE();
}

RETURN_TYPE vulkan_context::create_swapchain_views() {
        std::vector<vk::Image> images;
        CHECKED_ASSIGN(images, device.getSwapchainImagesKHR(swapchain));
        auto image_count = static_cast<uint32_t>(images.size());

        vk::ImageViewCreateInfo image_view_info = 
                vulkan_display::default_image_view_create_info(swapchain_atributes.format.format);

        swapchain_images.resize(image_count);
        for (uint32_t i = 0; i < image_count; i++) {
                swapchain_image& swapchain_image = swapchain_images[i];
                swapchain_image.image = images[i];

                image_view_info.setImage(swapchain_image.image);
                CHECKED_ASSIGN(swapchain_image.view, device.createImageView(image_view_info));
        }
        return RETURN_TYPE();
}

RETURN_TYPE vulkan_context::init(vulkan_display::vulkan_instance&& instance, VkSurfaceKHR surface, 
        window_parameters parameters, uint32_t gpu_index) 
{
        assert(!this->instance);
        this->instance = instance.instance;
        this->dynamic_dispatcher = std::move(instance.dynamic_dispatcher);
        this->messenger = instance.messenger;
        this->vulkan_version = instance.vulkan_version;
        instance.instance = nullptr;
        instance.messenger = nullptr;

        this->surface = surface;
        window_size = vk::Extent2D{ parameters.width, parameters.height };
        vsync = parameters.vsync;

        PASS_RESULT(create_physical_device(gpu_index));
        PASS_RESULT(get_queue_family_index(queue_family_index, gpu, surface));
        PASS_RESULT(create_logical_device());
        queue = device.getQueue(queue_family_index, 0);
        PASS_RESULT(create_swap_chain());
        PASS_RESULT(create_swapchain_views());
        return RETURN_TYPE();
}

RETURN_TYPE vulkan_context::create_framebuffers(vk::RenderPass render_pass) {
        vk::FramebufferCreateInfo framebuffer_info;
        framebuffer_info
                .setRenderPass(render_pass)
                .setWidth(window_size.width)
                .setHeight(window_size.height)
                .setLayers(1);

        for(auto& swapchain_image : swapchain_images){
                framebuffer_info
                        .setAttachmentCount(1)
                        .setPAttachments(&swapchain_image.view);
                CHECKED_ASSIGN(swapchain_image.framebuffer, device.createFramebuffer(framebuffer_info));
        }
        return RETURN_TYPE();
}

RETURN_TYPE vulkan_context::recreate_swapchain(window_parameters parameters, vk::RenderPass render_pass) {
        window_size = vk::Extent2D{ parameters.width, parameters.height };
        vsync = parameters.vsync;

        PASS_RESULT(device.waitIdle());

        destroy_framebuffers();
        destroy_swapchain_views();
        vk::SwapchainKHR old_swap_chain = swapchain;
        create_swap_chain(old_swap_chain);
        device.destroySwapchainKHR(old_swap_chain);
        create_swapchain_views();
        create_framebuffers(render_pass);
        return RETURN_TYPE();
}

RETURN_TYPE vulkan_context::acquire_next_swapchain_image(uint32_t& image_index, vk::Semaphore acquire_semaphore) const {
        auto acquired = device.acquireNextImageKHR(swapchain, UINT64_MAX, acquire_semaphore, nullptr, &image_index);
        if (acquired == vk::Result::eSuboptimalKHR || acquired == vk::Result::eErrorOutOfDateKHR) {
                image_index = SWAPCHAIN_IMAGE_OUT_OF_DATE;
                return RETURN_TYPE();
        }
        CHECK(acquired, "Next swapchain image cannot be acquired."s + vk::to_string(acquired));
        return RETURN_TYPE();
}

RETURN_TYPE vulkan_context::destroy() {
        if (device) {
                // static_cast to silence nodiscard warning
                PASS_RESULT(device.waitIdle());
                destroy_framebuffers();
                destroy_swapchain_views();
                device.destroy(swapchain);
                device.destroy();
        }
        if (instance) {
                instance.destroy(surface);
                if (messenger) {
                        instance.destroy(messenger, nullptr, *dynamic_dispatcher);
                }
                instance.destroy();
        }
        dynamic_dispatcher = nullptr;
        return RETURN_TYPE();
}


} //vulkan_display_detail
