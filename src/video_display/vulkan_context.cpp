#define DEBUG
#include "vulkan_context.h"
#include<iostream>

using namespace vulkan_display_detail;

std::string vulkan_display_error_message = "";

vk::ImageViewCreateInfo default_image_view_create_info(vk::Format format) {
        vk::ImageViewCreateInfo image_view_info{};
        image_view_info
                .setViewType(vk::ImageViewType::e2D)
                .setFormat(format);
        image_view_info.components
                .setR(vk::ComponentSwizzle::eIdentity)
                .setG(vk::ComponentSwizzle::eIdentity)
                .setB(vk::ComponentSwizzle::eIdentity)
                .setA(vk::ComponentSwizzle::eIdentity);
        image_view_info.subresourceRange
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setBaseMipLevel(0)
                .setLevelCount(1)
                .setBaseArrayLayer(0)
                .setLayerCount(1);
        return image_view_info;
}

namespace {
        VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
                [[maybe_unused]] VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT messageType,
                const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                [[maybe_unused]] void* pUserData)
        {
                if (false && strstr(pCallbackData->pMessage, "VUID-vkDestroyDevice-device-00378") != NULL) {
                        return VK_FALSE;
                }
                std::cout << "validation layer: " << pCallbackData->pMessage << '\n' << std::endl;
                return VK_FALSE;
        }

        RETURN_VAL check_validation_layers(const std::vector<c_str>& required_layers) {
                std::vector<vk::LayerProperties>  layers;
                CHECKED_ASSIGN(layers, vk::enumerateInstanceLayerProperties());
                //for (auto& l : layers) puts(l.layerName);

                for (auto& req_layer : required_layers) {
                        auto layer_equals = [req_layer](auto layer) { return strcmp(req_layer, layer.layerName) == 0; };
                        bool found = std::any_of(layers.begin(), layers.end(), layer_equals);
                        CHECK(found, "Layer "s + req_layer + " is not supported.");
                }
                return RETURN_VAL();
        }

        RETURN_VAL check_instance_extensions(const std::vector<c_str>& required_extensions) {
                std::vector<vk::ExtensionProperties> extensions;
                CHECKED_ASSIGN(extensions, vk::enumerateInstanceExtensionProperties(nullptr));

                for (auto& req_exten : required_extensions) {
                        auto extension_equals = [req_exten](auto exten) { return strcmp(req_exten, exten.extensionName) == 0; };
                        bool found = std::any_of(extensions.begin(), extensions.end(), extension_equals);
                        CHECK(found, "Instance extension "s + req_exten + " is not supported.");
                }
                return RETURN_VAL();
        }

        RETURN_VAL check_device_extensions(const std::vector<c_str>& required_extensions, const vk::PhysicalDevice& device) {
                std::vector<vk::ExtensionProperties> extensions;
                CHECKED_ASSIGN(extensions, device.enumerateDeviceExtensionProperties(nullptr));

                for (auto& req_exten : required_extensions) {
                        auto extension_equals = [req_exten](auto exten) { return strcmp(req_exten, exten.extensionName) == 0; };
                        bool found = std::any_of(extensions.begin(), extensions.end(), extension_equals);
                        CHECK(found, "Device extension "s + req_exten + " is not supported.");
                }
                return RETURN_VAL();
        }

        vk::PhysicalDevice choose_GPU(const std::vector<vk::PhysicalDevice>& gpus) {
                for (auto& gpu : gpus) {
                        auto properties = gpu.getProperties();
                        if (properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
                                return gpu;
                        }
                }

                for (auto& gpu : gpus) {
                        auto properties = gpu.getProperties();
                        if (properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) {
                                return gpu;
                        }
                }
                return gpus[0];
        }

        vk::CompositeAlphaFlagBitsKHR get_composite_alpha(vk::CompositeAlphaFlagsKHR capabilities) {
                using underlying = std::underlying_type_t<vk::CompositeAlphaFlagBitsKHR>;
                underlying result = 1;
                while (!(result & static_cast<underlying>(capabilities))) {
                        result <<= 1;
                }
                return static_cast<vk::CompositeAlphaFlagBitsKHR>(result);
        }
} //namespace

namespace vulkan_display_detail {

        RETURN_VAL Vulkan_context::create_instance(std::vector<c_str>& required_extensions) {
#ifdef DEBUG
                std::vector validation_layers{ "VK_LAYER_KHRONOS_validation" };
                PASS_RESULT(check_validation_layers(validation_layers));
#else
                std::vector<c_str> validation_layers;
#endif
                const char* debug_extension = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
                required_extensions.push_back(debug_extension);
                PASS_RESULT(check_instance_extensions(required_extensions));

                vk::ApplicationInfo app_info{};
                app_info.setApiVersion(VK_API_VERSION_1_0);

                vk::InstanceCreateInfo instance_info{};
                instance_info
                        .setPApplicationInfo(&app_info)
                        .setPEnabledLayerNames(validation_layers)
                        .setPEnabledExtensionNames(required_extensions);
                CHECKED_ASSIGN(instance, vk::createInstance(instance_info));

#ifdef DEBUG
                dynamic_dispatch_loader = std::make_unique<vk::DispatchLoaderDynamic>(instance, vkGetInstanceProcAddr);
                PASS_RESULT(init_validation_layers_error_messenger());
#endif

                return RETURN_VAL();
        }

        RETURN_VAL Vulkan_context::init_validation_layers_error_messenger() {
                vk::DebugUtilsMessengerCreateInfoEXT messenger_info{};
                using severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
                using type = vk::DebugUtilsMessageTypeFlagBitsEXT;
                messenger_info
                        .setMessageSeverity(severity::eError | severity::eInfo | severity::eWarning) // severity::eInfo |
                        .setMessageType(type::eGeneral | type::ePerformance | type::eValidation)
                        .setPfnUserCallback(debugCallback)
                        .setPUserData(nullptr);
                CHECKED_ASSIGN(messenger, instance.createDebugUtilsMessengerEXT(messenger_info, nullptr, *dynamic_dispatch_loader));
                return RETURN_VAL();
        }

        RETURN_VAL Vulkan_context::create_physical_device() {
                assert(instance != vk::Instance{});
                std::vector<vk::PhysicalDevice> gpus;
                CHECKED_ASSIGN(gpus, instance.enumeratePhysicalDevices());

                gpu = choose_GPU(gpus);
                auto properties = gpu.getProperties();
                std::cout << "Vulkan uses GPU called: "s + properties.deviceName.data() << std::endl;
                return RETURN_VAL();
        }

        RETURN_VAL Vulkan_context::get_queue_family_index() {
                assert(gpu != vk::PhysicalDevice{});

                std::vector<vk::QueueFamilyProperties> families = gpu.getQueueFamilyProperties();

                queue_family_index = UINT32_MAX;
                for (uint32_t i = 0; i < families.size(); i++) {
                        VkBool32 surface_supported;
                        CHECKED_ASSIGN(surface_supported, gpu.getSurfaceSupportKHR(i, surface));

                        if (surface_supported && (families[i].queueFlags & vk::QueueFlagBits::eGraphics)) {
                                queue_family_index = i;
                                break;
                        }
                }
                CHECK(queue_family_index != UINT32_MAX, "No suitable GPU queue found.");
                return RETURN_VAL();
        }

        RETURN_VAL Vulkan_context::create_logical_device() {
                assert(gpu != vk::PhysicalDevice{});
                assert(queue_family_index != INT32_MAX);

                std::vector<c_str> required_extensions = { "VK_KHR_swapchain" };
                PASS_RESULT(check_device_extensions(required_extensions, gpu));

                float priorities[] = { 1.0 };
                vk::DeviceQueueCreateInfo queue_info{};
                queue_info
                        .setQueueFamilyIndex(queue_family_index)
                        .setPQueuePriorities(priorities)
                        .setQueueCount(1);

                vk::DeviceCreateInfo device_info{};
                device_info
                        .setQueueCreateInfoCount(1)
                        .setPQueueCreateInfos(&queue_info)
                        .setPEnabledExtensionNames(required_extensions);

                CHECKED_ASSIGN(device, gpu.createDevice(device_info));
                return RETURN_VAL();
        }

        RETURN_VAL Vulkan_context::get_present_mode() {
                std::vector<vk::PresentModeKHR> modes;
                CHECKED_ASSIGN(modes, gpu.getSurfacePresentModesKHR(surface));

                bool vsync = true;
                vk::PresentModeKHR first_choice{}, second_choice{};
                if (vsync) {
                        first_choice = vk::PresentModeKHR::eFifo;
                        second_choice = vk::PresentModeKHR::eFifoRelaxed;
                }
                else {
                        first_choice = vk::PresentModeKHR::eMailbox;
                        second_choice = vk::PresentModeKHR::eImmediate;
                }

                if (std::any_of(modes.begin(), modes.end(), [first_choice](auto mode) { return mode == first_choice; })) {
                        swapchain_atributes.mode = first_choice;
                        swapchain_atributes.mode = first_choice;
                        return RETURN_VAL();
                }
                if (std::any_of(modes.begin(), modes.end(), [second_choice](auto mode) { return mode == second_choice; })) {
                        swapchain_atributes.mode = second_choice;
                        return RETURN_VAL();
                }
                swapchain_atributes.mode = modes[0];
                return RETURN_VAL();
        }

        RETURN_VAL Vulkan_context::get_surface_format() {
                std::vector<vk::SurfaceFormatKHR> formats;
                CHECKED_ASSIGN(formats, gpu.getSurfaceFormatsKHR(surface));

                vk::SurfaceFormatKHR default_format{};
                default_format.format = vk::Format::eB8G8R8A8Srgb;
                default_format.colorSpace = vk::ColorSpaceKHR::eVkColorspaceSrgbNonlinear;

                if (std::any_of(formats.begin(), formats.end(), [default_format](auto& format) {return format == default_format; })) {
                        swapchain_atributes.format = default_format;
                        return RETURN_VAL();
                }
                swapchain_atributes.format = formats[0];
                return RETURN_VAL();
        }

        RETURN_VAL Vulkan_context::create_swap_chain(vk::SwapchainKHR old_swapchain) {
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
                return RETURN_VAL();
        }

        RETURN_VAL Vulkan_context::create_swapchain_views() {
                std::vector<vk::Image> images;
                CHECKED_ASSIGN(images, device.getSwapchainImagesKHR(swapchain));
                uint32_t image_count = static_cast<uint32_t>(images.size());

                vk::ImageViewCreateInfo image_view_info = default_image_view_create_info(swapchain_atributes.format.format);

                swapchain_images.resize(image_count);
                for (uint32_t i = 0; i < image_count; i++) {
                        Swapchain_image& image = swapchain_images[i];
                        image.image = std::move(images[i]);

                        image_view_info.setImage(swapchain_images[i].image);
                        CHECKED_ASSIGN(image.view, device.createImageView(image_view_info));
                }
                return RETURN_VAL();
        }

        RETURN_VAL Vulkan_context::init(VkSurfaceKHR surface, uint32_t width, uint32_t height) {
                this->surface = surface;
                this->window_size = vk::Extent2D{ width, height };

                PASS_RESULT(create_physical_device());
                PASS_RESULT(get_queue_family_index());
                PASS_RESULT(create_logical_device());
                queue = device.getQueue(queue_family_index, 0);
                PASS_RESULT(create_swap_chain());
                PASS_RESULT(create_swapchain_views());
        }

        RETURN_VAL Vulkan_context::create_framebuffers(vk::RenderPass render_pass) {
                vk::FramebufferCreateInfo framebuffer_info;
                framebuffer_info
                        .setRenderPass(render_pass)
                        .setWidth(window_size.width)
                        .setHeight(window_size.height)
                        .setLayers(1);

                for (size_t i = 0; i < swapchain_images.size(); i++) {
                        framebuffer_info.setAttachments(swapchain_images[i].view);
                        CHECKED_ASSIGN(swapchain_images[i].framebuffer, device.createFramebuffer(framebuffer_info));
                }
                return RETURN_VAL();
        }

        RETURN_VAL Vulkan_context::recreate_swapchain(vk::Extent2D window_size, vk::RenderPass render_pass) {
                device.waitIdle();

                window_size = window_size;
                destroy_framebuffers();
                destroy_swapchain_views();
                vk::SwapchainKHR old_swap_chain = swapchain;
                create_swap_chain(old_swap_chain);
                device.destroySwapchainKHR(old_swap_chain);
                create_swapchain_views();
                create_framebuffers(render_pass);
        }

        Vulkan_context::~Vulkan_context() {
                device.waitIdle();

                destroy_framebuffers();
                destroy_swapchain_views();
                device.destroy(swapchain);
                instance.destroy(surface);
                device.destroy();
                instance.destroy(messenger, nullptr, *dynamic_dispatch_loader);
                instance.destroy();
        }

} //vulkan_display_detail
