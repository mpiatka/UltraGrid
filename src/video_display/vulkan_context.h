#pragma once
#undef VULKAN_HPP_NO_EXCEPTIONS

#ifdef NO_EXCEPTIONS
#define VULKAN_HPP_NO_EXCEPTIONS
#endif //NO_EXCEPTIONS

#include <vulkan/vulkan.hpp>
//remove leaking macros
#undef min
#undef max

#include <memory>
#include <string>


namespace vulkan_display_detail {

constexpr inline vk::Result to_vk_result(bool b) {
        return b ? vk::Result::eSuccess : vk::Result::eErrorFeatureNotPresent;
}

constexpr inline vk::Result to_vk_result(vk::Result res) {
        return res;
}

} // namespace vulkan_display_detail


#ifdef NO_EXCEPTIONS //-------------------------------------------------------
//EXCEPTIONS ARE DISABLED
extern std::string vulkan_display_error_message;

#define RETURN_TYPE vk::Result

#define PASS_RESULT(expr) {                                              \
        if (vk::Result res = expr; res != vk::Result::eSuccess) {        \
                assert(false);                                           \
                return res;                                              \
        }                                                                \
}

#define CHECK(expr, msg) {                                               \
        vk::Result res = to_vk_result(expr);                             \
        if ( res != vk::Result::eSuccess) {                              \
                assert(false);                                           \
                vulkan_display_error_message = msg;                      \
                return res;                                              \
        }                                                                \
}

#define CHECKED_ASSIGN(variable, expr) {                                 \
        auto[checked_assign_return_val, checked_assign_value] = expr;    \
        if (checked_assign_return_val != vk::Result::eSuccess) {         \
                assert(false);                                           \
                return checked_assign_return_val;                        \
        } else {variable = std::move(checked_assign_value);}             \
}

#else //NO_EXCEPTIONS -------------------------------------------------------
//EXCEPTIONS ARE ENABLED
#include<exception>

struct  vulkan_display_exception : public std::runtime_error {
        explicit vulkan_display_exception(const std::string& msg) :
                std::runtime_error{ msg } { }
};

#define RETURN_TYPE void

#define PASS_RESULT(expr) { expr; }

#define CHECK(expr, msg) { if (to_vk_result(expr) != vk::Result::eSuccess) throw vulkan_display_exception{msg}; }

#define CHECKED_ASSIGN(variable, expr) { (variable) = (expr); }

#endif //NO_EXCEPTIONS -------------------------------------------------------


namespace vulkan_display {

struct window_parameters {
        uint32_t width;
        uint32_t height;
        bool vsync;

        constexpr bool operator==(const window_parameters& other) const {
                return width == other.width &&
                        height == other.height &&
                        vsync == other.vsync;
        }
        constexpr bool operator!=(const window_parameters& other) const {
                return !(*this == other);
        }
};

constexpr uint32_t NO_GPU_SELECTED = UINT32_MAX;

constexpr vk::ImageViewCreateInfo default_image_view_create_info(vk::Format format) {
        vk::ImageViewCreateInfo image_view_info{ {}, {}, vk::ImageViewType::e2D, format };
        image_view_info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        image_view_info.subresourceRange.levelCount = 1;
        image_view_info.subresourceRange.layerCount = 1;
        return image_view_info;
}

class vulkan_instance;

} // namespace vulkan_display ---------------------------------------------


namespace vulkan_display_detail {

using c_str = const char*;
using namespace std::literals;

constexpr uint32_t NO_QUEUE_FAMILY_INDEX_FOUND = UINT32_MAX;
constexpr uint32_t SWAPCHAIN_IMAGE_OUT_OF_DATE = UINT32_MAX;

class vulkan_context {
        vk::Instance instance;
        std::unique_ptr<vk::DispatchLoaderDynamic> dynamic_dispatcher{};
        vk::DebugUtilsMessengerEXT messenger;
        uint32_t vulkan_version{};

        vk::PhysicalDevice gpu;
        vk::Device device;
        bool yCbCr_supported = false;

        uint32_t queue_family_index = NO_QUEUE_FAMILY_INDEX_FOUND;
        vk::Queue queue;

        vk::SurfaceKHR surface;
        vk::SwapchainKHR swapchain;
        struct {
                vk::SurfaceCapabilitiesKHR capabilities;
                vk::SurfaceFormatKHR format;
                vk::PresentModeKHR mode = vk::PresentModeKHR::eFifo;
        } swapchain_atributes;

        struct swapchain_image {
                vk::Image image;
                vk::ImageView view;
                vk::Framebuffer framebuffer;
        };
        std::vector<swapchain_image> swapchain_images{};

        vk::Extent2D window_size{ 0, 0 };
        bool vsync = true;
public:
        //getters
        vk::PhysicalDevice get_gpu() { return gpu; }
        vk::Device get_device() { return device; }
        bool is_yCbCr_supported() { return yCbCr_supported; }
        uint32_t get_queue_familt_index() { return queue_family_index; }
        vk::Queue get_queue() { return queue; }
        vk::SwapchainKHR get_swapchain() { return swapchain; }
        vk::Format get_swapchain_image_format() { return swapchain_atributes.format.format; };
        vk::Extent2D get_window_size() { return window_size; }
private:
        RETURN_TYPE create_physical_device(uint32_t gpu_index);

        RETURN_TYPE create_logical_device();

        RETURN_TYPE get_present_mode();

        RETURN_TYPE get_surface_format();

        RETURN_TYPE create_swap_chain(vk::SwapchainKHR old_swap_chain = vk::SwapchainKHR{});

        RETURN_TYPE create_swapchain_views();

        void destroy_swapchain_views() {
                for (auto& image : swapchain_images) {
                        device.destroy(image.view);
                }
        }

        void destroy_framebuffers() {
                for (auto& image : swapchain_images) {
                        device.destroy(image.framebuffer);
                }
        }

public:
        using window_parameters = vulkan_display::window_parameters;

        vulkan_context() = default;

        RETURN_TYPE init(vulkan_display::vulkan_instance&& instance, VkSurfaceKHR surface, 
                window_parameters, uint32_t gpu_index);

        RETURN_TYPE destroy();

        RETURN_TYPE create_framebuffers(vk::RenderPass render_pass);

        RETURN_TYPE acquire_next_swapchain_image(uint32_t& image_index, vk::Semaphore acquire_semaphore) const;

        vk::Framebuffer get_framebuffer(uint32_t framebuffer_id) {
                return swapchain_images[framebuffer_id].framebuffer;
        }

        window_parameters get_window_parameters() {
                return { window_size.width, window_size.height, vsync };
        }

        RETURN_TYPE recreate_swapchain(window_parameters parameters, vk::RenderPass render_pass);
};

}//namespace vulkan_display_detail ----------------------------------------------------------------


namespace vulkan_display {

class vulkan_instance {
        vk::Instance instance{};
        std::unique_ptr<vk::DispatchLoaderDynamic> dynamic_dispatcher = nullptr;
        vk::DebugUtilsMessengerEXT messenger{};
        uint32_t vulkan_version = VK_API_VERSION_1_1;

        RETURN_TYPE init_validation_layers_error_messenger();

        friend RETURN_TYPE vulkan_display_detail::vulkan_context::init(vulkan_instance&& instance, 
                VkSurfaceKHR surface, window_parameters parameters, uint32_t gpu_index);
public:
        vulkan_instance() = default;
        vulkan_instance(const vulkan_instance& other) = delete;
        vulkan_instance& operator=(const vulkan_instance& other) = delete;
        vulkan_instance(vulkan_instance&& other) = delete;
        vulkan_instance& operator=(vulkan_instance&& other) = delete;
        
        ~vulkan_instance() {
                destroy();
        }

        /**
         * @param required_extensions   Vulkan instance extensions requested by aplication,
         *                              usually needed for creating vulkan surface
         * @param enable_validation     Enable vulkan validation layers, they should be disabled in release build.
         */
        RETURN_TYPE init(std::vector<const char*>& required_extensions, bool enable_validation);
        
        /**
         * @brief returns all available grafhics cards
         *  first parameter is gpu name,
         *  second parameter is true only if the gpu is suitable for vulkan_display
         */
        RETURN_TYPE get_available_gpus(std::vector<std::pair<std::string, bool>>& gpus);

        vk::Instance& get_instance() {
                return instance;
        }

        RETURN_TYPE destroy();
};

}
