#pragma once

#include "vulkan_context.h"
#include <utility>
#include <mutex>


class transfer_image {
        vk::DeviceMemory memory;
        vk::Image image;
        vk::ImageLayout layout;
        vk::AccessFlagBits access;
public:
        vk::ImageView view;
        std::byte* ptr;

        vk::Format format;
        vk::Extent2D size;
        size_t row_pitch;

        vk::Fence is_available_fence;

        bool update_desciptor_set;
        vk::Sampler sampler;

        RETURN_VAL create(vk::Device device, vk::PhysicalDevice gpu, vk::Extent2D size, vk::Format format);
        
        vk::ImageMemoryBarrier create_memory_barrier( 
                vk::ImageLayout new_layout, 
                vk::AccessFlagBits new_access_mask,
                uint32_t src_queue_family_index = VK_QUEUE_FAMILY_IGNORED,
                uint32_t dst_queue_family_index = VK_QUEUE_FAMILY_IGNORED);

        /// update_description_sets should be called everytime before recording the command buffer
        RETURN_VAL update_description_set(vk::Device device, vk::DescriptorSet descriptor_set, vk::Sampler sampler);

        RETURN_VAL destroy(vk::Device device, bool destroy_fence = true);
public:
        transfer_image() = default;
};




class window_changed_callback {
protected:
        ~window_changed_callback() = default;
public:
        virtual window_parameters get_window_parameters() = 0;
};

class vulkan_display {
        window_changed_callback* window = nullptr;
        vulkan_display_detail::vulkan_context context;
        vk::Device device;
        std::mutex device_lock;

        vk::Viewport viewport;
        vk::Rect2D scissor;

        vk::ShaderModule vertex_shader;
        vk::ShaderModule fragment_shader;

        vk::RenderPass render_pass;
        vk::ClearValue clear_color;

        vk::Sampler sampler{};
        vk::DescriptorSetLayout descriptor_set_layout;
        vk::DescriptorPool descriptor_pool;
        std::vector<vk::DescriptorSet> descriptor_sets;

        vk::PipelineLayout pipeline_layout;
        vk::Pipeline pipeline;

        vk::CommandPool command_pool;
        std::vector<vk::CommandBuffer> command_buffers;

        constexpr static unsigned concurent_paths_count = 3;
        unsigned current_path_id = 0;
        struct path {
                vk::Semaphore image_acquired_semaphore;
                vk::Semaphore image_rendered_semaphore;
        };
        std::vector<path> concurent_paths;

        std::vector<transfer_image> transfer_images;
        vk::Extent2D current_image_size;

        struct {
                uint32_t x;
                uint32_t y;
                uint32_t width;
                uint32_t height;
        } render_area{};

        bool minimalised = false;
private:     

        RETURN_VAL create_texture_sampler();

        RETURN_VAL create_render_pass();

        RETURN_VAL create_descriptor_set_layout();

        RETURN_VAL create_pipeline_layout();

        RETURN_VAL create_graphics_pipeline();

        RETURN_VAL create_paths();

        RETURN_VAL create_command_pool();

        RETURN_VAL create_command_buffers();

        RETURN_VAL create_concurrent_paths();

        RETURN_VAL create_transfer_image(uint32_t width, uint32_t height, vk::Format format);

        RETURN_VAL allocate_description_sets();

        RETURN_VAL record_graphics_commands(unsigned current_path_id, uint32_t image_index);

        RETURN_VAL update_render_area();

public:
        vulkan_display() = default;
        vulkan_display(const vulkan_display& other) = delete;
        vulkan_display& operator=(const vulkan_display& other) = delete;
        vulkan_display(vulkan_display&& other) = delete;
        vulkan_display& operator=(vulkan_display&& other) = delete;

        ~vulkan_display();

        /**
         * @param required_extensions   Vulkan instance extensions requested by aplication,
         *                              usually needed for creating vulkan surface
         * @param enable_validation     Enable vulkan validation layers, they should be disabled in release build.
         */
        RETURN_VAL create_instance(std::vector<const char*>& required_extensions, bool enable_validation) {
                return context.create_instance(required_extensions, enable_validation);
        }

        const vk::Instance& get_instance() {
                return context.instance;
        }

        /**
         * @brief returns all available grafhics cards
         *  first parameter is gpu name,
         *  second parameter is true only if the gpu is suitable for vulkan_display
         */
        RETURN_VAL get_available_gpus(std::vector<std::pair<std::string, bool>>& gpus) {
                return context.get_available_gpus(gpus);
        }

        RETURN_VAL init(VkSurfaceKHR surface, window_changed_callback* window, uint32_t gpu_index = NO_GPU_SELECTED);

        RETURN_VAL render(
                std::byte* frame,
                uint32_t image_width,
                uint32_t image_height,
                vk::Format format = vk::Format::eR8G8B8A8Srgb);

        /**
         * @brief Hint to vulkan display that some window parameters spicified in struct Window_parameters changed
         */
        RETURN_VAL window_parameters_changed(window_parameters new_parameters);

        RETURN_VAL window_parameters_changed() {
                PASS_RESULT(window_parameters_changed(window->get_window_parameters()));
                return RETURN_VAL();
        }
};
