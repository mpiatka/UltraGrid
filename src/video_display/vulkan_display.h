#pragma once

#include "concurent_queue.h"
#include "vulkan_context.h"
#include "vulkan_transfer_image.h"

#include <mutex>
#include <utility>

namespace vulkan_display_detail {

struct render_area {
        uint32_t x;
        uint32_t y;
        uint32_t width;
        uint32_t height;
};

} // vulkan_display_detail

namespace vulkan_display {

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
        std::mutex device_mutex;

        vulkan_display_detail::render_area render_area;
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


        struct image_semaphores {
                vk::Semaphore image_acquired;
                vk::Semaphore image_rendered;
        };
        std::vector<image_semaphores> image_semaphores;


        using transfer_image = vulkan_display_detail::transfer_image;
        unsigned transfer_image_count;
        std::vector<transfer_image> transfer_images;
        image_description current_image_description;

        concurrent_queue<transfer_image*> available_img_queue;
        concurrent_queue<transfer_image*> filled_img_queue;


        bool minimalised = false;
private:

        RETURN_TYPE create_texture_sampler();

        RETURN_TYPE create_render_pass();

        RETURN_TYPE create_descriptor_set_layout();

        RETURN_TYPE create_pipeline_layout();

        RETURN_TYPE create_graphics_pipeline();

        RETURN_TYPE create_command_pool();

        RETURN_TYPE create_command_buffers();

        RETURN_TYPE create_transfer_image(transfer_image*& result, image_description description);

        RETURN_TYPE create_image_semaphores();

        RETURN_TYPE allocate_description_sets();

        RETURN_TYPE record_graphics_commands(transfer_image& transfer_image, uint32_t swapchain_image_id);

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
        RETURN_TYPE create_instance(std::vector<const char*>& required_extensions, bool enable_validation) {
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
        RETURN_TYPE get_available_gpus(std::vector<std::pair<std::string, bool>>& gpus) {
                return context.get_available_gpus(gpus);
        }

        RETURN_TYPE init(VkSurfaceKHR surface, uint32_t transfer_image_count,
                window_changed_callback* window, uint32_t gpu_index = NO_GPU_SELECTED);

        RETURN_TYPE acquire_image(image& image, image_description description);

        RETURN_TYPE queue_image(image image) {
                filled_img_queue.push(&image.get_transfer_image());
        }

        RETURN_TYPE copy_and_queue_image(std::byte* frame, image_description description);

        RETURN_TYPE display_queued_image();

        /**
         * @brief Hint to vulkan display that some window parameters spicified in struct Window_parameters changed
         */
        RETURN_TYPE window_parameters_changed(window_parameters new_parameters);

        RETURN_TYPE window_parameters_changed() {
                PASS_RESULT(window_parameters_changed(window->get_window_parameters()));
                return RETURN_TYPE();
        }
};

} //vulkan_display
