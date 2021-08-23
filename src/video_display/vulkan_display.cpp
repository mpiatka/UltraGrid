#include "vulkan_display.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>


using namespace vulkan_display_detail;

namespace {

/* TODO: enable when gcc 8 is no longer supported
   because compilers under gcc 8 doesn't support std::filesystem
  #include <filesystem>
RETURN_TYPE create_shader(vk::ShaderModule& shader,
        const std::filesystem::path& file_path,
        const vk::Device& device)
{
        std::ifstream file(file_path, std::ios::binary);
        CHECK(file.is_open(), "Failed to open file:"s + file_path.string());
        auto size = std::filesystem::file_size(file_path);
        assert(size % 4 == 0);
        std::vector<std::uint32_t> shader_code(size / 4);
        file.read(reinterpret_cast<char*>(shader_code.data()), static_cast<std::streamsize>(size));
        CHECK(file.good(), "Error reading from file:"s + file_path.string());

        vk::ShaderModuleCreateInfo shader_info;
        shader_info
                .setCodeSize(shader_code.size() * 4)
                .setPCode(shader_code.data());
        CHECKED_ASSIGN(shader, device.createShaderModule(shader_info));
        return RETURN_TYPE();
}
*/

RETURN_TYPE create_shader(vk::ShaderModule& shader,
        const std::string& file_path,
        const vk::Device& device)
{
        std::ifstream file(file_path, std::ios::binary | std::ios::ate);
        CHECK(file.is_open(), "Failed to open file:"s + file_path);
        size_t size = static_cast<size_t>(file.tellg());
        file.seekg(0);
        assert(size % 4 == 0);
        std::vector<std::uint32_t> shader_code(size / 4);
        file.read(reinterpret_cast<char*>(shader_code.data()), static_cast<std::streamsize>(size));
        CHECK(file.good(), "Error reading from file:"s + file_path);

        vk::ShaderModuleCreateInfo shader_info;
        shader_info
                .setCodeSize(shader_code.size() * 4)
                .setPCode(shader_code.data());
        CHECKED_ASSIGN(shader, device.createShaderModule(shader_info));
        return RETURN_TYPE();
}


RETURN_TYPE update_render_area_viewport_scissor(render_area& render_area, vk::Viewport& viewport, vk::Rect2D& scissor, 
        vk::Extent2D window_size, vk::Extent2D transfer_image_size) {

        double wnd_aspect = static_cast<double>(window_size.width) / window_size.height;
        double img_aspect = static_cast<double>(transfer_image_size.width) / transfer_image_size.height;

        if (wnd_aspect > img_aspect) {
                render_area.height = window_size.height;
                render_area.width = static_cast<uint32_t>(std::round(window_size.height * img_aspect));
                render_area.x = (window_size.width - render_area.width) / 2;
                render_area.y = 0;
        } else {
                render_area.width = window_size.width;
                render_area.height = static_cast<uint32_t>(std::round(window_size.width / img_aspect));
                render_area.x = 0;
                render_area.y = (window_size.height - render_area.height) / 2;
        }

        viewport
                .setX(static_cast<float>(render_area.x))
                .setY(static_cast<float>(render_area.y))
                .setWidth(static_cast<float>(render_area.width))
                .setHeight(static_cast<float>(render_area.height))
                .setMinDepth(0.f)
                .setMaxDepth(1.f);
        scissor
                .setOffset({ static_cast<int32_t>(render_area.x), static_cast<int32_t>(render_area.y) })
                .setExtent({ render_area.width, render_area.height });
        return RETURN_TYPE();
}

transfer_image& acquire_transfer_image(concurrent_queue<transfer_image*>& available_img_queue,
        concurrent_queue<vulkan_display::image>& filled_img_queue, unsigned filled_img_max_count)
{
        // first try available_img_queue
        auto maybe_transfer_image = available_img_queue.try_pop();
        if (maybe_transfer_image.has_value()) {
                assert(*maybe_transfer_image);
                return **maybe_transfer_image;
        }
        // if available_img_queue is empty and filled_img_queue is almost full,
        // take frame from filled_img_queue
        {
                auto [lock, deque] = filled_img_queue.get_underlying_deque();
                while (deque.size() > filled_img_max_count) {
                        vulkan_display::image front = deque.front();
                        deque.pop_front();
                        auto* front_image_ptr = front.get_transfer_image();
                        if (front_image_ptr) {
                                return *front_image_ptr;
                        }
                }
        }
        //else wait for frame from available_img_queue
        return *available_img_queue.pop();
}

} //namespace -------------------------------------------------------------


namespace vulkan_display {

RETURN_TYPE vulkan_display::create_texture_sampler() {
        vk::SamplerCreateInfo sampler_info;
        sampler_info
                .setAddressModeU(vk::SamplerAddressMode::eClampToBorder)
                .setAddressModeV(vk::SamplerAddressMode::eClampToBorder)
                .setAddressModeW(vk::SamplerAddressMode::eClampToBorder)
                .setMagFilter(vk::Filter::eLinear)
                .setMinFilter(vk::Filter::eLinear)
                .setAnisotropyEnable(false)
                .setUnnormalizedCoordinates(false);
        CHECKED_ASSIGN(sampler, device.createSampler(sampler_info));
        return RETURN_TYPE();
}

RETURN_TYPE vulkan_display::create_render_pass() {
        vk::RenderPassCreateInfo render_pass_info;

        vk::AttachmentDescription color_attachment;
        color_attachment
                .setFormat(context.get_swapchain_image_format())
                .setSamples(vk::SampleCountFlagBits::e1)
                .setLoadOp(vk::AttachmentLoadOp::eClear)
                .setStoreOp(vk::AttachmentStoreOp::eStore)
                .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                .setInitialLayout(vk::ImageLayout::eUndefined)
                .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);
        render_pass_info
                .setAttachmentCount(1)
                .setPAttachments(&color_attachment);

        vk::AttachmentReference attachment_reference;
        attachment_reference
                .setAttachment(0)
                .setLayout(vk::ImageLayout::eColorAttachmentOptimal);
        vk::SubpassDescription subpass;
        subpass
                .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
                .setColorAttachmentCount(1)
                .setPColorAttachments(&attachment_reference);
        render_pass_info
                .setSubpassCount(1)
                .setPSubpasses(&subpass);

        vk::SubpassDependency subpass_dependency{};
        subpass_dependency
                .setSrcSubpass(VK_SUBPASS_EXTERNAL)
                .setDstSubpass(0)
                .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
                .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
                .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
        render_pass_info
                .setDependencyCount(1)
                .setPDependencies(&subpass_dependency);

        CHECKED_ASSIGN(render_pass, device.createRenderPass(render_pass_info));

        vk::ClearColorValue clear_color_value{};
        clear_color_value.setFloat32({ 0.01f, 0.01f, 0.01f, 1.0f });
        clear_color.setColor(clear_color_value);

        return RETURN_TYPE();
}

RETURN_TYPE vulkan_display::create_descriptor_set_layout() {
        vk::DescriptorSetLayoutBinding descriptor_set_layout_bindings;
        descriptor_set_layout_bindings
                .setBinding(1)
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
                .setStageFlags(vk::ShaderStageFlagBits::eFragment)
                .setPImmutableSamplers(&sampler);

        vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_info{};
        descriptor_set_layout_info
                .setBindingCount(1)
                .setPBindings(&descriptor_set_layout_bindings);

        CHECKED_ASSIGN(descriptor_set_layout,
                device.createDescriptorSetLayout(descriptor_set_layout_info));
        return RETURN_TYPE();
}

RETURN_TYPE vulkan_display::create_graphics_pipeline() {
        create_descriptor_set_layout();

        vk::PipelineLayoutCreateInfo pipeline_layout_info{};

        vk::PushConstantRange push_constants;
        push_constants
                .setOffset(0)
                .setSize(sizeof(render_area))
                .setStageFlags(vk::ShaderStageFlagBits::eFragment);
        pipeline_layout_info
                .setPushConstantRangeCount(1)
                .setPPushConstantRanges(&push_constants)
                .setSetLayoutCount(1)
                .setPSetLayouts(&descriptor_set_layout);
        CHECKED_ASSIGN(pipeline_layout, device.createPipelineLayout(pipeline_layout_info));

        vk::GraphicsPipelineCreateInfo pipeline_info{};

        std::array<vk::PipelineShaderStageCreateInfo, 2> shader_stages_infos;
        shader_stages_infos[0]
                .setModule(vertex_shader)
                .setPName("main")
                .setStage(vk::ShaderStageFlagBits::eVertex);
        shader_stages_infos[1]
                .setModule(fragment_shader)
                .setPName("main")
                .setStage(vk::ShaderStageFlagBits::eFragment);
        pipeline_info
                .setStageCount(static_cast<uint32_t>(shader_stages_infos.size()))
                .setPStages(shader_stages_infos.data());

        vk::PipelineVertexInputStateCreateInfo vertex_input_state_info{};
        pipeline_info.setPVertexInputState(&vertex_input_state_info);

        vk::PipelineInputAssemblyStateCreateInfo input_assembly_state_info{};
        input_assembly_state_info.setTopology(vk::PrimitiveTopology::eTriangleList);
        pipeline_info.setPInputAssemblyState(&input_assembly_state_info);

        vk::PipelineViewportStateCreateInfo viewport_state_info;
        viewport_state_info
                .setScissorCount(1)
                .setViewportCount(1);
        pipeline_info.setPViewportState(&viewport_state_info);

        vk::PipelineRasterizationStateCreateInfo rasterization_info{};
        rasterization_info
                .setPolygonMode(vk::PolygonMode::eFill)
                .setLineWidth(1.f);
        pipeline_info.setPRasterizationState(&rasterization_info);

        vk::PipelineMultisampleStateCreateInfo multisample_info;
        multisample_info
                .setSampleShadingEnable(false)
                .setRasterizationSamples(vk::SampleCountFlagBits::e1);
        pipeline_info.setPMultisampleState(&multisample_info);

        using color_flags = vk::ColorComponentFlagBits;
        vk::PipelineColorBlendAttachmentState color_blend_attachment{};
        color_blend_attachment
                .setBlendEnable(false)
                .setColorWriteMask(color_flags::eR | color_flags::eG | color_flags::eB | color_flags::eA);
        vk::PipelineColorBlendStateCreateInfo color_blend_info{};
        color_blend_info
                .setAttachmentCount(1)
                .setPAttachments(&color_blend_attachment);
        pipeline_info.setPColorBlendState(&color_blend_info);

        std::array dynamic_states{ vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamic_state_info{};
        dynamic_state_info
                .setDynamicStateCount(static_cast<uint32_t>(dynamic_states.size()))
                .setPDynamicStates(dynamic_states.data());
        pipeline_info.setPDynamicState(&dynamic_state_info);

        pipeline_info
                .setLayout(pipeline_layout)
                .setRenderPass(render_pass);

        vk::Result result;
        std::tie(result, pipeline) = device.createGraphicsPipeline(nullptr, pipeline_info);
        CHECK(result, "Pipeline cannot be created.");
        return RETURN_TYPE();
}

RETURN_TYPE vulkan_display::create_image_semaphores()
{
        vk::SemaphoreCreateInfo semaphore_info;

        image_semaphores.resize(transfer_image_count);

        for (auto& image_semaphores : image_semaphores) {
                CHECKED_ASSIGN(image_semaphores.image_acquired, device.createSemaphore(semaphore_info));
                CHECKED_ASSIGN(image_semaphores.image_rendered, device.createSemaphore(semaphore_info));
        }

        return RETURN_TYPE();
}

RETURN_TYPE vulkan_display::create_command_pool() {
        vk::CommandPoolCreateInfo pool_info{};
        using bits = vk::CommandPoolCreateFlagBits;
        pool_info
                .setQueueFamilyIndex(context.get_queue_familt_index())
                .setFlags(bits::eTransient | bits::eResetCommandBuffer);
        CHECKED_ASSIGN(command_pool, device.createCommandPool(pool_info));
        return RETURN_TYPE();
}

RETURN_TYPE vulkan_display::create_command_buffers() {
        vk::CommandBufferAllocateInfo allocate_info{};
        allocate_info
                .setCommandPool(command_pool)
                .setLevel(vk::CommandBufferLevel::ePrimary)
                .setCommandBufferCount(static_cast<uint32_t>(transfer_image_count));
        CHECKED_ASSIGN(command_buffers, device.allocateCommandBuffers(allocate_info));
        return RETURN_TYPE();
}

RETURN_TYPE vulkan_display::allocate_description_sets() {
        assert(transfer_image_count != 0);
        assert(descriptor_set_layout);
        vk::DescriptorPoolSize descriptor_sizes{};
        descriptor_sizes
                .setType(vk::DescriptorType::eCombinedImageSampler)
                .setDescriptorCount(transfer_image_count);
        vk::DescriptorPoolCreateInfo pool_info{};
        pool_info
                .setPoolSizeCount(1)
                .setPPoolSizes(&descriptor_sizes)
                .setMaxSets(transfer_image_count);
        CHECKED_ASSIGN(descriptor_pool, device.createDescriptorPool(pool_info));

        std::vector<vk::DescriptorSetLayout> layouts(transfer_image_count, descriptor_set_layout);

        vk::DescriptorSetAllocateInfo allocate_info;
        allocate_info
                .setDescriptorPool(descriptor_pool)
                .setDescriptorSetCount(static_cast<uint32_t>(layouts.size()))
                .setPSetLayouts(layouts.data());

        CHECKED_ASSIGN(descriptor_sets, device.allocateDescriptorSets(allocate_info));

        return RETURN_TYPE();
}

RETURN_TYPE vulkan_display::init(vulkan_instance&& instance, VkSurfaceKHR surface, uint32_t transfer_image_count,
        window_changed_callback* window, uint32_t gpu_index) {
        // Order of following calls is important
        assert(surface);
        this->window = window;
        this->transfer_image_count = transfer_image_count;
        this->filled_img_max_count = (transfer_image_count + 1) / 2;
        auto window_parameters = window->get_window_parameters();
        PASS_RESULT(context.init(std::move(instance), surface, window_parameters, gpu_index));
        device = context.get_device();
        PASS_RESULT(create_shader(vertex_shader, "shaders/vert.spv", device));
        PASS_RESULT(create_shader(fragment_shader, "shaders/frag.spv", device));
        PASS_RESULT(create_render_pass());
        context.create_framebuffers(render_pass);
        PASS_RESULT(create_texture_sampler());
        PASS_RESULT(create_graphics_pipeline());
        PASS_RESULT(create_command_pool());
        PASS_RESULT(create_command_buffers());
        PASS_RESULT(create_image_semaphores());
        PASS_RESULT(allocate_description_sets());

        transfer_images.reserve(transfer_image_count);
        auto[lock, deque] = available_img_queue.get_underlying_deque();
        for (uint32_t i = 0; i < transfer_image_count; i++) {
                transfer_images.emplace_back(device, i);
                //push_front - discarded images should be pushed at the front,
                // because they will not wait in the function device.waitForFences
                deque.push_front(&transfer_images.back());
        }
        return RETURN_TYPE();
}

RETURN_TYPE vulkan_display::destroy() {
        if (!destroyed) {
                destroyed = true;
                if (device) {
                        PASS_RESULT(device.waitIdle());
                        device.destroy(descriptor_pool);

                        for (auto& image : transfer_images) {
                                PASS_RESULT(image.destroy(device));
                        }
                        device.destroy(command_pool);
                        device.destroy(render_pass);
                        device.destroy(fragment_shader);
                        device.destroy(vertex_shader);
                        for (auto& image_semaphores : image_semaphores) {
                                device.destroy(image_semaphores.image_acquired);
                                device.destroy(image_semaphores.image_rendered);
                        }
                        device.destroy(pipeline);
                        device.destroy(pipeline_layout);
                        device.destroy(descriptor_set_layout);
                        device.destroy(sampler);
                }
                context.destroy();
        }
        return RETURN_TYPE();
}

RETURN_TYPE vulkan_display::record_graphics_commands(transfer_image& transfer_image, uint32_t swapchain_image_id) {
        vk::CommandBuffer& cmd_buffer = command_buffers[transfer_image.get_id()];
        cmd_buffer.reset(vk::CommandBufferResetFlags{});

        vk::CommandBufferBeginInfo begin_info{};
        begin_info.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        PASS_RESULT(cmd_buffer.begin(begin_info));

        auto render_begin_memory_barrier = transfer_image.create_memory_barrier(
                vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead);
        cmd_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eFragmentShader,
                vk::DependencyFlagBits::eByRegion, nullptr, nullptr, render_begin_memory_barrier);

        vk::RenderPassBeginInfo render_pass_begin_info;
        render_pass_begin_info
                .setRenderPass(render_pass)
                .setRenderArea(vk::Rect2D{ {0,0}, context.get_window_size() })
                .setClearValueCount(1)
                .setPClearValues(&clear_color)
                .setFramebuffer(context.get_framebuffer(swapchain_image_id));
        cmd_buffer.beginRenderPass(render_pass_begin_info, vk::SubpassContents::eInline);

        cmd_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

        cmd_buffer.setScissor(0, scissor);
        cmd_buffer.setViewport(0, viewport);
        cmd_buffer.pushConstants(pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(render_area), &render_area);
        cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                pipeline_layout, 0, descriptor_sets[transfer_image.get_id()], nullptr);
        cmd_buffer.draw(6, 1, 0, 0);

        cmd_buffer.endRenderPass();

        auto render_end_memory_barrier = transfer_image.create_memory_barrier(
                vk::ImageLayout::eGeneral, vk::AccessFlagBits::eHostWrite | vk::AccessFlagBits::eHostRead);
        cmd_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eHost,
                vk::DependencyFlagBits::eByRegion, nullptr, nullptr, render_end_memory_barrier);

        PASS_RESULT(cmd_buffer.end());

        return RETURN_TYPE();
}

RETURN_TYPE vulkan_display::acquire_image(image& result, image_description description) {

        transfer_image& transfer_image = acquire_transfer_image(available_img_queue, 
                filled_img_queue, filled_img_max_count);
        assert(transfer_image.get_id() != transfer_image::NO_ID);
        {
                std::unique_lock device_lock(device_mutex, std::defer_lock);
                if (transfer_image.fence_set) {
                        device_lock.lock();
                        CHECK(device.waitForFences(transfer_image.is_available_fence, VK_TRUE, UINT64_MAX),
                                "Waiting for fence failed.");
                }

                if (transfer_image.get_description() != description) {
                        if (!device_lock.owns_lock()) {
                                device_lock.lock();
                        }
                        //todo another formats
                        transfer_image.create(device, context.get_gpu(), description);
                }
        }
        result = image{ transfer_image };
        return RETURN_TYPE();
}

RETURN_TYPE vulkan_display::copy_and_queue_image(std::byte* frame, image_description description) {
        image image;
        acquire_image(image, description);
        memcpy(image.get_memory_ptr(), frame, image.get_size().height * image.get_row_pitch());
        queue_image(image);
        return RETURN_TYPE();
}

RETURN_TYPE vulkan_display::queue_image(image image) {
        filled_img_queue.push(image);
        return RETURN_TYPE();
}

RETURN_TYPE vulkan_display::display_queued_image() {
        auto window_parameters = window->get_window_parameters();
        if (window_parameters.width * window_parameters.height == 0) {
                auto image = filled_img_queue.try_pop();
                if (image.has_value()) {
                        discard_image(*image);
                }
                return RETURN_TYPE();
        }

        auto image = filled_img_queue.pop();
        if (!image.get_transfer_image()) {
                return RETURN_TYPE();
        }

        image.preprocess();

        transfer_image& transfer_image = *image.get_transfer_image();

        auto& semaphores = image_semaphores[transfer_image.get_id()];

        uint32_t swapchain_image_id = 0;
        std::unique_lock lock(device_mutex);
        if (transfer_image.get_description() != current_image_description) {
                current_image_description = transfer_image.get_description();
                auto parameters = context.get_window_parameters();
                update_render_area_viewport_scissor(render_area, viewport, scissor,
                        { parameters.width, parameters.height }, current_image_description.size);
        }
        PASS_RESULT(context.acquire_next_swapchain_image(swapchain_image_id, semaphores.image_acquired));
        while (swapchain_image_id == SWAPCHAIN_IMAGE_OUT_OF_DATE) {
                window_parameters = window->get_window_parameters();
                if (window_parameters.width * window_parameters.height == 0) {
                        // window is minimalised
                        auto image = filled_img_queue.try_pop();
                        if (image.has_value()) {
                                discard_image(*image);
                        }
                        return RETURN_TYPE();
                }
                window_parameters_changed(window_parameters);
                PASS_RESULT(context.acquire_next_swapchain_image(swapchain_image_id, semaphores.image_acquired));
        }
        transfer_image.update_description_set(device, descriptor_sets[transfer_image.get_id()], sampler);
        lock.unlock();

        record_graphics_commands(transfer_image, swapchain_image_id);
        transfer_image.fence_set = true;
        device.resetFences(transfer_image.is_available_fence);
        std::vector<vk::PipelineStageFlags> wait_masks{ vk::PipelineStageFlagBits::eColorAttachmentOutput };
        vk::SubmitInfo submit_info{};
        submit_info
                .setCommandBufferCount(1)
                .setPCommandBuffers(&command_buffers[transfer_image.get_id()])
                .setPWaitDstStageMask(wait_masks.data())
                .setWaitSemaphoreCount(1)
                .setPWaitSemaphores(&semaphores.image_acquired)
                .setSignalSemaphoreCount(1)
                .setPSignalSemaphores(&semaphores.image_rendered);

        PASS_RESULT(context.get_queue().submit(submit_info, transfer_image.is_available_fence));

        auto swapchain = context.get_swapchain();
        vk::PresentInfoKHR present_info{};
        present_info
                .setPImageIndices(&swapchain_image_id)
                .setSwapchainCount(1)
                .setPSwapchains(&swapchain)
                .setWaitSemaphoreCount(1)
                .setPWaitSemaphores(&semaphores.image_rendered);

        auto present_result = context.get_queue().presentKHR(&present_info);
        if (present_result != vk::Result::eSuccess) {
                using res = vk::Result;
                switch (present_result) {
                        // skip recoverable errors, othervise return/throw error 
                case res::eErrorOutOfDateKHR: break;
                case res::eSuboptimalKHR: break;
                default: CHECK(false, "Error presenting image:"s + vk::to_string(present_result));
                }
        }

        available_img_queue.push(&transfer_image);
        return RETURN_TYPE();
}

RETURN_TYPE vulkan_display::window_parameters_changed(window_parameters new_parameters) {
        if (new_parameters != context.get_window_parameters() && new_parameters.width * new_parameters.height != 0) {
                context.recreate_swapchain(new_parameters, render_pass);
                update_render_area_viewport_scissor(render_area, viewport, scissor,
                        { new_parameters.width, new_parameters.height }, current_image_description.size);
        }
        return RETURN_TYPE();
}

} //namespace vulkan_display
