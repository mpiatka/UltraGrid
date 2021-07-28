#include "vulkan_display.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>

using namespace vulkan_display_detail;

namespace {
        vk::DeviceSize add_padding(vk::DeviceSize size, vk::DeviceSize allignment) {
                vk::DeviceSize remainder = size % allignment;
                if (remainder == 0)
                        return size;
                return size + allignment - remainder;
        }

        RETURN_VAL create_shader(vk::ShaderModule& shader,
                const std::filesystem::path& file_path,
                const vk::Device& device)
        {
                std::ifstream file(file_path, std::ios::binary);
                CHECK(file.is_open(), "Failed to open file:"s + file_path.string());
                auto size = std::filesystem::file_size(file_path);
                assert(size % 4 == 0);
                std::vector<std::uint32_t> shader_code(size / 4);
                file.read(reinterpret_cast<char*>(shader_code.data()), size);
                CHECK(file.good(), "Error reading from file:"s + file_path.string());

                vk::ShaderModuleCreateInfo shader_info;
                shader_info.setCode(shader_code);
                CHECKED_ASSIGN(shader, device.createShaderModule(shader_info));
                return RETURN_VAL();
        }

        RETURN_VAL get_memory_type(
                uint32_t& memory_type,
                uint32_t memory_type_bits,
                vk::MemoryPropertyFlags requested_properties,
                vk::PhysicalDevice gpu)
        {
                auto supported_properties = gpu.getMemoryProperties();
                for (uint32_t i = 0; i < supported_properties.memoryTypeCount; i++) {
                        auto& mem_type = supported_properties.memoryTypes[i];
                        if (((mem_type.propertyFlags & requested_properties) == requested_properties) &&
                                ((1 << i) & memory_type_bits))
                        {
                                memory_type = i;
                                return RETURN_VAL();
                        }
                }
                CHECK(false, "No available memory for transfer images found.");
        }

        RETURN_VAL transport_image(std::byte* dest, std::byte* source,
                size_t image_width, size_t image_height, // image width and height should be in pixels
                vk::Format from_format, size_t row_pitch)
        {
                using f = vk::Format;
                switch (from_format) {
                case f::eR8G8B8A8Srgb: {
                        auto row_size = image_width * 4;
                        assert(row_size <= row_pitch);
                        for (size_t row = 0; row < image_height; row++) {
                                memcpy(dest, source, row_size);
                                source += row_size;
                                dest += row_pitch;
                        }
                        break;
                }
                case f::eR8G8B8Srgb: {
                        auto row_size = image_width * 4;
                        assert(row_size <= row_pitch);
                        auto row_padding = row_pitch - row_size;
                        for (size_t row = 0; row < image_height; row++) {
                                for (size_t col = 0; col < image_width; col++) {
                                        memcpy(dest, source, 3);
                                        dest += 4;
                                        source += 3;
                                }
                                dest += row_padding;
                        }
                        break;
                }
                default:
                        CHECK(false, "Unsupported picture format");
                }
                return RETURN_VAL();
        }

} //namespace

vk::ImageMemoryBarrier  Vulkan_display::create_memory_barrier( Vulkan_display::Transfer_image& image,
        vk::ImageLayout new_layout, vk::AccessFlagBits new_access_mask,
        uint32_t src_queue_family_index, uint32_t dst_queue_family_index)
{
        vk::ImageMemoryBarrier memory_barrier{};
        memory_barrier
                .setImage(image.image)
                .setOldLayout(image.layout)
                .setNewLayout(new_layout)
                .setSrcAccessMask(image.access)
                .setDstAccessMask(new_access_mask)
                .setSrcQueueFamilyIndex(src_queue_family_index)
                .setDstQueueFamilyIndex(dst_queue_family_index);
        memory_barrier.subresourceRange
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setLayerCount(1)
                .setLevelCount(1);

        image.layout = new_layout;
        image.access = new_access_mask;
        return memory_barrier;
}

RETURN_VAL Vulkan_display::create_texture_sampler()
{
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
        return RETURN_VAL();
}

RETURN_VAL Vulkan_display::create_render_pass() {
        vk::RenderPassCreateInfo render_pass_info;

        vk::AttachmentDescription color_attachment;
        color_attachment
                .setFormat(context.swapchain_atributes.format.format)
                .setSamples(vk::SampleCountFlagBits::e1)
                .setLoadOp(vk::AttachmentLoadOp::eClear)
                .setStoreOp(vk::AttachmentStoreOp::eStore)
                .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                .setInitialLayout(vk::ImageLayout::eUndefined)
                .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);
        render_pass_info.setAttachments(color_attachment);

        vk::AttachmentReference attachment_reference;
        attachment_reference
                .setAttachment(0)
                .setLayout(vk::ImageLayout::eColorAttachmentOptimal);
        vk::SubpassDescription subpass;
        subpass
                .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
                .setColorAttachments(attachment_reference);
        render_pass_info.setSubpasses(subpass);

        vk::SubpassDependency subpass_dependency{};
        subpass_dependency
                .setSrcSubpass(VK_SUBPASS_EXTERNAL)
                .setDstSubpass(0)
                .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
                .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
                .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
        render_pass_info.setDependencies(subpass_dependency);

        render_pass = device.createRenderPass(render_pass_info);

        vk::ClearColorValue clear_color_value{};
        clear_color_value.setFloat32({ 0.01f, 0.01f, 0.01f, 1.0f });
        clear_color.setColor(clear_color_value);

        return RETURN_VAL();
}

RETURN_VAL Vulkan_display::create_descriptor_set_layout() {
        vk::DescriptorSetLayoutBinding descriptor_set_layout_bindings;
        descriptor_set_layout_bindings
                .setBinding(1)
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
                .setStageFlags(vk::ShaderStageFlagBits::eFragment)
                .setImmutableSamplers(sampler);

        vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_info{};
        descriptor_set_layout_info
                .setBindings(descriptor_set_layout_bindings);
        CHECKED_ASSIGN(descriptor_set_layout,
                device.createDescriptorSetLayout(descriptor_set_layout_info));
}

RETURN_VAL Vulkan_display::create_graphics_pipeline() {
        create_descriptor_set_layout();

        vk::PipelineLayoutCreateInfo pipeline_layout_info{};

        vk::PushConstantRange push_constants;
        push_constants
                .setOffset(0)
                .setSize(sizeof(render_area))
                .setStageFlags(vk::ShaderStageFlagBits::eFragment);
        pipeline_layout_info.setPushConstantRanges(push_constants);

        pipeline_layout_info.setSetLayouts(descriptor_set_layout);
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
        pipeline_info.setStages(shader_stages_infos);

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
        color_blend_info.setAttachments(color_blend_attachment);
        pipeline_info.setPColorBlendState(&color_blend_info);

        std::array dynamic_states{ vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamic_state_info{};
        dynamic_state_info.setDynamicStates(dynamic_states);
        pipeline_info.setPDynamicState(&dynamic_state_info);

        pipeline_info
                .setLayout(pipeline_layout)
                .setRenderPass(render_pass);

        vk::Result result;
        std::tie(result, pipeline) = device.createGraphicsPipeline(VK_NULL_HANDLE, pipeline_info);
        CHECK(result, "Pipeline cannot be created.");
        return RETURN_VAL();
}

RETURN_VAL Vulkan_display::create_concurrent_paths()
{
        vk::SemaphoreCreateInfo semaphore_info;

        vk::FenceCreateInfo fence_info{};
        fence_info.setFlags(vk::FenceCreateFlagBits::eSignaled);

        concurent_paths.resize(concurent_paths_count);

        for (auto& path : concurent_paths) {
                CHECKED_ASSIGN(path.image_acquired_semaphore, device.createSemaphore(semaphore_info));
                CHECKED_ASSIGN(path.image_rendered_semaphore, device.createSemaphore(semaphore_info));
                CHECKED_ASSIGN(path.path_available_fence, device.createFence(fence_info));
        }

        return RETURN_VAL();
}

RETURN_VAL Vulkan_display::create_transfer_images(uint32_t width, uint32_t height, vk::Format format) {
        transfer_image_size = vk::Extent2D{ width, height };
        transfer_image_format = format;
        vk::ImageCreateInfo image_info;
        image_info
                .setImageType(vk::ImageType::e2D)
                .setExtent(vk::Extent3D{ width, height, 1 })
                .setMipLevels(1)
                .setArrayLayers(1)
                .setFormat(format)
                .setTiling(vk::ImageTiling::eLinear)
                .setInitialLayout(vk::ImageLayout::ePreinitialized)
                .setUsage(vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst)
                .setSharingMode(vk::SharingMode::eExclusive)
                .setSamples(vk::SampleCountFlagBits::e1);

        transfer_images.resize(concurent_paths_count);
        for (auto& image : transfer_images) {
                CHECKED_ASSIGN(image.image, device.createImage(image_info));
                image.layout = vk::ImageLayout::ePreinitialized;
                image.access = vk::AccessFlagBits::eHostWrite;
        }

        vk::MemoryRequirements memory_requirements = device.getImageMemoryRequirements(transfer_images[0].image);
        using mem_bits = vk::MemoryPropertyFlagBits;
        uint32_t memory_type;
        PASS_RESULT(get_memory_type(memory_type, memory_requirements.memoryTypeBits,
                mem_bits::eHostVisible | mem_bits::eHostCoherent, context.gpu));

        vk::DeviceSize image_size = add_padding(memory_requirements.size, memory_requirements.alignment);
        vk::MemoryAllocateInfo allocInfo{};
        allocInfo
                .setAllocationSize(image_size * concurent_paths_count)
                .setMemoryTypeIndex(memory_type);
        CHECKED_ASSIGN(transfer_image_memory, device.allocateMemory(allocInfo));

        auto ptr = device.mapMemory(transfer_image_memory, 0, image_size * concurent_paths_count);
        CHECK(ptr != nullptr, "Image memory cannot be mapped.");

        for (size_t i = 0; i < transfer_images.size(); i++) {
                device.bindImageMemory(transfer_images[i].image, transfer_image_memory, i * image_size);
                transfer_images[i].ptr = reinterpret_cast<std::byte*>(ptr) + i * image_size;
        }
        vk::ImageViewCreateInfo view_info = default_image_view_create_info(format);
        for (auto& image : transfer_images) {
                view_info.setImage(image.image);
                CHECKED_ASSIGN(image.view, device.createImageView(view_info));
        }

        vk::ImageSubresource subresource;
        subresource
                .setAspectMask(vk::ImageAspectFlagBits::eColor);
        auto image_subresource_layout = device.getImageSubresourceLayout(transfer_images[0].image, subresource);

        transfer_image_row_pitch = image_subresource_layout.rowPitch;
        transfer_image_byte_size = memory_requirements.size;
        return RETURN_VAL();
}

void Vulkan_display::destroy_transfer_images() {
        for (auto& image : transfer_images) {
                device.destroy(image.view);
                device.destroy(image.image);
        }
        if (transfer_image_memory) {
                device.unmapMemory(transfer_image_memory);
                device.freeMemory(transfer_image_memory);
        }
}

RETURN_VAL Vulkan_display::create_command_pool() {
        vk::CommandPoolCreateInfo pool_info{};
        using bits = vk::CommandPoolCreateFlagBits;
        pool_info
                .setQueueFamilyIndex(context.queue_family_index)
                .setFlags(bits::eTransient | bits::eResetCommandBuffer);
        CHECKED_ASSIGN(command_pool, device.createCommandPool(pool_info));
        return RETURN_VAL();
}

RETURN_VAL Vulkan_display::create_command_buffers() {
        vk::CommandBufferAllocateInfo allocate_info{};
        allocate_info
                .setCommandPool(command_pool)
                .setLevel(vk::CommandBufferLevel::ePrimary)
                .setCommandBufferCount(static_cast<uint32_t>(concurent_paths_count));
        CHECKED_ASSIGN(command_buffers, device.allocateCommandBuffers(allocate_info));
        return RETURN_VAL();
}

RETURN_VAL Vulkan_display::create_descriptor_pool()
{
        assert(concurent_paths_count != 0);
        std::array<vk::DescriptorPoolSize, 1> descriptor_sizes{};
        descriptor_sizes[0]
                .setType(vk::DescriptorType::eCombinedImageSampler)
                .setDescriptorCount(concurent_paths_count);
        vk::DescriptorPoolCreateInfo pool_info{};
        pool_info
                .setPoolSizes(descriptor_sizes)
                .setMaxSets(concurent_paths_count);
        CHECKED_ASSIGN(descriptor_pool, device.createDescriptorPool(pool_info));
        return RETURN_VAL();
}

RETURN_VAL Vulkan_display::update_render_area() {
        vk::Extent2D wnd_size = context.window_size;
        vk::Extent2D img_size = transfer_image_size;

        double wnd_aspect = static_cast<double>(wnd_size.width) / wnd_size.height;
        double img_aspect = static_cast<double>(img_size.width) / img_size.height;

        if (wnd_aspect > img_aspect) {
                render_area.height = wnd_size.height;
                render_area.width = static_cast<uint32_t>(std::round(wnd_size.height * img_aspect));
                render_area.x = (wnd_size.width - render_area.width) / 2;
                render_area.y = 0;
        }
        else {
                render_area.width = wnd_size.width;
                render_area.height = static_cast<uint32_t>(std::round(wnd_size.width / img_aspect));
                render_area.x = 0;
                render_area.y = (wnd_size.height - render_area.height) / 2;

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
        return RETURN_VAL();
}

RETURN_VAL Vulkan_display::create_description_sets() {
        assert(descriptor_pool);
        std::vector<vk::DescriptorSetLayout> layouts(concurent_paths_count, descriptor_set_layout);
        vk::DescriptorSetAllocateInfo allocate_info;
        allocate_info
                .setDescriptorSetCount(concurent_paths_count)
                .setDescriptorPool(descriptor_pool)
                .setSetLayouts(layouts);

        CHECKED_ASSIGN(descriptor_sets, device.allocateDescriptorSets(allocate_info));

        vk::DescriptorImageInfo description_image_info;
        description_image_info
                .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
                .setSampler(sampler);

        std::array<vk::WriteDescriptorSet, 1> descriptor_writes{};
        descriptor_writes[0]
                .setDstBinding(1)
                .setDstArrayElement(0)
                .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
                .setDescriptorCount(1)
                .setPImageInfo(&description_image_info);

        for (unsigned i = 0; i < concurent_paths_count; i++) {
                description_image_info.setImageView(transfer_images[i].view);
                descriptor_writes[0].setDstSet(descriptor_sets[i]);
                device.updateDescriptorSets(descriptor_writes, nullptr);
        }

        return RETURN_VAL();
}

RETURN_VAL Vulkan_display::init(VkSurfaceKHR surface, Window_inteface* window) {
        // Order of following calls is important
        assert(surface != VK_NULL_HANDLE);
        this->window = window;
        auto window_parameters = window->get_window_parameters();
        PASS_RESULT(context.init(surface, window_parameters));
        device = context.device;
        PASS_RESULT(create_shader(vertex_shader, "shaders/vert.spv", device));
        PASS_RESULT(create_shader(fragment_shader, "shaders/frag.spv", device));
        PASS_RESULT(create_render_pass());
        context.create_framebuffers(render_pass);
        PASS_RESULT(create_texture_sampler());
        PASS_RESULT(create_graphics_pipeline());
        PASS_RESULT(create_command_pool());
        PASS_RESULT(create_command_buffers());
        PASS_RESULT(create_concurrent_paths());
        PASS_RESULT(create_descriptor_pool());
        return RETURN_VAL();
}

Vulkan_display::~Vulkan_display() {
        device.waitIdle();
        device.destroy(descriptor_pool);

        destroy_transfer_images();
        device.destroy(command_pool);
        device.destroy(render_pass);
        device.destroy(fragment_shader);
        device.destroy(vertex_shader);
        for (auto& path : concurent_paths) {
                device.destroy(path.image_acquired_semaphore);
                device.destroy(path.image_rendered_semaphore);
                device.destroy(path.path_available_fence);
        }
        device.destroy(pipeline);
        device.destroy(pipeline_layout);
        device.destroy(descriptor_set_layout);
        device.destroy(sampler);
}

RETURN_VAL Vulkan_display::record_graphics_commands(unsigned current_path_id, uint32_t image_index) {
        vk::CommandBuffer& cmd_buffer = command_buffers[current_path_id];
        cmd_buffer.reset();

        vk::CommandBufferBeginInfo begin_info{};
        begin_info.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        PASS_RESULT(cmd_buffer.begin(begin_info));

        auto render_begin_memory_barrier = create_memory_barrier(transfer_images[current_path_id],
                vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead);
        cmd_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eFragmentShader,
                vk::DependencyFlagBits::eByRegion, nullptr, nullptr, render_begin_memory_barrier);

        vk::RenderPassBeginInfo render_pass_begin_info;
        render_pass_begin_info
                .setRenderPass(render_pass)
                .setRenderArea(vk::Rect2D{ {0,0}, context.window_size })
                .setClearValues(clear_color)
                .setFramebuffer(context.get_framebuffer(image_index));
        cmd_buffer.beginRenderPass(render_pass_begin_info, vk::SubpassContents::eInline);

        cmd_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

        cmd_buffer.setScissor(0, scissor);
        cmd_buffer.setViewport(0, viewport);
        cmd_buffer.pushConstants(pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(render_area), &render_area);
        cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                pipeline_layout, 0, descriptor_sets[current_path_id], nullptr);
        cmd_buffer.draw(6, 1, 0, 0);

        cmd_buffer.endRenderPass();

        auto render_end_memory_barrier = create_memory_barrier(transfer_images[current_path_id],
                vk::ImageLayout::eGeneral, vk::AccessFlagBits::eHostWrite);
        cmd_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eHost,
                vk::DependencyFlagBits::eByRegion, nullptr, nullptr, render_end_memory_barrier);

        PASS_RESULT(cmd_buffer.end());

        return RETURN_VAL();
}
RETURN_VAL Vulkan_display::acquire_new_image(uint32_t& image_index, const Path& path) {
        while (true) {
                auto acquired = device.acquireNextImageKHR(context.swapchain, UINT64_MAX, path.image_acquired_semaphore, nullptr, &image_index);
                if (acquired == vk::Result::eSuboptimalKHR || acquired == vk::Result::eErrorOutOfDateKHR) {
                        window_parameters_changed();
                        continue;
                }
                
                CHECK(acquired, "Next swapchain image cannot be acquired."s + vk::to_string(acquired));
                return RETURN_VAL();
        }
}

RETURN_VAL Vulkan_display::render(std::byte* frame,
        uint32_t image_width, uint32_t image_height, vk::Format format)
{
        if (vk::Extent2D{ image_width, image_height } != transfer_image_size) {
                //todo another formats
                device.waitIdle();
                device.resetDescriptorPool(descriptor_pool);
                destroy_transfer_images();
                create_transfer_images(image_width, image_height, vk::Format::eR8G8B8A8Srgb);
                create_description_sets();
                update_render_area();
        }

        Path& path = concurent_paths[current_path_id];

        CHECK(device.waitForFences(path.path_available_fence, VK_TRUE, UINT64_MAX),
                "Waiting for fence failed.");
        device.resetFences(path.path_available_fence);

        transport_image(transfer_images[current_path_id].ptr, frame, image_width, image_height,
                format, transfer_image_row_pitch);

        uint32_t image_index;
        PASS_RESULT(acquire_new_image(image_index, path));

        record_graphics_commands(current_path_id, image_index);

        std::vector<vk::PipelineStageFlags> wait_masks{ vk::PipelineStageFlagBits::eColorAttachmentOutput };
        vk::SubmitInfo submit_info{};
        submit_info
                .setCommandBuffers(command_buffers[current_path_id])
                .setWaitDstStageMask(wait_masks)
                .setWaitSemaphores(path.image_acquired_semaphore)
                .setSignalSemaphores(path.image_rendered_semaphore);

        PASS_RESULT(context.queue.submit(submit_info, path.path_available_fence));

        vk::PresentInfoKHR present_info{};
        present_info
                .setImageIndices(image_index)
                .setSwapchains(context.swapchain)
                .setWaitSemaphores(path.image_rendered_semaphore);

        auto present_result = context.queue.presentKHR(&present_info);
        if (present_result != vk::Result::eSuccess) {
                using res = vk::Result;
                switch (present_result) {
                        // skip recoverable errors, othervise return/throw error 
                        case res::eErrorOutOfDateKHR: break;
                        case res::eSuboptimalKHR: break;
                        default: CHECK(false, "Error presenting image:"s + vk::to_string(present_result));
                }
        }
        
        current_path_id++;
        current_path_id %= concurent_paths_count;

        return RETURN_VAL();
}

RETURN_VAL Vulkan_display::window_parameters_changed() {
        Window_parameters new_parameters = window->get_window_parameters();
        if (new_parameters != context.get_window_parameters() && new_parameters.width * new_parameters.height != 0) {
                context.recreate_swapchain(new_parameters, render_pass);
                update_render_area();
        }
        return RETURN_VAL();
}
