#pragma once
#include "vulkan_context.h"


struct image_description {
        vk::Extent2D size;
        vk::Format format;

        image_description() = default;
        image_description(vk::Extent2D size, vk::Format format) :
                size{ size }, format{ format } { }
        image_description(uint32_t width, uint32_t height, vk::Format format = vk::Format::eR8G8B8A8Srgb) :
                image_description{ vk::Extent2D{width, height}, format } { }

        bool operator==(const image_description& other) const {
                return size == other.size && format == other.format;
        }

        bool operator!=(const image_description& other) const {
                return !(*this == other);
        }
};


namespace vulkan_display_detail {

class transfer_image {
        static constexpr uint32_t NO_ID = UINT32_MAX;
        vk::DeviceMemory memory;
        vk::Image image;
        vk::ImageLayout layout;
        vk::AccessFlagBits access;

public:
        uint32_t id;
        vk::ImageView view;
        std::byte* ptr;
        image_description description;

        size_t row_pitch;

        vk::Fence is_available_fence; // is_available_fence isn't signalled when gpu uses the image

        bool update_desciptor_set;
        vk::Sampler sampler;

        RETURN_TYPE init(vk::Device device, uint32_t id);

        RETURN_TYPE create(vk::Device device, vk::PhysicalDevice gpu, image_description description);

        vk::ImageMemoryBarrier create_memory_barrier(
                vk::ImageLayout new_layout,
                vk::AccessFlagBits new_access_mask,
                uint32_t src_queue_family_index = VK_QUEUE_FAMILY_IGNORED,
                uint32_t dst_queue_family_index = VK_QUEUE_FAMILY_IGNORED);

        /// update_description_sets should be called everytime before recording the command buffer
        RETURN_TYPE update_description_set(vk::Device device, vk::DescriptorSet descriptor_set, vk::Sampler sampler);

        RETURN_TYPE destroy(vk::Device device, bool destroy_fence = true);

        transfer_image() = default;
        transfer_image(vk::Device device, uint32_t id) {
                init(device, id);
        }
};

} // vulkan_display_detail
