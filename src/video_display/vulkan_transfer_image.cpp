#include "vulkan_transfer_image.h"

using namespace vulkan_display_detail;

namespace {

vk::DeviceSize add_padding(vk::DeviceSize size, vk::DeviceSize allignment) {
        vk::DeviceSize remainder = size % allignment;
        if (remainder == 0)
                return size;
        return size + allignment - remainder;
}

RETURN_TYPE get_memory_type(
        uint32_t& memory_type, uint32_t memory_type_bits,
        vk::MemoryPropertyFlags requested_properties, vk::PhysicalDevice gpu)
{
        auto supported_properties = gpu.getMemoryProperties();
        for (uint32_t i = 0; i < supported_properties.memoryTypeCount; i++) {
                auto& mem_type = supported_properties.memoryTypes[i];
                if (((mem_type.propertyFlags & requested_properties) == requested_properties) &&
                        ((1 << i) & memory_type_bits))
                {
                        memory_type = i;
                        return RETURN_TYPE();
                }
        }
        CHECK(false, "No available memory for transfer images found.");
        return RETURN_TYPE();
}

} //namespace -------------------------------------------------------------

namespace vulkan_display_detail{

RETURN_TYPE transfer_image::init(vk::Device device, uint32_t id) {
        this->id = id;
        vk::FenceCreateInfo fence_info{ vk::FenceCreateFlagBits::eSignaled };
        CHECKED_ASSIGN(is_available_fence, device.createFence(fence_info));
}

RETURN_TYPE transfer_image::create(vk::Device device, vk::PhysicalDevice gpu,
        vulkan_display::image_description description)
{
        assert(id != NO_ID);
        destroy(device, false);

        this->description = description;
        this->layout = vk::ImageLayout::ePreinitialized;
        this->access = vk::AccessFlagBits::eHostWrite;
        this->update_desciptor_set = true;

        vk::ImageCreateInfo image_info;
        image_info
                .setImageType(vk::ImageType::e2D)
                .setExtent(vk::Extent3D{ description.size, 1 })
                .setMipLevels(1)
                .setArrayLayers(1)
                .setFormat(description.format)
                .setTiling(vk::ImageTiling::eLinear)
                .setInitialLayout(vk::ImageLayout::ePreinitialized)
                .setUsage(vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst)
                .setSharingMode(vk::SharingMode::eExclusive)
                .setSamples(vk::SampleCountFlagBits::e1);
        CHECKED_ASSIGN(image, device.createImage(image_info));

        vk::MemoryRequirements memory_requirements = device.getImageMemoryRequirements(image);
        vk::DeviceSize byte_size = add_padding(memory_requirements.size, memory_requirements.alignment);

        using mem_bits = vk::MemoryPropertyFlagBits;
        uint32_t memory_type;
        PASS_RESULT(get_memory_type(memory_type, memory_requirements.memoryTypeBits,
                mem_bits::eHostVisible | mem_bits::eHostCoherent, gpu));

        vk::MemoryAllocateInfo allocInfo{ byte_size , memory_type };
        CHECKED_ASSIGN(memory, device.allocateMemory(allocInfo));

        PASS_RESULT(device.bindImageMemory(image, memory, 0));

        void* void_ptr;
        CHECKED_ASSIGN(void_ptr, device.mapMemory(memory, 0, memory_requirements.size));
        CHECK(void_ptr != nullptr, "Image memory cannot be mapped.");
        ptr = reinterpret_cast<std::byte*>(void_ptr);

        vk::ImageViewCreateInfo view_info = vulkan_display::default_image_view_create_info(description.format);
        view_info.setImage(image);
        CHECKED_ASSIGN(view, device.createImageView(view_info));

        vk::ImageSubresource subresource{ vk::ImageAspectFlagBits::eColor, 0, 0 };
        row_pitch = device.getImageSubresourceLayout(image, subresource).rowPitch;
}

vk::ImageMemoryBarrier  transfer_image::create_memory_barrier(
        vk::ImageLayout new_layout, vk::AccessFlagBits new_access_mask,
        uint32_t src_queue_family_index, uint32_t dst_queue_family_index)
{
        vk::ImageMemoryBarrier memory_barrier{};
        memory_barrier
                .setImage(image)
                .setOldLayout(layout)
                .setNewLayout(new_layout)
                .setSrcAccessMask(access)
                .setDstAccessMask(new_access_mask)
                .setSrcQueueFamilyIndex(src_queue_family_index)
                .setDstQueueFamilyIndex(dst_queue_family_index);
        memory_barrier.subresourceRange
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setLayerCount(1)
                .setLevelCount(1);

        layout = new_layout;
        access = new_access_mask;
        return memory_barrier;
}

RETURN_TYPE transfer_image::update_description_set(vk::Device device, vk::DescriptorSet descriptor_set, vk::Sampler sampler) {
        if (update_desciptor_set || sampler != this->sampler) {
                update_desciptor_set = false;
                this->sampler = sampler;
                vk::DescriptorImageInfo description_image_info;
                description_image_info
                        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
                        .setSampler(sampler)
                        .setImageView(view);

                vk::WriteDescriptorSet descriptor_writes{};
                descriptor_writes
                        .setDstBinding(1)
                        .setDstArrayElement(0)
                        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
                        .setPImageInfo(&description_image_info)
                        .setDescriptorCount(1)
                        .setDstSet(descriptor_set);

                device.updateDescriptorSets(descriptor_writes, nullptr);
                return RETURN_TYPE();
        }
}

RETURN_TYPE transfer_image::destroy(vk::Device device, bool destroy_fence) {
        if (is_available_fence) {
                auto result = device.waitForFences(is_available_fence, true, UINT64_MAX);
                CHECK(result, "Waiting for transfer image fence failed.");
        }

        device.destroy(view);
        device.destroy(image);

        if (memory) {
                device.unmapMemory(memory);
                device.freeMemory(memory);
        }
        if (destroy_fence) {
                device.destroy(is_available_fence);
        }
}

} //vulkan_display_detail
