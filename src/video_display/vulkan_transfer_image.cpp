#include "vulkan_transfer_image.h"

using namespace vulkan_display_detail;
namespace vkd = vulkan_display;
namespace {

constexpr vk::DeviceSize add_padding(vk::DeviceSize size, vk::DeviceSize allignment) {
        vk::DeviceSize remainder = size % allignment;
        if (remainder == 0) {
                return size;
        }
        return size + allignment - remainder;
}


/**
 * Check if the required flags are present among the provided flags
 */
template<typename T>
constexpr bool flags_present(T provided_flags, T required_flags) {
        return (provided_flags & required_flags) == required_flags;
}

RETURN_TYPE get_memory_type(
        uint32_t& memory_type, uint32_t memory_type_bits,
        vk::MemoryPropertyFlags requested_properties, vk::MemoryPropertyFlags optional_properties,
        vk::PhysicalDevice gpu)
{
        uint32_t possible_memory_type = UINT32_MAX;
        auto supported_properties = gpu.getMemoryProperties();
        for (uint32_t i = 0; i < supported_properties.memoryTypeCount; i++) {
                // if i-th bit in memory_type_bits is set, than i-th memory type can be used
                bool is_type_usable = (1u << i) & memory_type_bits;
                auto& mem_type = supported_properties.memoryTypes[i];
                if (flags_present(mem_type.propertyFlags, requested_properties) && is_type_usable) {
                        if (flags_present(mem_type.propertyFlags, optional_properties)) {
                                memory_type = i;
                                return RETURN_TYPE();
                        }
                        possible_memory_type = i;
                }
        }
        if (possible_memory_type != UINT32_MAX) {
                memory_type = possible_memory_type;
                return RETURN_TYPE();
        }
        CHECK(false, "No available memory for transfer images found.");
        return RETURN_TYPE();
}

constexpr vk::ImageType image_type = vk::ImageType::e2D;
constexpr vk::ImageTiling image_tiling = vk::ImageTiling::eLinear;
const vk::ImageUsageFlags image_usage_flags = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
constexpr vk::ImageCreateFlags image_create_flags = {};
} //namespace -------------------------------------------------------------

namespace vulkan_display_detail{

RETURN_TYPE transfer_image::is_image_description_supported(bool& supported, vk::PhysicalDevice gpu, 
        vkd::image_description description)
{
        vk::ImageFormatProperties properties;
        auto result = gpu.getImageFormatProperties(
                description.format,
                image_type,
                image_tiling,
                image_usage_flags,
                image_create_flags,
                &properties);
        if (result == vk::Result::eErrorFormatNotSupported) {
                supported = false;
                return RETURN_TYPE();
        }
        CHECK(result, "Error queriing image properties:")
        supported = true
                && description.size.height <= properties.maxExtent.height
                && description.size.width <= properties.maxExtent.width;
        return RETURN_TYPE();
}

RETURN_TYPE transfer_image::init(vk::Device device, uint32_t id) {
        this->id = id;
        vk::FenceCreateInfo fence_info{ vk::FenceCreateFlagBits::eSignaled };
        CHECKED_ASSIGN(is_available_fence, device.createFence(fence_info));
        return RETURN_TYPE();
}

RETURN_TYPE transfer_image::create(vk::Device device, vk::PhysicalDevice gpu,
        vkd::image_description description)
{
        assert(id != NO_ID);
        destroy(device, false);
        
        this->view = nullptr;
        this->description = description;
        this->layout = vk::ImageLayout::ePreinitialized;
        this->access = vk::AccessFlagBits::eHostWrite | vk::AccessFlagBits::eHostRead;

        vk::ImageCreateInfo image_info;
        image_info
                .setFlags(image_create_flags)
                .setImageType(image_type)
                .setExtent(vk::Extent3D{ description.size, 1 })
                .setMipLevels(1)
                .setArrayLayers(1)
                .setFormat(description.format)
                .setTiling(image_tiling)
                .setInitialLayout(vk::ImageLayout::ePreinitialized)
                .setUsage(image_usage_flags)
                .setSharingMode(vk::SharingMode::eExclusive)
                .setSamples(vk::SampleCountFlagBits::e1);
        CHECKED_ASSIGN(image, device.createImage(image_info));

        vk::MemoryRequirements memory_requirements = device.getImageMemoryRequirements(image);
        vk::DeviceSize byte_size = add_padding(memory_requirements.size, memory_requirements.alignment);

        using mem_bits = vk::MemoryPropertyFlagBits;
        uint32_t memory_type = 0;
        PASS_RESULT(get_memory_type(memory_type, memory_requirements.memoryTypeBits,
                mem_bits::eHostVisible | mem_bits::eHostCoherent, mem_bits::eHostCached, gpu));

        vk::MemoryAllocateInfo allocInfo{ byte_size , memory_type };
        CHECKED_ASSIGN(memory, device.allocateMemory(allocInfo));

        PASS_RESULT(device.bindImageMemory(image, memory, 0));

        void* void_ptr = nullptr;
        CHECKED_ASSIGN(void_ptr, device.mapMemory(memory, 0, memory_requirements.size));
        CHECK(void_ptr != nullptr, "Image memory cannot be mapped.");
        ptr = reinterpret_cast<std::byte*>(void_ptr);

        vk::ImageSubresource subresource{ vk::ImageAspectFlagBits::eColor, 0, 0 };
        row_pitch = device.getImageSubresourceLayout(image, subresource).rowPitch;
        return RETURN_TYPE();
}

vk::ImageMemoryBarrier  transfer_image::create_memory_barrier(
        vk::ImageLayout new_layout, vk::AccessFlags new_access_mask,
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

RETURN_TYPE transfer_image::prepare_for_rendering(vk::Device device, 
        vk::DescriptorSet descriptor_set, vk::Sampler sampler, vk::SamplerYcbcrConversion conversion) 
{
        if (!view) {
                device.destroy(view);
                vk::ImageViewCreateInfo view_info = 
                        vkd::default_image_view_create_info(description.format);
                view_info.setImage(image);

                vk::SamplerYcbcrConversionInfo yCbCr_info{ conversion };
                view_info.setPNext(conversion ? &yCbCr_info : nullptr);
                CHECKED_ASSIGN(view, device.createImageView(view_info));

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
        }
        return RETURN_TYPE();
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
        return RETURN_TYPE();
}

} //vulkan_display_detail
