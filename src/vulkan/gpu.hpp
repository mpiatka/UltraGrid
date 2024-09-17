#ifndef GPU_HPP_4c855a947eb0
#define GPU_HPP_4c855a947eb0

#include "vulkan_wrapper.hpp"

struct Gpu{
        bool init();

        VkSemaphoreUniq getSemaphore();
        VkFenceUniq getFence(bool signaled);

        VkCommandPoolUniq getCmdPool(uint32_t queueFamIdx);

        VkbInstanceUniq inst;
        VkbDeviceUniq dev;

        VkQueue graphicsQueue = VK_NULL_HANDLE;

        VkQueue videoDecodeQueue = VK_NULL_HANDLE;
        uint32_t videoDecodeQueueIdx = UINT32_MAX;
};

#endif
