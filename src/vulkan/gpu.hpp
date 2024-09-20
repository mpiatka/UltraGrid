#ifndef GPU_HPP_4c855a947eb0
#define GPU_HPP_4c855a947eb0

#include "vulkan_wrapper.hpp"

struct AllocatedMemory{
        VkDeviceMemoryUniq deviceMemory;
        uint32_t memType = UINT32_MAX;
        VkDeviceSize size = 0;
        void *mapPtr = nullptr;
};

struct Gpu{
        bool init();

        VkSemaphoreUniq getSemaphore();
        VkFenceUniq getFence(bool signaled);
        VkCommandPoolUniq getCmdPool(uint32_t queueFamIdx);
        AllocatedMemory allocateMem(VkDeviceSize size, VkDeviceSize alignment,
                VkMemoryPropertyFlags flags, uint32_t allowedTypes);

        VkbInstanceUniq inst;
        VkbDeviceUniq dev;

        VkQueue graphicsQueue = VK_NULL_HANDLE;

        VkQueue videoDecodeQueue = VK_NULL_HANDLE;
        uint32_t videoDecodeQueueIdx = UINT32_MAX;
};

#endif
