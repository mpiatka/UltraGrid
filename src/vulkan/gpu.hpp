#ifndef GPU_HPP_4c855a947eb0
#define GPU_HPP_4c855a947eb0

#include "vulkan_wrapper.hpp"

struct Gpu{
	bool init();

	VkSemaphoreUniq getSemaphore();
	VkFenceUniq getFence(bool signaled);

	VkbInstanceUniq inst;
	VkbDeviceUniq dev;

	VkQueue graphicsQueue;
	VkQueue presentQueue;
};

#endif
