#include "gpu.hpp"
#include "debug.h"

#define MOD_NAME "[vulkan GPU] "

bool Gpu::init(){
	if(volkInitialize() != VK_SUCCESS){
		log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to initialize volk\n");
		return false;
	}

	vkb::InstanceBuilder builder;
	auto instance_ret = builder.set_app_name("UltraGrid")
		.request_validation_layers()
		.use_default_debug_messenger()
		.require_api_version(1, 1, 0)
		.build();

	if(!instance_ret){
		log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to create instance. Err: %s\n", instance_ret.error().message().c_str());
		return false;
	}
	inst = VkbInstanceUniq(instance_ret.value());

	volkLoadInstance(inst.get());

	vkb::PhysicalDeviceSelector devSelector(inst);
	auto phys_ret = devSelector.set_minimum_version(1, 1)
		.select();

	if(!phys_ret){
		log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to select phys dev. Err: %s\n", phys_ret.error().message().c_str());
		return false;
	}

	vkb::DeviceBuilder devBuilder(phys_ret.value());
	auto dev_ret = devBuilder.build();
	if(!dev_ret){
		log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to build dev. Err: %s\n", dev_ret.error().message().c_str());
		return false;
	}

	dev = VkbDeviceUniq(dev_ret.value());

	auto queue_ret = dev->get_queue(vkb::QueueType::graphics);
	if(!queue_ret){
		log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get queue. Err: %s\n", queue_ret.error().message().c_str());
		return false;
	}
	graphicsQueue = queue_ret.value();

	queue_ret = dev->get_queue(vkb::QueueType::present);
	if(!queue_ret){
		log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get queue. Err: %s\n", queue_ret.error().message().c_str());
		return false;
	}

	queue_ret = dev->get_queue(vkb::QueueType::present);
	if(!queue_ret){
		log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get queue. Err: %s\n", queue_ret.error().message().c_str());
		return false;
	}
	presentQueue = queue_ret.value();
	presentQueue = queue_ret.value();

	return true;
}

VkSemaphoreUniq Gpu::getSemaphore(){
	VkSemaphoreCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkSemaphore sem{};
	auto res = vkCreateSemaphore(dev.get(), &createInfo, nullptr, &sem);

	if(res < 0)
		return {};

	return VkSemaphoreUniq(dev.get(), sem);
}

VkFenceUniq Gpu::getFence(bool signaled){
	VkFenceCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	if(signaled){
		createInfo.flags |= VK_FENCE_CREATE_SIGNALED_BIT;
	}

	VkFence fence{};
	auto res = vkCreateFence(dev.get(), &createInfo, nullptr, &fence);

	if(res < 0)
		return {};

	return VkFenceUniq(dev.get(), fence);
}
