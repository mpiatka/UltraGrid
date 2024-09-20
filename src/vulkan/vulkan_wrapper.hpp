#ifndef VULKAN_WRAPPER_HPP_d28b8f718c3a
#define VULKAN_WRAPPER_HPP_d28b8f718c3a

#include <utility>
#include <volk.h>
#include <VkBootstrap.h>

template<typename T, auto deleter>
class UniqHandle{
public:
	UniqHandle() = default;
	UniqHandle(T &&val) : val(std::move(val)), initialized(true) {  }
	UniqHandle(const T &val) : val(val), initialized(true) {  }
	UniqHandle(const UniqHandle &) = delete;
	UniqHandle(UniqHandle &&o){
		std::swap(val, o.val);
		std::swap(initialized, o.initialized);
	};
	~UniqHandle() { if(initialized) deleter(val); }

	UniqHandle& operator=(const UniqHandle&) = delete;
	UniqHandle& operator=(UniqHandle&& o){
		std::swap(o.val, val);
		std::swap(o.initialized, initialized);
		return *this;
	}

	T* operator->() { return &val;}

	T& get() { return val; }
	operator T&() { return val; }
private:
	T val;
	bool initialized = false;
};

template<typename parentT, typename T, auto deleter>
class VkUniqueHandle{
public:
	VkUniqueHandle() = default;
	VkUniqueHandle(parentT parent, T val): parent(parent), val(val) {  }
	VkUniqueHandle(const VkUniqueHandle&) = delete;
	VkUniqueHandle(VkUniqueHandle&& o){
		std::swap(parent, o.parent);
		std::swap(val, o.val);
	}
	~VkUniqueHandle() { if(*this) (*deleter)(parent, val, nullptr); }

	VkUniqueHandle& operator=(const VkUniqueHandle&) = delete;
	VkUniqueHandle& operator=(VkUniqueHandle&& o){
		std::swap(o.parent, parent);
		std::swap(o.val, val);
		return *this;
	}

	operator bool() const { return parent && val; }
	operator T() const { return val; }

	T get() { return val; }
private:
	parentT parent = VK_NULL_HANDLE;
	T val = VK_NULL_HANDLE;
};

using VkbInstanceUniq = UniqHandle<vkb::Instance, vkb::destroy_instance>;
using VkbDeviceUniq = UniqHandle<vkb::Device, vkb::destroy_device>;
using VkbSwapchainUniq = UniqHandle<vkb::Swapchain, vkb::destroy_swapchain>;

using VkSurfaceUniq = VkUniqueHandle<VkInstance, VkSurfaceKHR, &vkDestroySurfaceKHR>;
using VkPipelineUniq = VkUniqueHandle<VkDevice, VkPipeline, &vkDestroyPipeline>;
using VkSemaphoreUniq = VkUniqueHandle<VkDevice, VkSemaphore, &vkDestroySemaphore>;
using VkFenceUniq = VkUniqueHandle<VkDevice, VkFence, &vkDestroyFence>;
using VkImageUniq = VkUniqueHandle<VkDevice, VkImage, &vkDestroyImage>;
using VkImageViewUniq = VkUniqueHandle<VkDevice, VkImageView, &vkDestroyImageView>;
using VkCommandPoolUniq = VkUniqueHandle<VkDevice, VkCommandPool, &vkDestroyCommandPool>;
using VkPipelineUniq = VkUniqueHandle<VkDevice, VkPipeline, &vkDestroyPipeline>;
using VkPipelineLayoutUniq = VkUniqueHandle<VkDevice, VkPipelineLayout, &vkDestroyPipelineLayout>;
using VkShaderModuleUniq = VkUniqueHandle<VkDevice, VkShaderModule, &vkDestroyShaderModule>;
using VkDescriptorPoolUniq = VkUniqueHandle<VkDevice, VkDescriptorPool, &vkDestroyDescriptorPool>;
using VkDescriptorSetLayoutUniq = VkUniqueHandle<VkDevice, VkDescriptorSetLayout, &vkDestroyDescriptorSetLayout>;
using VkBufferUniq = VkUniqueHandle<VkDevice, VkBuffer, &vkDestroyBuffer>;
using VkDeviceMemoryUniq = VkUniqueHandle<VkDevice, VkDeviceMemory, &vkFreeMemory>;

#endif
