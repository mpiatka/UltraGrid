#ifndef UG_HOLE_PUNCH_MESSAGE_HPP
#define UG_HOLE_PUNCH_MESSAGE_HPP

#include <string_view>
#include <functional>
#include <asio.hpp>

class Message{
public:
	Message() = default;

	void async_readMsg(asio::ip::tcp::socket& socket,
			std::function<void(bool)> onComplete);

	std::string_view getStr() const;

	void setStr(std::string_view msg);
	void async_sendMsg(asio::ip::tcp::socket& socket,
			std::function<void(bool)> onComplete);

	size_t getSize() const { return size; }
	bool empty() const { return size == 0; }
	bool isSendingNow() const { return sendingNow; }
	void clear();

private:
	void async_readBody(asio::ip::tcp::socket& socket,
			std::function<void(bool)> onComplete,
			const std::error_code& ec, size_t readLen);

	void async_readBodyComplete(std::function<void(bool)> onComplete,
			const std::error_code& ec, size_t readLen);

	void async_sendComplete(std::function<void(bool)> onComplete,
			const std::error_code& ec, size_t sentLen);

	size_t size = 0;

	bool sendingNow = false;

	static constexpr size_t headerSize = 5;
	static constexpr size_t bodySize = 2048;
	char data[headerSize + bodySize];
};

#endif
