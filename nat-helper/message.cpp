#include <charconv>
#include <cassert>
#include <cstring>
#include "message.hpp"

std::string_view Message::getStr() const{
	return std::string_view(data + headerSize, size);
}

void Message::async_readMsg(asio::ip::tcp::socket& socket,
		std::function<void(bool)> onComplete)
{
	using namespace std::placeholders;
	asio::async_read(socket,
			asio::buffer(data, headerSize),
			std::bind(&Message::async_readBody,
				this, std::ref(socket), onComplete, _1, _2));
}

void Message::setStr(std::string_view msg){
	assert(!sendingNow);
	size = std::min(bodySize, msg.size());

	memset(data, ' ', headerSize);
	std::to_chars(data, data + headerSize, size);

	memcpy(data + headerSize, msg.data(), size);
}

void Message::async_sendMsg(asio::ip::tcp::socket& socket,
		std::function<void(bool)> onComplete)
{
	using namespace std::placeholders;
	sendingNow = true;
	asio::async_write(socket,
			asio::buffer(data, headerSize + size),
			std::bind(&Message::async_sendComplete,
				this, onComplete, _1, _2));
}

void Message::async_readBody(asio::ip::tcp::socket& socket,
		std::function<void(bool)> onComplete,
		const std::error_code& ec, [[maybe_unused]] size_t readLen)
{
	if(ec){
		onComplete(false);
		return;
	}

	assert(readLen == headerSize);
	size_t expectedSize = 0;

	auto res = std::from_chars(data, data + headerSize, expectedSize);
	if(res.ec != std::errc()){
		onComplete(false);
	}

	using namespace std::placeholders;
	asio::async_read(socket,
			asio::buffer(data + headerSize, expectedSize),
			std::bind(&Message::async_readBodyComplete,
				this, onComplete, _1, _2));

}

void Message::async_readBodyComplete(std::function<void(bool)> onComplete,
		const std::error_code& ec, size_t readLen)
{
	if(ec){
		onComplete(false);
		return;
	}

	size = readLen;

	onComplete(true);
}

void Message::async_sendComplete(std::function<void(bool)> onComplete,
		const std::error_code& ec, [[maybe_unused]] size_t sentLen)
{
	sendingNow = false;
	onComplete(!ec);
}
