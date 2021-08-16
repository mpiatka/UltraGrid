#ifndef UG_HOLE_PUNCH_CLIENT_HPP
#define UG_HOLE_PUNCH_CLIENT_HPP

#include <string>
#include <vector>
#include <queue>
#include <functional>
#include <memory>
#include <asio.hpp>
#include "message.hpp"

class Client : public std::enable_shared_from_this<Client>{
public:
	Client(asio::ip::tcp::socket&& socket);
	Client(const Client&) = delete;
	Client(Client&&) = delete;

	Client& operator=(const Client&) = delete;
	Client& operator=(Client&&) = delete;

	void readDescription(std::function<void(Client&, bool)> onComplete);
	std::string getClientName() { return clientName; }
	std::string getRoomName() { return roomName; }
	std::string getSdpDesc() { return sdpDesc; }

	void sendMsg(std::string_view msg);

	void readCandidate(std::function<void(Client&, bool)> onComplete);
	const std::vector<std::string>& getCandidates() { return candidates; }

	bool isSendCallbackPending() const;

private:
	void readNameComplete(asio::ip::tcp::socket& socket,
			std::function<void(Client&, bool)> onComplete, bool success);

	void readRoomComplete(asio::ip::tcp::socket& socket,
			std::function<void(Client&, bool)> onComplete, bool success);

	void readDescComplete(
			std::function<void(Client&, bool)> onComplete, bool success);

	void readCandidateComplete(
			std::function<void(Client&, bool)> onComplete, bool success);

	void onMsgSent(bool success);

	std::string clientName;
	std::string roomName;
	std::string sdpDesc;
	std::vector<std::string> candidates;

	std::queue<std::string> sendQueue;

	Message inMsg;
	Message outMsg;
	asio::ip::tcp::socket socket;
};


#endif
