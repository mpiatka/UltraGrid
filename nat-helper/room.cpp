#include <functional>
#include <iostream>
#include "room.hpp"

Room::Room(std::string name) : name(std::move(name)) {  }

void Room::addClient(std::shared_ptr<Client>&& client){
	assert(!isFull());

	Client *clientPtr = client.get();

	for(auto& [c, _] : clients){
		c->sendMsg(clientPtr->getClientName());
		c->sendMsg(clientPtr->getSdpDesc());

		clientPtr->sendMsg(c->getClientName());
		clientPtr->sendMsg(c->getSdpDesc());
		for(const auto& candidate : c->getCandidates()){
			clientPtr->sendMsg(candidate);
		}
	}

	clients.insert({clientPtr, std::move(client)});

	using namespace std::placeholders;
	clientPtr->readCandidate(
			std::bind(&Room::onClientCandidate, shared_from_this(), _1, _2));

}

void Room::onClientCandidate(Client& client, bool success){
	if(!success){
		std::cerr << "Error reading candidate, removing client "
			<< client.getClientName() << std::endl;

		clients.erase(&client);
		return;
	}

	std::cout << "Client candidate recieved" << std::endl;

	for(auto& [c, cu] : clients){
		if(&client == c)
			continue;

		c->sendMsg(client.getCandidates().back());
	}

	using namespace std::placeholders;
	client.readCandidate(
			std::bind(&Room::onClientCandidate, shared_from_this(), _1, _2));
}
