#include <iostream>
#include <functional>
#include "nat-helper.hpp"

NatHelper::NatHelper(int port) : port(port),
	io_service(),
	//work_guard(io_service.get_executor()),
	acceptor(io_service),
	pendingSocket(io_service)
{

}

NatHelper::NatHelper() : NatHelper(12558)
{

}


void NatHelper::worker(){
	std::cout << "Running" << std::endl;
	io_service.run();
	std::cout << "End" << std::endl;
}

void NatHelper::run(){
	using namespace std::placeholders;

	asio::ip::tcp::endpoint endpoint(asio::ip::tcp::v4(), port);
	acceptor.open(endpoint.protocol());
	acceptor.bind(endpoint);
	acceptor.listen(asio::socket_base::max_connections);
	acceptor.set_option(asio::ip::tcp::acceptor::reuse_address(false));
	acceptor.async_accept(pendingSocket,
			std::bind(&NatHelper::onConnectionAccepted, this, _1));

	std::cout << "Starting thread" << std::endl;
	worker_thread = std::thread(&NatHelper::worker, this);
}

void NatHelper::stop(){
	std::cout << "Stopping..." << std::endl;
	io_service.stop();
	worker_thread.join();
}

void NatHelper::onConnectionAccepted(const std::error_code& ec){
	if(ec){
		std::cerr << "Error accepting client connection: " << ec.message() << std::endl;
		return;
	}

	using namespace std::placeholders;

	auto client = std::make_shared<Client>(std::move(pendingSocket));
	client->readDescription(
			std::bind(&NatHelper::onClientDesc, this, _1, _2));
	incomingClients.insert({client.get(), std::move(client)});

	acceptor.async_accept(pendingSocket,
			std::bind(&NatHelper::onConnectionAccepted, this, _1));
}

void NatHelper::cleanEmptyRooms(){
	for(auto it = rooms.begin(); it != rooms.end(); ){
		if(it->second->isEmpty()){
			std::cout << "Removing empty room " << it->first << "\n";
			it = rooms.erase(it);
		} else {
			it++;
		}
	}
}

void NatHelper::onClientDesc(Client& client, bool success){
	if(!success){
		std::cerr << "Error reading client description" << std::endl;
		incomingClients.erase(&client);
		return;
	}

	const auto& roomName = client.getRoomName();
	std::cout << "Moving client " << client.getClientName()
		<< " to room " << roomName
		<< std::endl;

	auto roomIt = rooms.find(roomName);
	if(roomIt == rooms.end()){
		cleanEmptyRooms();
		std::cout << "Creating room " << roomName << std::endl;
		roomIt = rooms.insert({roomName, std::make_shared<Room>(roomName)}).first;
	}

	auto clientIt = incomingClients.find(&client);
	auto& room = roomIt->second;
	if(room->isFull()){
		std::cerr << "Room " << roomName << " is full, dropping client" << std::endl;
	} else {
		roomIt->second->addClient(std::move(clientIt->second));
	}
	incomingClients.erase(clientIt);
}
