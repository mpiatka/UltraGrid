#ifndef UG_HOLE_PUNCH_ROOM_HPP
#define UG_HOLE_PUNCH_ROOM_HPP

#include <map>
#include <memory>
#include "client.hpp"

class Room : public std::enable_shared_from_this<Room>{
public:
	Room(std::string name);

	void addClient(std::shared_ptr<Client>&& client);

	bool isFull() const { return clients.size() >= 2; }
	bool isEmpty() const { return clients.empty(); }

private:
	void onClientCandidate(Client& client, bool success);

	std::string name;
	std::map<Client *, std::shared_ptr<Client>> clients;
};

#endif
