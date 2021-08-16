#include <iostream>
#include <charconv>
#include "nat-helper.hpp"

namespace {
	void printUsage(std::string_view name){
		std::cout << "Usage: "
			<< name << " [-h/--help] [-p/--port <port>]\n";
	}
}

int main(int argc, char **argv){
	int port = 12558;

	for(int i = 1; i < argc; i++){
		std::string_view arg(argv[i]);
		if(arg == "-h" || arg == "--help"){
			printUsage(argv[0]);
			return 0;
		} else if(arg == "-p" || arg == "--port"){
			if(i + 1 >= argc){
				std::cerr << "Expected port number\n";
				printUsage(argv[0]);
				return 1;
			}
			std::string_view portStr(argv[++i]);
			auto res = std::from_chars(portStr.data(), portStr.data() + portStr.size(), port);
			if(res.ec != std::errc() || res.ptr == portStr.data()){
				std::cerr << "Failed to parse port number\n";
				printUsage(argv[0]);
				return 1;
			}
		} else {
			std::cerr << "Unknown argument " << arg << std::endl;
			printUsage(argv[0]);
			return 1;
		}
	}


	NatHelper server(port);

	server.run();

	for(std::string in; std::getline(std::cin, in);){
		if(in == "exit"){
			break;
		}
	}

	server.stop();

	return 0;
}
