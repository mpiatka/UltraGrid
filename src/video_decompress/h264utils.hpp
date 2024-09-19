#ifndef H264UTILS_HPP_e2e63570adcb
#define H264UTILS_HPP_e2e63570adcb

#include <vector>
#include "rtp/rtpdec_h264.h"
#include "rtp/rtpenc_h264.h"

struct Nal_unit{
        int type = 0;
        size_t len = 0;

        uint8_t header = 0;

        std::vector<unsigned char> rbsp;
};

class Nal_unit_reader{
public:
        Nal_unit_reader() = default;

        void set_data(const unsigned char *data, size_t data_len){
                this->data = data;
                this->data_len = data_len;
                parse_ptr = data;
                nalu = {};
        }

        Nal_unit *get_next(){
                while(parse_ptr && parse_ptr < data + data_len){
                        auto nal_start = rtpenc_get_next_nal(parse_ptr, data_len - (parse_ptr - data), &parse_ptr);
                        if(!nal_start)
                                break;

                        size_t nal_len = parse_ptr ? (parse_ptr - nal_start) : ((data + data_len) - nal_start);

                        if(parse_nal(nal_start, nal_len))
                                return &nalu;
                }

                return nullptr;
        }

private:
        const unsigned char *data = nullptr;
        size_t data_len = 0;

        bool parse_nal(const unsigned char *start, int len){
                nalu.rbsp.resize(len);

                int rbsp_size = len;
                int ret = nal_to_rbsp(start, &len, nalu.rbsp.data(), &rbsp_size);

                if(ret < 0)
                        return false;

                nalu.len = len;
                nalu.type = H264_NALU_HDR_GET_TYPE(*start);
                nalu.rbsp.resize(rbsp_size);
                nalu.header = start[0];

                return true;
        }

        const unsigned char *parse_ptr = nullptr;

        Nal_unit nalu = {};
};

#endif
