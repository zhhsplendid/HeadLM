#include "head_ccl/topo.hpp"

#include <string>

#include "utils/device.hpp"
#include "utils/json.hpp"
#include "head_ccl/topo.hpp"

namespace comm_backend {
namespace head_ccl {

using nlohmann::json;
using utils::DeviceCompany;

int TopoGraph::add_node_from_json(const json &j)
{
    int rank = j.at("rank").get<int>();
    rank_node.insert({rank, TopoNode(j)});
    return 0;
}

void TopoGraph::remove_node_by_rank(int rank)
{
    rank_node.erase(rank);
}

int ConstructGraphFromInitJson(const std::vector<json>& json_data, TopoGraph* graph) {
    if (graph != nullptr) {
        delete graph;
    }
    graph = new TopoGraph();

    std::unordered_map<std::string, int> node_addr_rank;
    for (const json& j : json_data) {
        const std::string& addr = j.at("address").get<std::string>();
        const int rank = j.at("rank").get<int>();
        if (node_addr_rank.count(addr)) {
            int stored_rank = node_addr_rank[addr];
            if (stored_rank > rank) {
                graph->remove_node_by_rank(stored_rank);
                graph->add_node_from_json(j);
            }
        } else {
            graph->add_node_from_json(j);
            node_addr_rank.insert({addr, rank});
        }
    }
    return 0;
}

volatile bool user_stop = false;

void signal_handler(int) { user_stop = true; }

int MasterCollectInitJson(TopoGraph* graph) {
    signal(SIGINT, signal_handler);
    
    int port = utils::get_master_port();
    
    zmq::context_t ctx;
    zmq::socket_t puller(ctx, ZMQ_PULL);
    puller.bind("tcp://*:" + std::to_string(port));

    std::vector<json> collected_data;
    
    HEADLM_LOG_INFO("Master started initialization, waiting for data...");
    int world_size = std::stoi(get_env_variable("WORLD_SIZE"));
    // Master machine has processes of the number of device.
    // It expects receive every other process
    size_t expect_rec_size = world_size - utils::device_count();

    const auto start_time = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::seconds(HEADLM_TIMEOUT_SEC);
    while (collected_data.size() < expect_rec_size && !user_stop) {
        const auto now = std::chrono::steady_clock::now();
        if (now - start_time > timeout) {
            HEADLM_LOG_INFO("Timeout reached, stopping worker...");
            break;
        }
        zmq::message_t msg;
        if (puller.recv(msg, zmq::recv_flags::dontwait)) {
            try {
                auto json = json::parse(
                    std::string(static_cast<char*>(msg.data()), msg.size()));
                collected_data.push_back(json);
                HEADLM_LOG_INFO("Master received data number " << collected_data.size());
            } catch (const std::exception& e) {
                HEADLM_LOG_INFO("Master parse initialization json error: " << e.what());
                return -1;
            }
        }
    }
    if (collected_data.size() < expect_rec_size) {
        return -1;
    }

    return ConstructGraphFromInitJson(collected_data, graph);
}

int WorkerSendInitJson() {
    signal(SIGINT, signal_handler);

    zmq::context_t ctx;
    zmq::socket_t pusher(ctx, ZMQ_PUSH);

    // Worker可能比master更早启动，需做一些重连处理
    // 设置套接字选项
    int reconnect_ivl = 1000;    // 1秒重连间隔
    int sndhwm = 1000;           // 发送队列大小
    int linger = 5000;           // 退出时等待5秒

    pusher.setsockopt(ZMQ_RECONNECT_IVL, &reconnect_ivl, sizeof(reconnect_ivl));
    pusher.setsockopt(ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm));
    pusher.setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
    
    std::string master_addr = utils::get_master_addr();
    int master_port = utils::get_master_port();
    pusher.connect("tcp://" + master_addr + std::to_string(master_port));

    const auto start_time = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::seconds(HEADLM_TIMEOUT_SEC);
    while (!user_stop) {
        const auto now = std::chrono::steady_clock::now();
        if (now - start_time > timeout) {
            HEADLM_LOG_INFO("Timeout reached, stopping worker...");
            break;
        }

        try {
            json data = utils::device_init_info_json();
            std::string str_data = data.dump();

            zmq::message_t msg(str_data.size());
            memcpy(msg.data(), str_data.data(), str_data.size());

            pusher.send(msg, zmq::send_flags::dontwait);
            HEADLM_LOG_INFO("Queue data: " << data.dump());
            return 0;
        } catch (const zmq::error_t& e) {
            if (e.num() == EAGAIN) {
                HEADLM_LOG_INFO("Queue full, retrying...");
            } else {
                HEADLM_LOG_INFO("Error: " << e.what());
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    return -1;
}

} // namespace head_ccl
} // namespace comm_backend