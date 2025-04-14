#include "head_ccl/topo.hpp"

#include <string>

#include "utils/device.hpp"
#include "utils/json.hpp"
#include "head_ccl/topo.hpp"
#include "topo.hpp"

namespace comm_backend {
namespace head_ccl {

using nlohmann::json;
using utils::DeviceCompany;

TopoGraph::TopoGraph(const json &json_data)
{
    for (const auto& elem : json_data.items()) {
        int rank = std::stoi(elem.key());
        rank_node_.insert({rank, TopoNode(elem.value())});
    }
}

int TopoGraph::add_node_from_json(const json &j)
{
    int rank = j.at("rank").get<int>();
    rank_node_.insert({rank, TopoNode(j)});
    return 0;
}

void TopoGraph::remove_node_by_rank(int rank)
{
    rank_node_.erase(rank);
}

json TopoGraph::to_json() const
{
    json graph_json;
    for (const auto& p : rank_node_) {
        graph_json[p.first] = p.second.to_json();
    }
    return graph_json;
}

int ConstructGraphFromInitJson(const std::vector<json> &json_data, TopoGraph *graph)
{
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
    
    std::vector<json> collected_data;
    // Rank 0 data;
    collected_data.push_back(utils::device_init_info_json());

    int port = utils::get_master_port();
    zmq::context_t ctx;
    zmq::socket_t puller(ctx, ZMQ_PULL);
    puller.bind("tcp://*:" + std::to_string(port));

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


int MasterSendTopo(const TopoGraph& graph) {
    zmq::context_t ctx(1);

    int port = utils::get_master_port();
    // PUB 套接字广播数据
    zmq::socket_t pub_socket(ctx, ZMQ_PUB);
    pub_socket.bind("tcp://*:" + std::to_string(port));

    // REP 套接字处理同步请求
    zmq::socket_t sync_socket(ctx, ZMQ_REP);
    sync_socket.bind("tcp://*:" + std::to_string(port + 1));

    // 等待所有 Worker 就绪
    const int num_workers = graph.node_size();
    int ready_workers = 0;
    while (ready_workers < num_workers) {
        zmq::message_t msg;
        sync_socket.recv(msg, zmq::recv_flags::none); // 接收就绪信号
        sync_socket.send(zmq::message_t(), zmq::send_flags::none); // 回复确认
        ready_workers++;
        HEADLM_LOG_INFO(" Worker " << ready_workers << " 已就绪");
    }

    // 构造 JSON 数据
    std::string json_str = graph.to_json().dump();

    // 广播 JSON
    zmq::message_t msg(json_str.data(), json_str.size());
    pub_socket.send(msg, zmq::send_flags::none);
    HEADLM_LOG_INFO("Master sent graph JSON data");
    return 0;
}

int WorkerReceiveTopo(TopoGraph* graph) {
    
    zmq::context_t ctx(1);

    std::string master_addr = utils::get_master_addr();
    int master_port = utils::get_master_port();

    // SUB 套接字接收数据
    zmq::socket_t sub_socket(ctx, ZMQ_SUB);
    sub_socket.connect("tcp://" + master_addr + ":" + std::to_string(master_port));
    sub_socket.setsockopt(ZMQ_SUBSCRIBE, "", 0); // 订阅所有消息

    // REQ 套接字发送就绪信号
    zmq::socket_t sync_socket(ctx, ZMQ_REQ);
    sync_socket.connect("tcp://" + master_addr + ":" + std::to_string(master_port + 1));

    // 通知 Master 已就绪
    sync_socket.send(zmq::message_t(), zmq::send_flags::none);
    zmq::message_t ack;
    sync_socket.recv(ack, zmq::recv_flags::none); // 等待Master确认
    HEADLM_LOG_INFO("Synced and wait for topo graph");

    // 接收 JSON 数据
    zmq::message_t msg;
    sub_socket.recv(msg, zmq::recv_flags::none);
    std::string json_str(static_cast<char*>(msg.data()), msg.size());

    // 解析 JSON
    auto data = nlohmann::json::parse(json_str);
    HEADLM_LOG_INFO("收到数据: " << data.dump(4));
    if (graph != nullptr) {
        delete graph;
        graph = nullptr;
    }
    graph = new TopoGraph(data);
    return 0;
}

} // namespace head_ccl
} // namespace comm_backend