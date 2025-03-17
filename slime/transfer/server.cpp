#include "transfer/server.h"

#include <infiniband/verbs.h>

void on_new_connection(uv_stream_t *server, int status) {
    SLIME_LOG_INFO("new connection...");
    if (status < 0) {
        SLIME_ERROR("New connection error ", uv_strerror(status));
        return;
    }
    uv_tcp_t *client_handle = (uv_tcp_t *)malloc(sizeof(uv_tcp_t));
    uv_tcp_init(loop, client_handle);
    if (uv_accept(server, (uv_stream_t *)client_handle) == 0) {
        client_t *client = new client_t();
        // TODO: use constructor
        client->handle_ = client_handle;
        client_handle->data = client;
        client->state_ = READ_HEADER;
        client->bytes_read_ = 0;
        client->expected_bytes_ = FIXED_HEADER_SIZE;
        uv_read_start((uv_stream_t *)client_handle, alloc_buffer, on_read);
    }
    else {
        uv_close((uv_handle_t *)client_handle, NULL);
    }
}

int init_rdma_context(server_config_t config) {
    struct ibv_device **dev_list;
    struct ibv_device *ib_dev;
    int num_devices;
    dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list) {
        SLIME_ERROR("Failed to get RDMA devices list");
        return -1;
    }

    for (int i = 0; i < num_devices; ++i) {
        char *dev_name_from_list = (char *)ibv_get_device_name(dev_list[i]);
        if (strcmp(dev_name_from_list, config.dev_name.c_str()) == 0) {
            SLIME_LOG_INFO("found device ", dev_name_from_list);
            ib_dev = dev_list[i];
            ib_ctx = ibv_open_device(ib_dev);
            break;
        }
    }

    if (!ib_ctx) {
        SLIME_LOG_INFO(
            "Can't find or failed to open the specified device, try to open "
            "the default device ",
            (char *)ibv_get_device_name(dev_list[0]));
        ib_ctx = ibv_open_device(dev_list[0]);
        if (!ib_ctx) {
            SLIME_ERROR("Failed to open the default device");
            return -1;
        }
    }

    struct ibv_port_attr port_attr;
    ib_port = config.ib_port;
    if (ibv_query_port(ib_ctx, ib_port, &port_attr)) {
        SLIME_ERROR("Unable to query port {} attributes\n", ib_port);
        return -1;
    }
    if ((port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND && config.link_type == "Ethernet") ||
        (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET && config.link_type == "IB")) {
        SLIME_ERROR("port link layer and config link type don't match");
        return -1;
    }
    if (port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
        gidx = -1;
    }
    else {
        gidx = ibv_find_sgid_type(ib_ctx, ib_port, IBV_GID_TYPE_ROCE_V2, AF_INET);
        if (gidx < 0) {
            SLIME_ERROR("Failed to find GID");
            return -1;
        }
    }

    lid = port_attr.lid;
    active_mtu = port_attr.active_mtu;

    pd = ibv_alloc_pd(ib_ctx);
    if (!pd) {
        SLIME_ERROR("Failed to allocate PD");
        return -1;
    }

    return 0;
}

int register_server(unsigned long loop_ptr, server_config_t config) {
    signal(SIGPIPE, SIG_IGN);
    signal(SIGCHLD, SIG_IGN);

    signal(SIGSEGV, signal_handler);
    signal(SIGABRT, signal_handler);
    signal(SIGBUS, signal_handler);
    signal(SIGFPE, signal_handler);
    signal(SIGILL, signal_handler);

    // verification
    assert(config.num_stream > 0 &&
           (config.num_stream == 1 || config.num_stream == 2 || config.num_stream == 4));

    global_config = config;

    loop = uv_default_loop();

    loop = (uv_loop_t *)loop_ptr;
    assert(loop != NULL);
    uv_tcp_init(loop, &server);
    struct sockaddr_in addr;
    uv_ip4_addr("0.0.0.0", config.service_port, &addr);

    uv_tcp_bind(&server, (const struct sockaddr *)&addr, 0);
    int r = uv_listen((uv_stream_t *)&server, 128, on_new_connection);
    if (r) {
        fprintf(stderr, "Listen error: %s\n", uv_strerror(r));
        return -1;
    }

    if (init_rdma_context(config) < 0) {
        return -1;
    }
    mm = new MM(config.prealloc_size << 30, config.minimal_allocate_size << 10, pd);

    SLIME_LOG_INFO("register server done");

    return 0;
}