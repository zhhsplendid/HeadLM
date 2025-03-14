#include "utils.h"
#include "logging.h"

#include <boost/stacktrace.hpp>
#include <iostream>
#include <ostream>

void signal_handler(int signum) {
    SLIME_LOG_INFO("Interrupt signal (" + std::to_string(signum) + ") received.");
    boost::stacktrace::stacktrace st;
    std::ostringstream oss;
    oss << st;
    SLIME_ERROR("Stacktrace:\n" + oss.str());
    exit(1);
}