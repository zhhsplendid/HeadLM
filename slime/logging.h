//
// Created by jimy on 3/13/22.
//

#pragma once

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

inline std::string get_env_variable(char const *env_var_name) {
  if (!env_var_name) {
    return "";
  }
  char *lvl = getenv(env_var_name);
  if (lvl)
    return std::string(lvl);
  return "";
}

inline int get_log_level() {
  std::string lvl = get_env_variable("SLIME_LOG_LEVEL");
  return !lvl.empty() ? atoi(lvl.c_str()) : 0;
}

#define SLIME_ASSERT(Expr, Msg)                                                \
  {                                                                            \
    if (!(Expr)) {                                                             \
      std::cerr << "\033[1;91m"                                                \
                << "[Assertion Failed]"                                        \
                << "\033[m " << __FILE__ << ": " << __FUNCTION__ << ": Line"   \
                << __LINE__ << ", Expected :" << #Expr << std::endl;           \
      abort();                                                                 \
    }                                                                          \
  }

#define SLIME_ASSERT_EQ(A, B, Msg, ...) SLIME_ASSERT((A) == (B), Msg)
#define SLIME_ASSERT_NE(A, B, Msg, ...) SLIME_ASSERT((A) == (B), Msg)

#define SLIME_ABORT(Msg)                                                       \
  {                                                                            \
    std::cerr << ": \033[1;91m"                                                \
              << "[Fatal]"                                                     \
              << "\033[m " << __FILE__ << ": " << __FUNCTION__ << ": Line"     \
              << __LINE__ << ": " << Msg << std::endl;                         \
    abort();                                                                   \
  }

#define SLIME_ERROR(Msg) SLIME_ABORT(Msg)

#define SLIME_LOG_LEVEL(Msg, MsgType, Level)                                   \
  {                                                                            \
    if (get_log_level() >= Level) {                                            \
      std::cerr << ": \033[1;91m"                                              \
                << "[" << MsgType << "]"                                       \
                << "\033[m " << __FILE__ << ": " << __FUNCTION__ << ": Line"   \
                << __LINE__ << ": " << Msg << std::endl;                       \
    }                                                                          \
  }

#define SLIME_LOG_INFO(Msg) SLIME_LOG_LEVEL(Msg, "Info", 1)

#define SLIME_LOG_DEBUG(Msg) SLIME_LOG_LEVEL(Msg, "Debug", 2)

#define SLIME_LOG_WARN(Msg) SLIME_LOG_LEVEL(Msg, "Warn", 3)
