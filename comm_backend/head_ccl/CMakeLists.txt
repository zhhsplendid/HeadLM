
file(GLOB HEAD_CCL_SRCS
  transport/roce.cpp
  )

add_library(head_ccl SHARED ${HEAD_CCL_SRCS})
