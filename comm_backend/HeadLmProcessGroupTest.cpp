#include "HeadLmProcessGroup.hpp"

#include <iostream>
#include <gtest/gtest.h>


TEST(HelloTest, BasicAssertions) {
    ASSERT_EQ(1, 1);
    int a = 1;
}


int main(int argc, char *argv[]) {
    std::cout<<"hello main" << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}