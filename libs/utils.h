#if !defined(ALEX_UTILS_H)
#define ALEX_UTILS_H

#include <chrono>
using namespace std;

auto timeMillis()
{
    auto time = chrono::system_clock::now();
    auto since_epoch = time.time_since_epoch();
    auto millis = chrono::duration_cast<chrono::milliseconds>(since_epoch);
    return millis.count();
};

#endif // ALEX_UTILS_H
