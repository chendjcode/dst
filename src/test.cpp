#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

using namespace std;

int accum = 0;
mutex accum_mutex;

void square(int x) {
    accum += x * x;
}

int main() {
    vector<thread> ths;
    for (int i = 1; i <= 20; i++) {
        ths.push_back(thread(&square, i));
    }

    for (auto &th : ths) {
        th.join();
    }
    cout << "accum = " << accum << endl;
    return 0;
}