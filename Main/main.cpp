#include"ImgProdCons.h"
#include<thread>

using namespace std;

bool rm::ImgProdCons::_quit_flag = false;

int main()
{
    rm::ImgProdCons imgProdCons;

    imgProdCons.init();

    std::thread produceThread(&rm::ImgProdCons::produce, &imgProdCons);  // 获取图像保存到缓存队列里
    std::thread consumeThread(&rm::ImgProdCons::consume, &imgProdCons);  // 图片的处理和指令的发送
    std::thread senseThread(&rm::ImgProdCons::sense, &imgProdCons);      // 接收数据

    produceThread.join();
    consumeThread.join();
    senseThread.join();

    return 0;
}
