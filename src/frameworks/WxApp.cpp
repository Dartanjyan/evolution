#include "WxApp.h"
#include "WxFrame.h"

bool WxApp::OnInit() {
    WxFrame *frame = new WxFrame("Simulation", wxDefaultPosition, wxDefaultSize);
    frame->Show(true);
    std::cout<<"WxApp::OnInit: Frame shown\n";

    return true;
}
