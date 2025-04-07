#include "WxApp.h"
#include "WxFrame.h"
#include "../use_cases/print.h"

bool WxApp::OnInit() {
    WxFrame *frame = new WxFrame("Simulation", wxDefaultPosition, wxDefaultSize);
    frame->Show(true);
    print("WxApp::OnInit: Frame shown");
    return true;
}
