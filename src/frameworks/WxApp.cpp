#include "WxApp.h"
#include "WxFrame.h"

bool WxApp::OnInit() {
    WxFrame *frame = new WxFrame("Simulation", wxDefaultPosition, wxDefaultSize);
    frame->Show(true);

    return true;
}

int WxApp::Run()
{
    return wxApp::OnRun();
}
