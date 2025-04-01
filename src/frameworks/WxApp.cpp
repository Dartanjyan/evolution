#include "WxApp.h"
#include "WxFrame.h"

bool WxApp::OnInit() {
    WxFrame *frame = new WxFrame();
    frame->Show(true);
    return true;
}
