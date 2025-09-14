#include "WxApp.h"
#include "MainFrame.h"

bool WxApp::OnInit()
{
    MainFrame *frame = new MainFrame(physicsManager);
    frame->Show(true);
    return true;
}

int WxApp::Run()
{
    return wxApp::OnRun();
}

void WxApp::setPhysicsManager(PhysicsManager* physics_manager) 
{ 
    this->physicsManager = physics_manager;
}
