#include "WxApp.h"
#include "WxFrame.h"

bool WxApp::OnInit()
{
    WxFrame *frame = new WxFrame(std::move(physicsManager), "Simulation", wxDefaultPosition, wxDefaultSize);
    frame->Show(true);
    return true;
}

int WxApp::Run()
{
    return wxApp::OnRun();
}

void WxApp::setPhysicsManager(std::unique_ptr<PhysicsManager> physics_manager) 
{ 
    this->physicsManager = std::move(physics_manager);
}
