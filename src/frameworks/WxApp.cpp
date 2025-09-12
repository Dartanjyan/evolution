#include "WxApp.h"
#include "SimulationFrame.h"
#include "MainMenuFrame.h"

bool WxApp::OnInit()
{
    MainMenuFrame *frame = new MainMenuFrame(physicsManager, "Main menu", wxDefaultPosition, wxSize(800, 600));
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
