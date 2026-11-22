#ifndef WXAPP_H
#define WXAPP_H

#include <iostream>
#include <wx/wx.h>
#include <memory>
#include "IGUI.h"
#include "PhysicsManager.h"

class WxApp : public wxApp, public IGUI{
private:
    PhysicsManager* physicsManager;
public:
    bool OnInit() override;
    int Run() override;
    void setPhysicsManager(PhysicsManager* physics_manager);
};

#endif
