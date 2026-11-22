#ifndef MAIN_FRAME_H
#define MAIN_FRAME_H

#include <wx/wx.h>
#include "PhysicsManager.h"

class MainFrame : public wxFrame {
public:
    MainFrame(PhysicsManager* physicsManager);

    void ShowMenu();
    void ShowSimulation();
    void ShowSimulationLoadScreen();
    void ShowSettings();

private:
    PhysicsManager* physicsManager;
    wxPanel* currentPanel = nullptr;

    void ClearCurrentPanel();
};

#endif
