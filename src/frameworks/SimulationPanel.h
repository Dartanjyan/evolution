#ifndef SIMULATION_PANEL_H
#define SIMULATION_PANEL_H

#include <wx/wx.h>
#include "PhysicsManager.h"

class SimulationPanel : public wxPanel {
public:
    SimulationPanel(wxWindow* parent, PhysicsManager* physicsManager);
private:
    wxChoice* speedChoice;

    PhysicsManager* physicsManager;
    void OnBackToMenu(wxCommandEvent& event);
    void OnAdd(wxCommandEvent& event);
    void OnSpeedChoice(wxCommandEvent& event);
};

#endif // SIMULATION_PANEL_H
