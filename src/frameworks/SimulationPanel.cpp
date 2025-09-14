#include "SimulationPanel.h"
#include "MainFrame.h"
#include "DrawPanel.h"
#include <wx/dcbuffer.h>

SimulationPanel::SimulationPanel(wxWindow* parent, PhysicsManager* physicsManager)
    : wxPanel(parent, wxID_ANY), physicsManager(physicsManager)
{
    DrawPanel *drawPanel = new DrawPanel(std::move(physicsManager), this, wxID_ANY);
    drawPanel->SetBackgroundStyle(wxBG_STYLE_PAINT);
    
    wxButton *addButton = new wxButton(this, ID_ADD_CREATURE, "+");
    wxButton* backBtn = new wxButton(this, wxID_ANY, "Back to Menu");
    
    wxBoxSizer* controlSizer = new wxBoxSizer(wxHORIZONTAL);
    controlSizer->Add(backBtn, 0);
    controlSizer->Add(addButton, 0);
    
    wxBoxSizer* mainSizer = new wxBoxSizer(wxVERTICAL);
    mainSizer->Add(controlSizer, 0);
    mainSizer->Add(drawPanel, 1, wxEXPAND | wxALL, 0);

    SetSizer(mainSizer);

    backBtn->Bind(wxEVT_BUTTON, &SimulationPanel::OnBackToMenu, this);
    Bind(wxEVT_BUTTON, &SimulationPanel::OnAdd, this, ID_ADD_CREATURE);
}

void SimulationPanel::OnBackToMenu(wxCommandEvent& event) {
    static_cast<MainFrame*>(GetParent())->ShowMenu();
}

void SimulationPanel::OnAdd(wxCommandEvent &event)
{
    for (int i=0; i<1; i++) {
        this->physicsManager->addCreature(Creature::createBasicCreature());
    }
}
