#include "SimulationPanel.h"
#include "MainFrame.h"
#include "DrawPanel.h"
#include <wx/dcbuffer.h>

SimulationPanel::SimulationPanel(wxWindow* parent, PhysicsManager* physicsManager)
    : wxPanel(parent, wxID_ANY), physicsManager(physicsManager)
{
    DrawPanel *drawPanel = new DrawPanel(physicsManager, this, wxID_ANY);
    drawPanel->SetBackgroundStyle(wxBG_STYLE_PAINT);
    
    wxButton *addButton = new wxButton(this, ID_ADD_CREATURE, "+");
    wxButton* backBtn = new wxButton(this, wxID_ANY, "Back to Menu");

    wxArrayString speeds;
    speeds.Add("0.25x");
    speeds.Add("0.5x");
    speeds.Add("1x");
    speeds.Add("2x");
    speeds.Add("4x");
    speeds.Add("8x");
    speeds.Add("Unlimited");
    // TODO: ID_SPEED_CHANGE
    speedChoice = new wxChoice(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, speeds);
    speedChoice->SetSelection(2);
    speedChoice->Bind(wxEVT_CHOICE, &SimulationPanel::OnSpeedChoice, this);
    
    wxBoxSizer* controlSizer = new wxBoxSizer(wxHORIZONTAL);
    controlSizer->Add(backBtn, 0);
    controlSizer->Add(addButton, 0);
    controlSizer->Add(speedChoice, 0);
    
    wxBoxSizer* mainSizer = new wxBoxSizer(wxVERTICAL);
    mainSizer->Add(controlSizer, 0);
    mainSizer->Add(drawPanel, 1, wxEXPAND | wxALL, 0);

    SetSizer(mainSizer);

    backBtn->Bind(wxEVT_BUTTON, &SimulationPanel::OnBackToMenu, this);
    Bind(wxEVT_BUTTON, &SimulationPanel::OnAdd, this, ID_ADD_CREATURE);
}

void SimulationPanel::OnSpeedChoice(wxCommandEvent& event) {
    wxString value = speedChoice->GetStringSelection();
}

void SimulationPanel::OnBackToMenu(wxCommandEvent& event) {
    physicsManager->stop();
    static_cast<MainFrame*>(GetParent())->ShowMenu();
}

void SimulationPanel::OnAdd(wxCommandEvent &event) {
    for (int i=0; i<1; i++) {
        this->physicsManager->addCreature(Creature::createBasicCreature());
    }
}

