#include "SimulationPanel.h"
#include "MainFrame.h"
#include "DrawPanel.h"
#include <exception>
#include <wx/dcbuffer.h>

SimulationPanel::SimulationPanel(wxWindow* parent, PhysicsManager* physicsManager)
    : wxPanel(parent, wxID_ANY), physicsManager(physicsManager)
{
    drawPanel = new DrawPanel(physicsManager, this, wxID_ANY);
    drawPanel->SetBackgroundStyle(wxBG_STYLE_PAINT);

    wxButton *loadButton = new wxButton(this, wxID_ANY, "Load");
    wxButton *backBtn = new wxButton(this, wxID_ANY, "Back to Menu");
    wxButton *updateBtn = new wxButton(this, wxID_ANY, "Update");
    updateBtn->Bind(wxEVT_BUTTON, &SimulationPanel::OnUpdateButton, this);

    wxArrayString speeds;
    speeds.Add("0.25x");
    speeds.Add("0.5x");
    speeds.Add("1x");
    speeds.Add("2x");
    speeds.Add("4x");
    speeds.Add("8x");
    speeds.Add("Unlimited");
    speedChoice = new wxChoice(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, speeds);
    speedChoice->SetSelection(2);
    speedChoice->Bind(wxEVT_CHOICE, &SimulationPanel::OnSpeedChoice, this);
    
    wxBoxSizer* controlSizer = new wxBoxSizer(wxHORIZONTAL);
    controlSizer->Add(backBtn, 0);
    controlSizer->Add(loadButton, 0);
    controlSizer->Add(updateBtn, 0);
    controlSizer->Add(speedChoice, 0);
    
    wxBoxSizer* mainSizer = new wxBoxSizer(wxVERTICAL);
    mainSizer->Add(controlSizer, 0);
    mainSizer->Add(drawPanel, 1, wxEXPAND | wxALL, 0);

    SetSizer(mainSizer);

    backBtn->Bind(wxEVT_BUTTON, &SimulationPanel::OnBackToMenu, this);
    loadButton->Bind(wxEVT_BUTTON, &SimulationPanel::OnLoad, this);
}

void SimulationPanel::OnSpeedChoice(wxCommandEvent& event) {
    wxString value = speedChoice->GetStringSelection();
    float scale;

    if (value == "0.25x") {
        scale = 0.25;
    } else if (value == "0.5x") {
        scale = 0.5;
    } else if (value == "1x") {
        scale = 1;
    } else if (value == "2x") {
        scale = 2;
    } else if (value == "4x") {
        scale = 4;
    } else if (value == "8x") {
        scale = 8;
    } else {
        scale = 0;
    }

    physicsManager->setUpdateTimeScale(scale);
}

void SimulationPanel::OnUpdateButton(wxCommandEvent &event)
{
    drawPanel->OnUpdateBtn();
}

void SimulationPanel::OnBackToMenu(wxCommandEvent& event) {
    static_cast<MainFrame*>(GetParent())->ShowMenu();
}

void SimulationPanel::OnLoad(wxCommandEvent &event) {
    try {
        physicsManager->loadSimulation();
    } catch (const std::exception& error) {
        wxMessageBox(wxString::FromUTF8(error.what()), "Could not load simulation_save.json",
                     wxOK | wxICON_ERROR, this);
    }
}

