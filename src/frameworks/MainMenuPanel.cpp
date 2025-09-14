#include "MainMenuPanel.h"
#include "MainFrame.h"

MainMenuPanel::MainMenuPanel(wxWindow* parent)
    : wxPanel(parent, wxID_ANY)
{
    wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);

    wxButton* simBtn = new wxButton(this, wxID_ANY, "Simulation");
    wxButton* settingsBtn = new wxButton(this, wxID_ANY, "Settings");
    wxButton* exitBtn = new wxButton(this, wxID_EXIT, "Exit");

    sizer->Add(simBtn, 0, wxALL | wxEXPAND, 10);
    sizer->Add(settingsBtn, 0, wxALL | wxEXPAND, 10);
    sizer->Add(exitBtn, 0, wxALL | wxEXPAND, 10);

    SetSizer(sizer);

    simBtn->Bind(wxEVT_BUTTON, [parent](wxCommandEvent&) {
        static_cast<MainFrame*>(parent)->ShowSimulation();
    });
    settingsBtn->Bind(wxEVT_BUTTON, [parent](wxCommandEvent&) {
        static_cast<MainFrame*>(parent)->ShowSettings();
    });
    exitBtn->Bind(wxEVT_BUTTON, [](wxCommandEvent&) {
        wxTheApp->ExitMainLoop();
    });
}
