#include "SimulationLoadPanel.h"
#include "MainFrame.h"

SimulationLoadPanel::SimulationLoadPanel(wxWindow* parent)
    : wxPanel(parent, wxID_ANY)
{
    wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);

    wxButton* exitBtn = new wxButton(this, wxID_EXIT, "Exit");

    sizer->Add(exitBtn, 0, wxALL | wxEXPAND, 10);

    SetSizer(sizer);

    exitBtn->Bind(wxEVT_BUTTON, [parent](wxCommandEvent&) {
        static_cast<MainFrame*>(parent)->ShowMenu();
    });
}
