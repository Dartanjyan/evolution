#include "SettingsPanel.h"
#include "MainFrame.h"

SettingsPanel::SettingsPanel(wxWindow* parent)
    : wxPanel(parent, wxID_ANY)
{
    wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);

    new wxStaticText(this, wxID_ANY, "Hello World (Settings Placeholder)");
    wxButton* backBtn = new wxButton(this, wxID_ANY, "Back to Menu");

    sizer->AddStretchSpacer(1);
    sizer->Add(backBtn, 0, wxALL | wxCENTER, 10);

    SetSizer(sizer);

    backBtn->Bind(wxEVT_BUTTON, [parent](wxCommandEvent&) {
        static_cast<MainFrame*>(parent)->ShowMenu();
    });
}
