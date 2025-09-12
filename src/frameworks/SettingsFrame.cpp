#include "SettingsFrame.h"

SettingsFrame::SettingsFrame()
    : wxFrame(nullptr, wxID_ANY, "Settings", wxDefaultPosition, wxSize(400, 300))
{
    wxPanel* panel = new wxPanel(this);

    new wxStaticText(panel, wxID_ANY, "Hello World (Settings Placeholder)", wxPoint(20, 20));

    Bind(wxEVT_CLOSE_WINDOW, [this](wxCloseEvent& evt) {
     wxTheApp->ExitMainLoop();
    });
}
