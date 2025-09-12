#include "SettingsFrame.h"

SettingsFrame::SettingsFrame()
    : Frame("Settings")
{
    wxPanel* panel = new wxPanel(this);

    new wxStaticText(panel, wxID_ANY, "Hello World (Settings Placeholder)", wxPoint(20, 20));
}

void SettingsFrame::HandleExit()
{
}
