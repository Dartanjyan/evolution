#include "Frame.h"
#include "SettingsFrame.h"

Frame::Frame(const wxString & title, const wxPoint & pos, const wxSize & size)
    : wxFrame(nullptr, wxID_ANY, title, pos, size)
{
    Bind(wxEVT_CLOSE_WINDOW, [this](wxCloseEvent& evt) {
        wxTheApp->ExitMainLoop();
    });
}

Frame::~Frame()
{
}

void Frame::OnCloseWindow(wxCloseEvent &event) {
    HandleExit();
    event.Skip();
    Close(true);
}