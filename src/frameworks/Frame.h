#ifndef FRAME_H
#define FRAME_H

#include <wx/wx.h>

class Frame : public wxFrame {
public:
    Frame(const wxString &title, const wxPoint &pos = wxDefaultPosition, const wxSize &size = wxDefaultSize);
    ~Frame();
protected:
    void OnCloseWindow(wxCloseEvent& event);
    virtual void HandleExit() = 0;
};

#endif // FRAME_H