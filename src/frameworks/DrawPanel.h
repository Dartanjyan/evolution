#ifndef DRAW_PANEL_H
#define DRAW_PANEL_H

#include <wx/wx.h>
#include <wx/panel.h>
#include "WxIds.h"

class DrawPanel : public wxPanel {
public:
    DrawPanel(wxWindow* parent, wxWindowID id = wxID_ANY,
              const wxPoint& pos = wxDefaultPosition,
              const wxSize& size = wxDefaultSize,
              long style = wxFULL_REPAINT_ON_RESIZE);
    ~DrawPanel();
private:
    void OnPaint(wxPaintEvent& event);
    void OnTimer(wxTimerEvent& event);
    void OnSize(wxSizeEvent& event);
    wxTimer* timer;
};

#endif