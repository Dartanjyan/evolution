#include "DrawPanel.h"

#include <wx/dcbuffer.h>
#include <chrono>
#include <thread>
#include <array>

DrawPanel::DrawPanel(PhysicsManager* physicsManager, wxWindow* parent, wxWindowID id,
                     const wxPoint& pos,
                     const wxSize& size,
                     long style) 
    : wxPanel(parent, id, pos, size, style), physicsManager(physicsManager)
{
    bufferDrawer = std::make_unique<BufferDrawer>(physicsManager);
    bufferDrawer->start();
        
    SetBackgroundColour(wxColour(240, 240, 240));
    
    timer = new wxTimer(this, ID_TIMER);
    timer->SetOwner(this, ID_TIMER);
    timer->Start(1000/60);
    
    Bind(wxEVT_SIZE, &DrawPanel::OnSize, this);
    Bind(wxEVT_PAINT, &DrawPanel::OnPaint, this);
    Bind(wxEVT_TIMER, &DrawPanel::OnTimer, this, ID_TIMER);
}

DrawPanel::~DrawPanel()
{
    if (timer) {
        timer->Stop();
        delete timer;
        timer = nullptr;
    }
}

void DrawPanel::OnPaint(wxPaintEvent& event) {
    const wxSize size = GetClientSize();
    if (size.x <= 0 || size.y <= 0) return;

    wxBufferedPaintDC dc(this);
    wxBitmap bitmap;
    bufferDrawer->getFrame(bitmap);
    if (bitmap.IsOk()) {
        dc.DrawBitmap(bitmap, 0, 0, false);
    }
}

void DrawPanel::OnTimer(wxTimerEvent& event) {
    Refresh(false);
    Update();
}

void DrawPanel::OnSize(wxSizeEvent& event) {
    Refresh();
    event.Skip();
    
    auto _size = GetClientSize();
    bufferDrawer->setPanelSize(_size);
    std::cout<<"Size: "<<_size.x<<", "<<_size.y<<"\n";
    // std::cout<<"resize\n";
}

