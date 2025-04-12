#include "WxFrame.h"
#include "ChipmunkPhysicsEngine.h"
#include <wx/dcclient.h>
#include <chrono>

WxFrame::WxFrame(const wxString &title, const wxPoint &pos, const wxSize &size)
    : wxFrame(nullptr, wxID_ANY, title, pos, size)
{
    wxMenu *menuFile = new wxMenu;
    //menuFile->AppendSeparator();
    menuFile->Append(wxID_EXIT);

    wxMenu *menuHelp = new wxMenu;
    menuHelp->Append(wxID_ABOUT, "&About\tF1", "Show about dialog");

    wxMenuBar *menuBar = new wxMenuBar;
    menuBar->Append(menuFile, "&File");
    menuBar->Append(menuHelp, "&Help");

    SetMenuBar(menuBar);
    CreateStatusBar();
    SetStatusText("Simulation Running");
	
    SetMinSize(wxSize(800, 600));
    
    InitializeSimulation();
    CreateTestCreature();
    
    // Set up the timer for animation
    timer = new wxTimer(this, ID_TIMER);
    timer->Start(16);

    Bind(wxEVT_MENU, &WxFrame::OnExit, this, wxID_EXIT);
    Bind(wxEVT_MENU, &WxFrame::OnAbout, this, wxID_ABOUT);
    Bind(wxEVT_PAINT, &WxFrame::OnPaint, this, wxEVT_PAINT);
    // Bind(wxEVT_MENU, &WxFrame::OnTimer, this, wxEVT_TIMER);
}

WxFrame::~WxFrame()
{
    if (timer) {
        timer->Stop();
        delete timer;
    }
    
    // Clean up creatures
    for (Creature* creature : creatures) {
        delete creature;
    }
    creatures.clear();
    
    // Clean up physics
    CleanupSimulation();
}

void WxFrame::OnExit(wxCommandEvent& event)
{
    Close(true);
}

void WxFrame::OnAbout(wxCommandEvent& event)
{
    wxMessageBox("Physics Simulation using Chipmunk and wxWidgets",
        "About", wxOK | wxICON_INFORMATION | wxSTAY_ON_TOP | wxCENTER);
}

void WxFrame::OnPaint(wxPaintEvent& event)
{
    wxPaintDC dc(this);
    
    // Get window size
    wxSize size = GetClientSize();
    
    // Set origin to the center of the window
    dc.SetDeviceOrigin(0, 0);
    
    // Draw background
    dc.SetBrush(*wxWHITE_BRUSH);
    dc.SetPen(*wxTRANSPARENT_PEN);
    dc.DrawRectangle(0, 0, size.GetWidth(), size.GetHeight());
    
    // Draw creatures
    RenderCreatures(dc);
}

void WxFrame::OnTimer(wxTimerEvent& event)
{
    static auto lastTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    
    float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
    lastTime = currentTime;
    
    // Update physics
    UpdateSimulation(deltaTime);
    
    // Refresh the window to update the display
    Refresh();
}

void WxFrame::InitializeSimulation()
{
    physicsEngine = std::make_unique<ChipmunkPhysicsEngine>(1.0f/60.0f);
    
    physicsEngine->Initialize();

    physicsEngine->SetGravity(Vector2(0.0f, 9.8f));
}

void WxFrame::UpdateSimulation(float deltaTime)
{
    if (physicsEngine) {
        physicsEngine->Update(deltaTime);
    }
}

void WxFrame::CleanupSimulation()
{
    if (physicsEngine) {
        physicsEngine->Cleanup();
    }
}

void WxFrame::CreateTestCreature()
{
    Creature* creature = Creature::createBasicCreature();
    
    this->creatures.push_back(creature);
    
    physicsEngine->AddCreature(creature);
}

void WxFrame::RenderCreatures(wxDC& dc)
{
    const float SCALE = 30.0f; // Scale factor for rendering
    
    // For each creature, render all its parts
    for (const Creature* creature : creatures) {
        std::vector<BodyPart*> allParts = creature->getMainBodyParts();
        
        for (const BodyPart* part : allParts) {
            // Get the root part physics body from the engine
            void* body = nullptr;
            
            // TODO: Animate the body parts
            // In a real implementation, we would have a way to look up the body
            // For now, we'll just render the parts at their initial positions
            Vector2 position(0, 0);
            float rotation = 0.0f;
            
            if (physicsEngine) {
                // This is where we would get the actual position and rotation
                // from the physics engine
            }
            
            // Get vertices
            std::vector<Vector2> vertices = part->getVertices();
            
            // Transform vertices by position and rotation
            std::vector<wxPoint> points;
            for (const Vector2& v : vertices) {
                // Apply rotation
                float rotatedX = v.x * cos(rotation) - v.y * sin(rotation);
                float rotatedY = v.x * sin(rotation) + v.y * cos(rotation);
                
                // Apply translation and scale
                float worldX = (rotatedX + position.x) * SCALE;
                float worldY = (rotatedY + position.y) * SCALE;
                
                points.push_back(wxPoint(static_cast<int>(worldX), static_cast<int>(worldY)));
            }
            
            // Draw the polygon
            dc.SetBrush(wxBrush(wxColour(200, 200, 200)));
            dc.SetPen(wxPen(wxColour(0, 0, 0), 1));
            
            if (points.size() >= 3) {
                dc.DrawPolygon(static_cast<int>(points.size()), points.data());
            }
        }
    }
}
