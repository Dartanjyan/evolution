#include "DrawPanel.h"
#include <wx/dcbuffer.h>
#include <chrono>

DrawPanel::DrawPanel(PhysicsManager* physicsManager, wxWindow* parent, wxWindowID id,
                     const wxPoint& pos,
                     const wxSize& size,
                     long style) 
    : wxPanel(parent, id, pos, size, style), physicsManager(physicsManager) {
    
    if (!physicsManager) {
        std::cout << "DrawPanel constructor: got nullptr as physicsManager\n";
    }
        
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
    const wxColour world_shape_colour = wxColour(79, 73, 85);
    const wxBrush world_shape_brush = wxBrush(world_shape_colour);
    

    wxBufferedPaintDC dc(this);

    dc.SetBackground(*wxWHITE);
    dc.Clear();

    std::vector<BodyObject> bodies {};
    std::vector<ShapeObject> shapes {};
    std::vector<ConstraintObject> constraints {};
    std::vector<const ShapeObject*> circles, segments, polygons, world_circles, world_segments, world_polygons;
    
    physicsManager->getRenderObjects(bodies, shapes, constraints);

    for (auto& s: shapes) {
        if (s.isWorldObj) {
            switch (s.shapeType) {
                case ShapeType::Circle:   world_circles.push_back(&s); break;
                case ShapeType::Segment:  world_segments.push_back(&s); break;
                case ShapeType::Polygon:  world_polygons.push_back(&s); break;
            }
        }
        else {
            switch (s.shapeType) {
                case ShapeType::Circle:   circles.push_back(&s); break;
                case ShapeType::Segment:  segments.push_back(&s); break;
                case ShapeType::Polygon:  polygons.push_back(&s); break;
            }
        }

        if (!s.body) {
            std::cout<<"Shape with id="<<s.id<<" has no body\n";
            continue;
        }
    }

    // First draw polygons in order for segments to be on top
    dc.SetBrush(*wxBLUE_BRUSH);
    dc.SetPen(*wxBLACK_PEN);
    for (const auto *shape : polygons) {
        const BodyObject* body = shape->body;
        const float angle = body->angle;
        const float radius = shape->radius;

        std::vector<wxPoint> points;
        for (const auto& v : shape->vertices) {
            Vector2 vertex = v.rotated(angle) + body->position;
            points.emplace_back(vertex.x, vertex.y);
        }
        
        dc.DrawPolygon(points.size(), points.data());
    }

    for (const auto *shape : segments) {
        const BodyObject* body = shape->body;
        const float angle = body->angle;
        const float radius = shape->radius;
        
        std::array<wxPoint, 2> points;
        for (int i = 0; i < 2; ++i) {
            const auto v = shape->vertices[i].rotated(angle) + body->position;
            points[i] = wxPoint(v.x, v.y);
        }
        
        dc.SetPen(wxPen(shape->isWorldObj ? world_shape_colour : 0x888888, radius - 1));
        dc.DrawLine(points[0], points[1]);
    }

    for (const auto *shape : circles) {
        const BodyObject* body = shape->body;
        const float angle = body->angle;
        const float radius = shape->radius;

        const auto& v = shape->vertices[0].rotated(angle) + body->position;

        // if it is a world object then brush it with gray color
        dc.SetBrush(shape->isWorldObj ? world_shape_brush : *wxBLUE_BRUSH);
        dc.DrawCircle(wxPoint(v.x, v.y), radius);
    }
    
    auto now = std::chrono::system_clock::now();
    auto now_time = std::chrono::system_clock::to_time_t(now);
    std::string time_str = std::ctime(&now_time);
    time_str.pop_back();
    dc.SetTextForeground(*wxBLACK);
    wxFont font(12, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);
    dc.SetFont(font);
    dc.DrawText("Current time: " + wxString(time_str), 10, 30);

    // FPS counter
    static wxStopWatch sw;
    static int frameCount = 0;
    static float fps = 0;

    frameCount++;
    if (sw.Time() > 500) {
        fps = frameCount / (sw.Time() / 1000.0f);
        frameCount = 0;
        sw.Start();
    }

    dc.DrawText(wxString::Format("FPS: %.f", fps), 10, 10);
}

void DrawPanel::OnTimer(wxTimerEvent& event) {
    Refresh(false);
    Update();
}

void DrawPanel::OnSize(wxSizeEvent& event) {
    Refresh();
    event.Skip();
}

