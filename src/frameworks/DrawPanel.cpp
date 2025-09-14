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
    } else {
        this->physicsManager->start();
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

// Cache last used wxPen to reduce pen switching
inline void setPen(wxDC &dc, wxPen pen, bool force = false) {
    static wxPen last_pen;
    if (force || last_pen != pen) {
        dc.SetPen(pen);
        last_pen = pen;
    }
}

void DrawPanel::OnPaint(wxPaintEvent& event) {
    static const wxColour world_shape_color = wxColour(79, 73, 85);

    static const wxColour poly_color = wxColour(170, 153, 137);
    static const wxColour segment_color = wxColour(115, 126, 137);
    static const wxColour circle_color = segment_color;
    static const wxColour muscle_color = wxColour(255, 129, 110);
    const int muscle_width = 4;

    std::vector<BodyObject> bodies {};
    std::vector<ShapeObject> shapes {};
    std::vector<ConstraintObject> constraints {};
    
    physicsManager->getRenderObjects(bodies, shapes, constraints);
    std::vector<const ShapeObject*> circles, segments, polygons, world_circles, world_segments, world_polygons;
    std::vector<const ConstraintObject*> constraints_objects;

    wxBufferedPaintDC dc(this);

    dc.SetBackground(*wxWHITE);
    dc.Clear();
    
    // Fill vectors do draw them with different colors.
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
    for (auto& c: constraints) {
        switch (c.constraintType) {
            case ConstraintType::MUSCLE:
                constraints_objects.push_back(&c);
                break;
            default: break;
        }
    }

    setPen(dc, wxPen("black"), true);
    if (world_polygons.size() > 0) {
        dc.SetBrush(wxBrush(world_shape_color));
        setPen(dc, wxPen("black"));
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
    }

    if (world_segments.size() > 0) {    
        for (const auto *shape : world_segments) {
            const BodyObject* body = shape->body;
            const float angle = body->angle;
            const float radius = shape->radius;
            
            std::array<wxPoint, 2> points;
            for (int i = 0; i < 2; ++i) {
                const auto v = shape->vertices[i].rotated(angle) + body->position;
                points[i] = wxPoint(v.x, v.y);
            }

            setPen(dc, wxPen(world_shape_color, radius - 1));
            dc.DrawLine(points[0], points[1]);
        }
    }

    if(world_circles.size() > 0) {
        dc.SetBrush(world_shape_color);
        dc.SetPen(*wxBLACK_PEN);
        for (const auto *shape : world_circles) {
            const BodyObject* body = shape->body;
            const float angle = body->angle;
            const float radius = shape->radius;
    
            const auto& v = shape->vertices[0].rotated(angle) + body->position;
    
            dc.DrawCircle(wxPoint(v.x, v.y), radius);
        }
    }

    // First draw constraints
    if (constraints_objects.size() > 0) {
        for (const auto *constraint : constraints_objects) {
            const BodyObject* partA = constraint->partA;
            const BodyObject* partB = constraint->partB;
            if (!partA || !partB) {
                std::cout << "Constraint with id=" << constraint->id << " has no partA or partB\n";
                continue;
            }
            const Vector2 anchorA = constraint->anchorA + partA->position;
            const Vector2 anchorB = constraint->anchorB + partB->position;

            setPen(dc, wxPen("black", muscle_width), false);
            dc.DrawLine(wxPoint(anchorA.x, anchorA.y), wxPoint(anchorB.x, anchorB.y));
        }
        for (const auto *constraint : constraints_objects) {
            const BodyObject* partA = constraint->partA;
            const BodyObject* partB = constraint->partB;
            if (!partA || !partB) {
                std::cout << "Constraint with id=" << constraint->id << " has no partA or partB\n";
                continue;
            }
            const Vector2 anchorA = constraint->anchorA + partA->position;
            const Vector2 anchorB = constraint->anchorB + partB->position;

            setPen(dc, wxPen(muscle_color, muscle_width-2), false);
            dc.DrawLine(wxPoint(anchorA.x, anchorA.y), wxPoint(anchorB.x, anchorB.y));
        }
    }

    // Second draw polygons in order for segments to be on top
    if (polygons.size() > 0) {
        dc.SetBrush(wxBrush(poly_color));
        setPen(dc, wxPen("black"));
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
    }

    if (segments.size() > 0) {
        for (const auto *shape : segments) {
            const BodyObject* body = shape->body;
            const float angle = body->angle;
            const float radius = shape->radius;
            
            std::array<wxPoint, 2> points;
            for (int i = 0; i < 2; ++i) {
                const auto v = shape->vertices[i].rotated(angle) + body->position;
                points[i] = wxPoint(v.x, v.y);
            }
            
            // Radius may vary so i have to set pen every time drawing a segment
            setPen(dc, wxPen(wxColour(0, 0, 0), radius-1));
            dc.DrawLine(points[0], points[1]);
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
            
            // Radius may vary so i have to set pen every time drawing a segment
            setPen(dc, wxPen(segment_color, radius - 2));
            dc.DrawLine(points[0], points[1]);
        }
    }

    if(circles.size() > 0) {
        dc.SetPen(*wxBLACK_PEN);
        dc.SetBrush(circle_color);
        for (const auto *shape : circles) {
            const BodyObject* body = shape->body;
            const float angle = body->angle;
            const float radius = shape->radius;
    
            const auto& v = shape->vertices[0].rotated(angle) + body->position;
    
            dc.DrawCircle(wxPoint(v.x, v.y), radius);
        }
    }
    
    // auto now = std::chrono::system_clock::now();
    // auto now_time = std::chrono::system_clock::to_time_t(now);
    // std::string time_str = std::ctime(&now_time);
    // time_str.pop_back();
    dc.SetTextForeground(*wxColor("gray"));
    wxFont font(12, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);
    dc.SetFont(font);
    // dc.DrawText("Current time: " + wxString(time_str), 10, 30);

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

