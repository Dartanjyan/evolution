#ifndef DRAWCOMMANDCOLLECTOR_H
#define DRAWCOMMANDCOLLECTOR_H

#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include "PhysicsManager.h"
#include "Vector2.h"

enum DrawCommandType {
    TEXT,
    CIRCLE,
    LINE,
    POLYGON
};

struct Color {
    uint8_t red, green, blue, alpha=255;

    Color(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha=255)
        : red(red), green(green), blue(blue), alpha(alpha) {};
};

struct DrawCommand {
    DrawCommandType type;
    Color color;
    std::vector<Vector2> points;
    int32_t width = 1;
    std::string text;

    DrawCommand(DrawCommandType type, Color color, std::vector<Vector2> points, int32_t width = 1)
        : type(type), color(color), points(points), width(width) {}
    DrawCommand(DrawCommandType type, Color color, Vector2 point, int32_t width = 1)
        : type(type), color(color), points(std::vector<Vector2>{point}), width(width) {}
    
};
// auto a=sizeof(DrawCommand);


class DrawCommandCollector {
public:
    DrawCommandCollector(PhysicsManager* physicsManager);
    ~DrawCommandCollector();

    void start();
    void stop();
    void getCommands(std::vector<DrawCommand>& commands);
    void setPanelSize(Vector2 newSize);
private:
    void run();
    void flip();
    void updateBackBuffer();

    PhysicsManager* physicsManager;
    std::vector<DrawCommand> buffer1, buffer2;

    std::vector<DrawCommand>* frontBuffer = &buffer1;
    std::vector<DrawCommand>* backBuffer = &buffer2;

    // Initially front buffer is also not ready
    std::atomic<bool> frontBufferReady;
    std::atomic<bool> backBufferReady;
    std::atomic<bool> running;
    std::atomic<Vector2> panelSize;

    std::thread collectorThread;
    std::mutex bufferMutex;
};

#endif // DRAWCOMMANDCOLLECTOR_H
