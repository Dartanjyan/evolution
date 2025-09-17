#ifndef BUFFERDRAWER_H
#define BUFFERDRAWER_H

#include <wx/dcbuffer.h>
#include <thread>
#include <atomic>
#include <mutex>
#include "PhysicsManager.h"


class BufferDrawer {
public:
    BufferDrawer(PhysicsManager* physicsManager);
    ~BufferDrawer();

    void start();
    void stop();
    void getFrame(wxBitmap& frame);

    void setPanelSize(const wxSize& size);
private:
    void run();
    void flip();
    void renderBackBuffer();

    PhysicsManager* physicsManager;
    wxBitmap buffer1, buffer2;

    wxBitmap* frontBuffer = &buffer1;
    wxBitmap* backBuffer = &buffer2;

    // Initially front buffer is also not ready
    std::atomic<bool> frontBufferReady;
    std::atomic<bool> backBufferReady;
    std::atomic<bool> running;
    std::atomic<wxSize> panelSize;

    std::thread drawingThread;
    std::mutex bufferSwapMutex;
    std::mutex backBufferMutex;
};

#endif // BUFFERDRAWER_H