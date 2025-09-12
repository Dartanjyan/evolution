#ifndef SETTINGS_FRAME_H
#define SETTINGS_FRAME_H

#include <wx/wx.h>
#include "Frame.h"

class SettingsFrame : public Frame {
public:
    SettingsFrame();
private:
    void OnCloseWindow(wxCloseEvent& event);
    void HandleExit() override;
};

#endif // SETTINGS_FRAME_H
