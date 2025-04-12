#include <iostream>
#include "ApplicationManager.h"
#include "UI.h"
#include "WxApp.h"

wxDECLARE_APP(WxApp);
wxIMPLEMENT_APP_NO_MAIN(WxApp);

int main(int argc, char** argv) {
    wxInitializer initializer(argc, argv);
    if (!initializer.IsOk()) {
        std::cerr << "Failed to initialize wxWidgets\n";
        return 1;
    }

    WxApp* app = new WxApp();
    wxApp::SetInstance(app);

    if (!app->CallOnInit()) {
        std::cerr << "Failed to initialize the application\n";
        wxEntryCleanup();
        return 1;
    }

    UI* ui = new UI(app, nullptr);
    int result = ApplicationManager::Run(argc, argv, ui);
    wxEntryCleanup();
    return result;
}