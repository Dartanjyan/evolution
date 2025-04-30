#include <iostream>
#include "ApplicationManager.h"
#include "PhysicsManager.h"
#include "UI.h"
#include "WxApp.h"
#include "ChipmunkEngine.h"

#include "PhysicsObjects.h"

wxDECLARE_APP(WxApp);
wxIMPLEMENT_APP_NO_MAIN(WxApp);

int main(int argc, char** argv) {
    wxInitializer initializer(argc, argv);
    if (!initializer.IsOk()) {
        std::cerr << "Failed to initialize wxWidgets\n";
        return 1;
    }

    std::unique_ptr<ChipmunkEngine> engine = std::make_unique<ChipmunkEngine>();

    std::unique_ptr<PhysicsManager> physicsManager = std::make_unique<PhysicsManager>(std::move(engine));

    WxApp* app = new WxApp();
    app->setPhysicsManager(std::move(physicsManager));
    wxApp::SetInstance(app);

    if (!app->CallOnInit()) {
        std::cerr << "Failed to initialize the application\n";
        wxEntryCleanup();
        return 1;
    }

    std::unique_ptr<UI> ui = std::make_unique<UI>(app, nullptr);

    int result = ApplicationManager::Run(argc, argv, std::move(ui));
    wxEntryCleanup();

    return result;
}
