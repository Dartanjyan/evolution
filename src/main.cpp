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

    auto engine = std::make_unique<ChipmunkEngine>();

    // TODO: call here fabric to get IAICalculator object 
    // (GPU or CPU implementation) and pass to PhysicsManager constructor
    PhysicsManager* physicsManager = new PhysicsManager(std::move(engine));

    WxApp* app = new WxApp();
    app->setPhysicsManager(physicsManager);
    wxApp::SetInstance(app);

    if (!app->CallOnInit()) {
        std::cerr << "Failed to initialize the application\n";
        wxEntryCleanup();
        return 1;
    }

    auto ui = std::make_unique<UI>(app, nullptr);

    int result = ApplicationManager::Run(argc, argv, std::move(ui));
    wxEntryCleanup();

    return result;
}
