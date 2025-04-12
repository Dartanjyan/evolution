#include "UI.h"

UI::UI(IGUI *gui, ICLI *cli) : gui_(gui), cli_(cli)
{
    if (gui_ == nullptr)
    {
        std::cout << "Warning! GUI or CLI is nullptr\n";
    }
}

UI::~UI()
{
    delete gui_;
    delete cli_;
}

int UI::Run(bool gui)
{
    if (gui)
    {
        return gui_->Run();
    }
    else
    {
        return cli_->Run();
    }
}
