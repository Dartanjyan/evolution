#include "UI.h"

UI::UI(IGUI *gui, ICLI *cli) : gui_(gui), cli_(cli)
{
    if (gui_ == nullptr || cli_ == nullptr)
    {
        throw std::invalid_argument("GUI or CLI cannot be null");
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

void UI::Init(bool gui)
{
    if (gui)
{
        gui_->Init();
    }
    else
    {
        cli_->Init();
    }
}
