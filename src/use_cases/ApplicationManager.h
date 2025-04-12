#ifndef APPLICATION_MANAGER_H
#define APPLICATION_MANAGER_H

#include <string>
#include "IUI.h"

namespace ApplicationManager
{
    // This function is the entry point for the application manager.
    // It takes command line arguments and a renderer interface as parameters.
    // It returns an integer indicating the success or failure of the operation.
    // The function is responsible for managing the application lifecycle and rendering.
    int Run(int argc, char** argv, IUI* ui);
}

#endif
