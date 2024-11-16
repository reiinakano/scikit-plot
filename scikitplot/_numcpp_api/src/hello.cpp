// src/hello.cpp
// C++ Source File
#include <iostream>
#include <string>

#ifndef DEFAULT_MESSAGE      // 1. Check if DEFAULT_MESSAGE is not defined
#define DEFAULT_MESSAGE "Hello, from C++!"  // 2. If not, define it with the value "Hello, from C++!"
// This block will be ignored if DEFAULT_MESSAGE is already defined.
#endif // DEFAULT_MESSAGE  // 3. End the condition

// extern "C" disables name mangling, making the function callable from C
// and other languages that expect C-style linkage.
// Without extern "C", name mangling is enabled,
// and the function is only directly usable from C++.
// C++ function declaration with a default message
extern "C" void print_message(
    const std::string& message = DEFAULT_MESSAGE
) {
  // Print the message
  std::cout << message << std::endl;
}